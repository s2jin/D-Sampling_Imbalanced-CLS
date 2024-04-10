import os
import sys
import copy
import json
from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf
import logging
logging.basicConfig(level=logging.INFO)
logging = logging.getLogger(__name__)

import torch
import transformers

class Agent():
    def __init__(self,
            **kwargs
        ):
        self.config = DictConfig(kwargs)
        self.training_args = OmegaConf.to_container(self.config, resolve=True)

        model = hydra.utils.get_class(self.config.model._target_)

        ## Setting
        self.tokenizer = model.set_tokenizer(self.config.tokenizer) ## set tokenizer
        self.set_data(self.config.mode) ## set data and labels
        self.model = model(tokenizer=self.tokenizer, **self.config.model) ## set model
        self.optimizer = self.model.configure_optimizers(**self.config.optimizer)
                
    def run(self):
        if self.config.mode == 'train':
            self.fit()
        elif self.config.mode == 'predict':
            self.predict()
        else:
            raise NotImplementedError('OPTION "{}" is not supported'.format(self.config.mode))


	######## SETTING #######################################################

    def set_dataloader(self, config, **kwargs):
        ## source file: src/datamodule/
        config = dict(config)
        config['data_path'] = kwargs['data_path']
        config['tokenizer'] = kwargs['tokenizer']

        datamodule = hydra.utils.instantiate(config)
        dataloader = datamodule.get_dataloader()
        return dataloader

    def set_data(self, mode):
        def set_dataloader(target_file):
            if os.path.isfile(target_file):
                filename = target_file
            else:
                filename = os.path.join(self.config.work_dir, self.config.datamodule.data_dir, target_file)
            return self.set_dataloader(self.config.datamodule, tokenizer=self.tokenizer, data_path=filename)

        if mode in ['train']:
            self.train_dataloader = set_dataloader(self.config.datamodule.train_data)
            self.valid_dataloader = set_dataloader(self.config.datamodule.valid_data)
            #self.label_list = self.train_dataloader.dataset.get_label_list() ## for classification
            if self.config.agent.predict_after_training or self.config.agent.predict_after_all_training:
                self.test_dataloader = set_dataloader(self.config.datamodule.test_data)

        elif mode in ['predict']:
            self.test_dataloader = set_dataloader(self.config.datamodule.test_data)
            #self.label_list = self.test_dataloader.dataset.get_label_list() ## for classification

    ########################################################################

    def get_label_freq(self, data):
        labels = [d['labels'] for d in data]
        freq = dict()
        for lbl in labels:
            if lbl in freq: freq[lbl] += 1
            else: freq[lbl] = 1
        return labels, freq

    def get_sample_weight(self, labels, freq, gamma=1):
        key = sorted(freq.keys())
        value = [1-(freq[k]/len(labels)) for k in freq]
        value = torch.softmax(torch.tensor(value)/gamma, dim=0).tolist() ## softmax
        weights = dict(zip(key, value))
        weights = [weights[d] for d in labels]
        return weights

    
    def fit(self): ## TRAINING

        earlystop_threshold = self.config.agent.patience
        patience = 0
        max_loss = 999
        self.best_model = None

        ## set batch sampling prob
        labels, freq = self.get_label_freq(self.train_dataloader.dataset.data)
        weights = self.get_sample_weight(labels, freq, self.config.agent.batch_weight_gamma)
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(labels), replacement=True)

        for epoch in range(self.config.agent.epochs):
            print('',flush=True)

            if epoch >= self.config.agent.warmup and self.train_dataloader.sampler != sampler:
                self.train_dataloader = self.train_dataloader.dataset.get_dataloader(sampler=sampler)
                patience = 0
            logging.info(f"LOAD sampler {self.train_dataloader.sampler.__class__.__name__}")
            
            ## training step
            self.model.train()
            tr_loss = tr_acc = 0
            dataloader = tqdm(self.train_dataloader)#, ascii=True)
            batch_storage = list()
            for index, batch in enumerate(dataloader): ## 1 epoch
                batch_storage.append(batch['data'])
                output = self.model.training_step(batch)

                loss = output.pop('loss', None)
                acc = output.pop('acc', None)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)

                self.optimizer.step()
                self.optimizer.zero_grad()

                tr_loss += loss.item()
                tr_acc += acc.item()

                process_description = f"[TRAIN]_Epoch{epoch}_P{patience}_L{tr_loss/(index+1):.3f}_A{tr_acc/(index+1):.3f}_time{dataloader.format_dict['elapsed']:.4f}_"
                logging.debug(process_description+f'step{index}_')
                dataloader.set_description(process_description)
            logging.info(process_description)

            batch_save_path = os.path.join( self.config.checkpoint_path, 'batch_samples')
            os.makedirs(batch_save_path, exist_ok=True)
            with open(os.path.join(batch_save_path, f'{epoch:04d}_{self.train_dataloader.sampler.__class__.__name__}.jsonl'), 'w') as f:
                for b in batch_storage:
                    f.write(json.dumps(b, ensure_ascii=False)+'\n')

            ## validation step
            self.model.eval()
            val_loss = val_acc = 0
            dataloader = tqdm(self.valid_dataloader)#, ascii=True)
            for index, batch in enumerate(dataloader): ## 1 epoch
                output = self.model.validation_step(batch)

                loss = output.pop('loss', None)
                acc = output.pop('acc', None)

                val_loss += loss.item()
                val_acc += acc.item()

                process_description = f"[VALID]_Epoch{epoch}_P{patience}_L{val_loss/(index+1):.3f}_A{val_acc/(index+1):.3f}_time{dataloader.format_dict['elapsed']:.4f}_"
                logging.debug(process_description+f'step{index}_')
                dataloader.set_description(process_description)
            logging.info(process_description)

            if self.config.agent.model_all_save:
                path = os.path.join(
                        self.config.checkpoint_path,
                        f"train_{self.config.model.name}" +\
                        f"_lr_{self.config.optimizer.lr:.0E}" +\
                        f"_batch_{self.config.datamodule.batch_size}" +\
                        f"_pat_{self.config.agent.patience}" +\
                        f"_epoch_{epoch:07d}" +\
                        f"_loss_{val_loss/(index+1):.4f}" +\
                        f"_acc_{val_acc/(index+1):.4f}"
                        )
                self.model.save_model(path)
                if self.config.agent.predict_after_all_training: self.predict(dir_path=path)

            ## earlystop
            if epoch < self.config.agent.warmup: continue
            val_loss = val_loss/(index+1)
            val_acc = val_acc/(index+1)
            if val_loss < max_loss:
                max_loss = val_loss
                patience = 0
                path = os.path.join(
                        self.config.checkpoint_path,
                        f"valid_{self.config.model.name}" +\
                        f"_lr_{self.config.optimizer.lr:.0E}" +\
                        f"_batch_{self.config.datamodule.batch_size}" +\
                        f"_pat_{self.config.agent.patience}" +\
                        f"_epoch_{epoch:07d}" +\
                        f"_loss_{val_loss:.4f}"+\
                        f"_acc_{val_acc:.4f}"
                        )
                self.model.save_model(path)
                if self.config.agent.predict_after_all_training: self.predict(dir_path=path)
                path = os.path.join( self.config.checkpoint_path, "trained_model" )
                self.model.save_model(path)
                self.best_model = copy.deepcopy(self.model)
            else:
                patience += 1
                if patience > earlystop_threshold:
                    logging.info('Ran out of patience.')
                    break ## STOP training

        ## predict best model
        if self.config.agent.predict_after_training or self.config.agent.predict_after_all_training:
            path = os.path.join( self.config.checkpoint_path, "trained_model" )
            self.model = copy.deepcopy(self.best_model)
            self.predict(dir_path=path)

    
    def predict(self, dir_path=None):
        self.model.eval()
        if not dir_path:
            dir_path = self.config.model.path

        if self.config.mode == 'train':
            if not self.config.predict_file_path:
                raise ValueError('No empty predict_file_path supported. Make sure predict_file_path has a value.')
            ofp_name = os.path.join( dir_path, self.config.predict_file_path )
            ofp = open(ofp_name,'w')
            logging.info(f"WRITE {ofp_name}")
        elif self.config.predict_file_path:
            ofp_name = os.path.join( dir_path, self.config.predict_file_path )
            ofp = open(ofp_name,'w')
            logging.info(f"WRITE {ofp_name}")
        else:
            ofp = sys.stdout
            logging.info(f"WRITE sys.stdout")

        dataloader = tqdm(self.test_dataloader)
        for index, batch in enumerate(dataloader): ## 1 epoch
            output = self.model.predict_step(batch)

            for out in output:
                ofp.write(json.dumps(out, ensure_ascii=False)+'\n')
            dataloader.set_description(f"[PREDICT]")

        config = OmegaConf.to_container(self.config, resolve=True)
        ofp.write(json.dumps(config, ensure_ascii=False)+'\n')
        ofp.close()

