import os
import re
import copy
import json
import pandas as pd
from omegaconf import DictConfig

import logging
logging.basicConfig(level=logging.INFO)
logging = logging.getLogger(__name__)

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from .base import DataModuleBase 

class DataModule(DataModuleBase):
    def __init__(self, 
            tokenizer=None, 
            data_path=None, 
            **kwargs
        ):
        
        super(DataModule, self).__init__(tokenizer=tokenizer, data_path=data_path, **kwargs)
        self.label_list = self.set_label_list() 

    def load_data(self, filename):
        with open(filename, 'r') as f:
            data = [json.loads(d) for d in f]
        result = list()
        for index, d in enumerate(data):
            result.append({
                'inputs': d['sentence'],
                'labels': str(d['label']),
                'raw': copy.deepcopy(d),
                'index': index,
                })
        return result

    def __getitem__(self, index):
        return self.data[index]

    def collate_fn(self, data):
        result = {
                'inputs': [d['inputs'] for d in data],
                'labels': [d['labels'] for d in data],
                'data': [d['raw'] for d in data],
                'index': [d['index'] for d in data],
                }

        result['inputs'] = self.tokenizer(result['inputs'], max_length=self.config.max_source_length, truncation=True, padding='max_length', return_tensors='pt')
        result['labels'] = torch.tensor([ self.label_list.index(d) for d in result['labels']])

        return result 
