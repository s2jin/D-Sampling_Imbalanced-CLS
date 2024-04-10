import torch
import transformers
from .backbone import tokenization_korscibert

import logging
logging.basicConfig(level=logging.INFO)
logging = logging.getLogger(__name__)

class Tokenizer(tokenization_korscibert.FullTokenizer):
    def __init__(self, 
            path=None,
            do_lower_case=False,
            tokenizer_type="Mecab",
            **kwargs,
            ):
        super().__init__(
                vocab_file=path,
                do_lower_case=do_lower_case,
                tokenizer_type=tokenizer_type,
                )

        self.mask_token = '[MASK]'
        self.mask_token_id = self.vocab[self.mask_token]
        self.pad_token = '[PAD]'
        self.pad_token_id = self.vocab[self.pad_token]
        self.unk_token = '[UNK]'
        self.unk_token_id = self.vocab[self.unk_token]

        self.special_tokens = [self.mask_token, self.pad_token, self.unk_token]

        self.padding_side = 'right'
        self.truncation_side = 'right'

        logging.info(f'LOAD tokenizer: {path}')

    def get_tokenizer(self):
        symbols = '!#$%&\"\'()*+,-./0123456789;:<=>?@'
        hangle = '다람쥐 헌 쳇바퀴에 타고파'
        alphabet  = 'The Quick Brown Fox Jumps Over The Lazy Dog'
        for sample in [symbols, hangle, alphabet]:
            logging.info(f"\nTokenized sample:\n{sample}\n=> {self.tokenize(sample)}")
        return self
        

    def __call__(self, inputs, pair_input=None, max_length=512, padding=True, truncation=True, return_tensors=False):
        result = list()
        for inp in inputs:
            tokenized = self.tokenize(inp)

            dif = abs(max_length - len(tokenized))
            if truncation:
                if len(tokenized) > max_length: 
                    if self.truncation_side == 'left': tokenized = tokenized[dif:]
                    elif self.truncation_side == 'right': tokenized = tokenized[:max_length]
            if padding:
                if len(tokenized) < max_length:
                    if self.padding_side == 'left': tokenized = ([self.pad_token] * dif) + tokenized 
                    elif self.padding_side == 'right': tokenized += [self.pad_token] * dif

            tokenized = self.convert_tokens_to_ids(tokenized)
            result.append(tokenized)
        result = {'input_ids':result}
        if return_tensors == 'pt':
            result = {k:torch.tensor(v) for k,v in result.items()}
        result = transformers.BatchEncoding(result)
        return result
            
            
    def decode(self, inputs, skip_special_tokens=False):
        if type(inputs) == torch.Tensor:
            inputs = inputs.tolist()
        output = self.convert_ids_to_tokens(inputs)
        if skip_special_tokens:
            output = [d for d in output if d not in self.special_tokens]
        output = ' '.join(output)
        output = output.replace(' ##','')
        return output

    def batch_decode(self, inputs, skip_special_tokens=False):
#     input_text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        return [ self.decode(d, skip_special_tokens=skip_special_tokens) for d in inputs]
