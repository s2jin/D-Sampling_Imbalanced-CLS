import json
from datasets import load_dataset

dataset = load_dataset("stanfordnlp/sst2")

valid_size = round(len(dataset['train'])*0.1)
train = [d for d in dataset['train']][:-1*valid_size]
valid = [d for d in dataset['train']][-1*valid_size:]
test = [d for d in dataset['validation']]

with open('training.jsonl','w') as f:
    for item in train:
        f.write(json.dumps(item, ensure_ascii=False)+'\n')

with open('valid.jsonl','w') as f:
    for item in valid:
        f.write(json.dumps(item, ensure_ascii=False)+'\n')

with open('test.jsonl','w') as f:
    for item in test:
        f.write(json.dumps(item, ensure_ascii=False)+'\n')
