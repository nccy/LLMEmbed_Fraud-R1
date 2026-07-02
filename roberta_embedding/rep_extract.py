# -*- coding: utf-8 -*-
import os
import torch
import json
from transformers import RobertaTokenizer, RobertaModel
from tqdm import trange
from datasets import load_dataset
import argparse
from datasets import load_from_disk

ALLOWED_CUDA_IDS = {0, 1, 2}
ROBERTA_MODEL_PATH = os.environ.get('ROBERTA_MODEL_PATH', 'hf_model/roberta-large')
DATASET_ROOT = os.environ.get('DATASET_ROOT', 'dataset')
JSON_DATASET_ROOT = os.environ.get('JSON_DATASET_ROOT', DATASET_ROOT)

def validate_cuda_id(cuda_no):
    if cuda_no not in ALLOWED_CUDA_IDS:
        allowed = ', '.join(str(item) for item in sorted(ALLOWED_CUDA_IDS))
        raise ValueError(f'GPU id {cuda_no} is not allowed; use one of: {allowed}')

def rep_extract(task, mode, device, sents, labels):
    tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_MODEL_PATH)
    model = RobertaModel.from_pretrained(ROBERTA_MODEL_PATH).to(device)
    model.eval()

    max_len = 512
    sents_reps = []
    step = 512
    for idx in trange(0, len(sents), step):
        idx_end = idx + step
        if idx_end > len(sents):
            idx_end = len(sents)        
        sents_batch = sents[idx: idx_end]

        sents_batch_encoding = tokenizer(sents_batch, return_tensors='pt', max_length=max_len, padding="max_length", truncation=True)
        sents_batch_encoding = sents_batch_encoding.to(device)
        
        with torch.no_grad():
            batch_outputs = model(**sents_batch_encoding)
            reps_batch = batch_outputs.last_hidden_state[:, 0, :]  
        sents_reps.append(reps_batch.cpu())
    sents_reps = torch.cat(sents_reps)
    
    #源代码报错
    # for idx in range(len(labels)):
    #     labels[idx] = torch.tensor(labels[idx])
    # labels = torch.stack(labels)

    labels = torch.tensor(list(labels))
    
    print(sents_reps.shape)
    print(labels.shape)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), task, 'dataset_tensor')
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(sents_reps.to('cpu'), os.path.join(path, f'{mode}_sents.pt'))
    torch.save(labels, os.path.join(path, f'{mode}_labels.pt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cuda_no', type=int)
    parser.add_argument('task', type=str)   # sst2, mr, agnews, r8, r52
    args = parser.parse_args()
    validate_cuda_id(args.cuda_no)
    device = f'cuda:{args.cuda_no}'
    task = args.task
    
    if task == 'sst2':
        dataset = load_from_disk(os.path.join(DATASET_ROOT, "sst2"))
        
        sents = dataset['train']['sentence']
        labels = dataset['train']['label']
        rep_extract(task, 'train', device, sents, labels)
        
        sents = dataset['validation']['sentence']
        labels = dataset['validation']['label']
        rep_extract(task, 'test', device, sents, labels)

    elif task == 'mr':
        path = os.path.join(JSON_DATASET_ROOT, "MR", "train.json")
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        sents = dataset['text']
        labels = dataset['label']
        rep_extract(task, 'train', device, sents, labels)

        path = os.path.join(JSON_DATASET_ROOT, "MR", "test.json")
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        sents = dataset['text']
        labels = dataset['label']
        rep_extract(task, 'test', device, sents, labels)

    elif task == 'agnews':
        dataset = load_from_disk(os.path.join(DATASET_ROOT, "agnews"))
        
        sents = dataset['train']['text']
        labels = dataset['train']['label']
        rep_extract(task, 'train', device, sents, labels)

        sents = dataset['test']['text']
        labels = dataset['test']['label']
        rep_extract(task, 'test', device, sents, labels)

    elif task == 'r8':
        path = os.path.join(JSON_DATASET_ROOT, "R8", "train.json")
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        sents = dataset['text']
        labels = dataset['label']
        rep_extract(task, 'train', device, sents, labels)

        path = os.path.join(JSON_DATASET_ROOT, "R8", "test.json")
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        sents = dataset['text']
        labels = dataset['label']
        rep_extract(task, 'test', device, sents, labels)

    elif task == 'r52':
        path = os.path.join(JSON_DATASET_ROOT, "R52", "train.json")
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        sents = dataset['text']
        labels = dataset['label']
        rep_extract(task, 'train', device, sents, labels)

        path = os.path.join(JSON_DATASET_ROOT, "R52", "test.json")
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        sents = dataset['text']
        labels = dataset['label']
        rep_extract(task, 'test', device, sents, labels)

    elif task == 'fraud_binary':

        dataset = load_from_disk(os.path.join(DATASET_ROOT, "fraud_binary"))

        sents = dataset['train']['text']
        labels = dataset['train']['label']
        rep_extract(task, 'train', device, sents, labels)

        sents = dataset['test']['text']
        labels = dataset['test']['label']
        rep_extract(task, 'test', device, sents, labels)

    elif task == 'fraud_multi':

        dataset = load_from_disk(os.path.join(DATASET_ROOT, "fraud_multi"))

        sents = dataset['train']['text']
        labels = dataset['train']['label']
        rep_extract(task, 'train', device, sents, labels)

        sents = dataset['test']['text']
        labels = dataset['test']['label']
        rep_extract(task, 'test', device, sents, labels)
