# -*- coding: utf-8 -*-
import os
import torch
import json
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from tqdm import trange
from datasets import load_dataset
import argparse

from datasets import load_from_disk

ALLOWED_CUDA_IDS = {0, 1, 2}
LLAMA2_MODEL_PATH = os.environ.get('LLAMA2_MODEL_PATH', 'hf_model/llama2-7b-chat-hf')
DATASET_ROOT = os.environ.get('DATASET_ROOT', 'dataset')
JSON_DATASET_ROOT = os.environ.get('JSON_DATASET_ROOT', DATASET_ROOT)

def validate_cuda_id(cuda_no):
    if cuda_no not in ALLOWED_CUDA_IDS:
        allowed = ', '.join(str(item) for item in sorted(ALLOWED_CUDA_IDS))
        raise ValueError(f'GPU id {cuda_no} is not allowed; use one of: {allowed}')

def rep_extract(task, mode, device, sents, labels, max_len, step):
    model_id = LLAMA2_MODEL_PATH

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "right"

    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": None,
        "revision": 'main',
        "use_auth_token": None,
        "output_hidden_states": True
    }
    model_config = AutoConfig.from_pretrained(model_id, **config_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        device_map=device,
        torch_dtype=torch.float16)
    model.eval()

    sents_reps = []
    # for idx in trange(0, 20, step):
    for idx in trange(0, len(sents), step):
        idx_end = idx + step
        if idx_end > len(sents):
            idx_end = len(sents)        
        sents_batch = sents[idx: idx_end]

        sents_batch_encoding = tokenizer(sents_batch, return_tensors='pt', max_length=max_len, padding="max_length", truncation=True)
        sents_batch_encoding = sents_batch_encoding.to(device)
        
        with torch.no_grad():
            batch_outputs = model(**sents_batch_encoding)

            reps_batch_5L = []
            for layer in range(-1, -6, -1):
                reps_batch_5L.append(torch.mean(batch_outputs.hidden_states[layer], axis=1))    
            reps_batch_5L = torch.stack(reps_batch_5L, axis=1)

        sents_reps.append(reps_batch_5L.cpu())
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
        rep_extract(task, 'train', device, sents, labels, 128, 16)
        
        sents = dataset['validation']['sentence']
        labels = dataset['validation']['label']
        rep_extract(task, 'test', device, sents, labels, 128, 16)

    elif task == 'mr':
        path = os.path.join(JSON_DATASET_ROOT, "MR", "train.json")
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        sents = dataset['text']
        labels = dataset['label']
        rep_extract(task, 'train', device, sents, labels, 3000, 3)

        path = os.path.join(JSON_DATASET_ROOT, "MR", "test.json")
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        sents = dataset['text']
        labels = dataset['label']
        rep_extract(task, 'test', device, sents, labels, 1500, 7)

    elif task == 'agnews':
        dataset = load_from_disk(os.path.join(DATASET_ROOT, "agnews"))
        
        sents = dataset['train']['text']
        labels = dataset['train']['label']
        rep_extract(task, 'train', device, sents, labels, 256, 8)

        sents = dataset['test']['text']
        labels = dataset['test']['label']
        rep_extract(task, 'test', device, sents, labels, 256, 8)

    elif task == 'r8':
        path = os.path.join(JSON_DATASET_ROOT, "R8", "train.json")
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        sents = dataset['text']
        labels = dataset['label']
        rep_extract(task, 'train', device, sents, labels, 1024, 10)

        path = os.path.join(JSON_DATASET_ROOT, "R8", "test.json")
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        sents = dataset['text']
        labels = dataset['label']
        rep_extract(task, 'test', device, sents, labels, 1024, 10)

    elif task == 'r52':
        path = os.path.join(JSON_DATASET_ROOT, "R52", "train.json")
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        sents = dataset['text']
        labels = dataset['label']
        rep_extract(task, 'train', device, sents, labels, 1024, 10)

        path = os.path.join(JSON_DATASET_ROOT, "R52", "test.json")
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        sents = dataset['text']
        labels = dataset['label']
        rep_extract(task, 'test', device, sents, labels, 1024, 10)

    elif task == 'fraud_binary':

        dataset = load_from_disk(os.path.join(DATASET_ROOT, "fraud_binary"))

        sents = dataset['train']['text']
        labels = dataset['train']['label']
        rep_extract(task, 'train', device, sents, labels, 256, 8)

        sents = dataset['test']['text']
        labels = dataset['test']['label']
        rep_extract(task, 'test', device, sents, labels, 256, 8)
    elif task == 'fraud_multi':

        dataset = load_from_disk(os.path.join(DATASET_ROOT, "fraud_multi"))

        sents = dataset['train']['text']
        labels = dataset['train']['label']
        rep_extract(task, 'train', device, sents, labels, 256, 8)

        sents = dataset['test']['text']
        labels = dataset['test']['label']
        rep_extract(task, 'test', device, sents, labels, 256, 8)
