import argparse
import json
import os
import random

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from DownstreamModel import DownstreamModel
from MyDataset import MyDataset

ALLOWED_CUDA_IDS = {0, 1, 2}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def validate_cuda_id(cuda_no):
    if cuda_no not in ALLOWED_CUDA_IDS:
        allowed = ', '.join(str(item) for item in sorted(ALLOWED_CUDA_IDS))
        raise ValueError(f'GPU id {cuda_no} is not allowed; use one of: {allowed}')


def evaluate(dataloader, device, model, loss_fn, class_num):
    avg_loss = 0
    total_pred, total_y, total_prob = [], [], []

    model.eval()
    for batch_i, batch_loader in enumerate(tqdm(dataloader)):
        batch_l, batch_b, batch_r, batch_y = batch_loader
        batch_l = batch_l.to(device)
        batch_b = batch_b.to(device)
        batch_r = batch_r.to(device)
        batch_y = batch_y.to(device)

        with torch.no_grad():
            pred = model(batch_l.float(), batch_b.float(), batch_r.float())
            loss = loss_fn(pred, batch_y)
            avg_loss += loss.to('cpu').item()

        pred_y = torch.max(pred, 1).indices
        total_prob.append(pred.cpu())
        total_pred.append(pred_y.cpu())
        total_y.append(batch_y.cpu())

    avg_loss = avg_loss / (batch_i + 1)
    total_y = torch.cat(total_y)
    total_pred = torch.cat(total_pred)
    total_prob = torch.cat(total_prob)

    metrics = {
        'avg_loss': avg_loss,
        'acc': accuracy_score(total_y, total_pred),
    }
    if class_num == 2:
        metrics['precision'] = precision_score(total_y, total_pred)
        metrics['recall'] = recall_score(total_y, total_pred)
        metrics['f1'] = f1_score(total_y, total_pred)
    else:
        metrics['micro_f1'] = f1_score(total_y, total_pred, average='micro')
        metrics['macro_f1'] = f1_score(total_y, total_pred, average='macro')
        metrics['macro_precision'] = precision_score(total_y, total_pred, average='macro', zero_division=0)
        metrics['macro_recall'] = recall_score(total_y, total_pred, average='macro', zero_division=0)

    return metrics, total_y, total_pred, total_prob


def save_predictions(output_path, labels, preds, probs):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    records = []
    for idx, (label, pred, prob) in enumerate(zip(labels.tolist(), preds.tolist(), probs.tolist())):
        records.append({
            'index': idx,
            'label': label,
            'prediction': pred,
            'probabilities': prob,
        })
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path', type=str)
    parser.add_argument('cuda_no', type=int)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--prediction_output', type=str, default=None)
    args = parser.parse_args()

    validate_cuda_id(args.cuda_no)
    device = f'cuda:{args.cuda_no}'
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    task = args.task or checkpoint['task']
    class_num = checkpoint['class_num']
    sigma = checkpoint['SIGMA']
    seed = checkpoint.get('seed', 42)
    batch_size = args.batch_size or checkpoint.get('batch_size', 1024)

    set_seed(seed)

    l_dataset_path = f'llama2_embedding/{task}/dataset_tensor/'
    b_dataset_path = f'bert_embedding/{task}/dataset_tensor/'
    r_dataset_path = f'roberta_embedding/{task}/dataset_tensor/'

    dataset = MyDataset(args.split, l_dataset_path, b_dataset_path, r_dataset_path)
    data_generator = torch.Generator()
    data_generator.manual_seed(seed)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=data_generator,
    )

    model = DownstreamModel(class_num, sigma).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    loss_fn = nn.CrossEntropyLoss().to(device)

    print(f'checkpoint: {args.checkpoint_path}')
    print(f'task: {task}')
    print(f'split: {args.split}')
    print(f'seed: {seed}')
    metrics, labels, preds, probs = evaluate(dataloader, device, model, loss_fn, class_num)
    for key, value in metrics.items():
        print(f'{key}: {value:.4f}')

    if args.prediction_output:
        save_predictions(args.prediction_output, labels, preds, probs)
        print(f'predictions saved: {args.prediction_output}')
