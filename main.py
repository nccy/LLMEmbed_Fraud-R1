import argparse
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from DownstreamModel import DownstreamModel
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from model_op import Train, Test
from model_op_multi import Train_multi, Test_multi
import torch
from MyDataset import MyDataset
import json
import sys
from datetime import datetime
import random
import numpy as np

ALLOWED_CUDA_IDS = {0, 1, 2}

class TeeLogger:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def __getattr__(self, name):
        return getattr(self.streams[0], name)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cuda_no', type=int)
    parser.add_argument('task', type=str)
    parser.add_argument('epoches', type=int)
    parser.add_argument('SIGMA', type=float)
    parser.add_argument('batch_size', type=int, nargs='?', default=1024)
    parser.add_argument('lr', type=float, nargs='?', default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    validate_cuda_id(args.cuda_no)
    device = f'cuda:{args.cuda_no}'
    task = args.task
    epoches = args.epoches
    SIGMA = args.SIGMA
    batch_size = args.batch_size
    lr = args.lr
    seed = args.seed
    set_seed(seed)

    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'{task}_{log_time}.log')
    log_file = open(log_path, 'a', encoding='utf-8')
    sys.stdout = TeeLogger(sys.stdout, log_file)
    sys.stderr = TeeLogger(sys.stderr, log_file)
    print(f'log file: {log_path}')
    print(f'args: {vars(args)}')
    print(f'seed: {seed}')

    # 加入你的 task_a (二分类) 和 task_b (多分类)
    # 注意：这里的 5 是我随便写的一个占位符，请务必把它改成你刚才跑 Python 预处理脚本时，终端最后打印出来的“任务 B 一共有 X 个诈骗类别”的那个具体数字！
    class_num = {'sst2': 2, 'mr': 2, 'agnews': 4, 'r8': 8, 'r52': 52, 'fraud_binary': 2, 'fraud_multi': 7}
    class_num = class_num[task]

    l_dataset_path = f'llama2_embedding/{task}/dataset_tensor/'
    b_dataset_path = f'bert_embedding/{task}/dataset_tensor/'
    r_dataset_path = f'roberta_embedding/{task}/dataset_tensor/'
    mode = 'train'
    train_data = MyDataset(mode, l_dataset_path, b_dataset_path, r_dataset_path)  
    data_generator = torch.Generator()
    data_generator.manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=data_generator)
    mode = 'test'
    test_data = MyDataset(mode, l_dataset_path, b_dataset_path, r_dataset_path)   
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=data_generator)

    model = DownstreamModel(class_num, SIGMA).to(device)

    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr)

    if class_num == 2:
        print('training ...')
        for epoch in range(epoches):
            model = model.to(device)
            print(f'--------------------------- epoch {epoch} ---------------------------')
            Train(train_loader, device, model, loss_fn, optimizer)
        print()
        print('evaluate ...')
        Test(test_loader, device, model, loss_fn)
        
    # multi-class
    elif class_num > 2:
        print('training ...')
        for epoch in range(epoches):
            model = model.to(device)
            print(f'--------------------------- epoch {epoch} ---------------------------')
            Train_multi(train_loader, device, model, loss_fn, optimizer)
        print()
        print('evaluate ...')
        Test_multi(test_loader, device, model, loss_fn)

    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'{task}_{log_time}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),
        'task': task,
        'class_num': class_num,
        'SIGMA': SIGMA,
        'seed': seed,
        'epoches': epoches,
        'batch_size': batch_size,
        'lr': lr,
    }, checkpoint_path)
    print(f'checkpoint saved: {checkpoint_path}')
