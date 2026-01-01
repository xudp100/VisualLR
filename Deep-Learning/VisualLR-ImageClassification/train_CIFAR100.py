import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import time
import os
import numpy as np
import warnings
import argparse
from Model.Resnet import ResNet18
from Optimizer.Adam import Adam
from Optimizer.AdaFom import AdaFom
from Optimizer.Yogi import Yogi
warnings.filterwarnings("ignore")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def worker_init_fun(worker_id, seed):
    np.random.seed(seed + worker_id)

def worker_init_fn(worker_id):
    seed = 1111
    worker_init_fun(worker_id, seed)

def build_dataset():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    trainset = torchvision.datasets.CIFAR100(
        root='./Dataset/data_CIFAR100',
        train=True,
        download=True,
        transform=transform_train
    )
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=128,
        shuffle=True,
        num_workers=2,
        worker_init_fn=worker_init_fn
    )
    return train_loader


def select_model(model_name, num_classes=100):
    if model_name == 'ResNet18':
        return ResNet18(num_classes)
    else:
        raise ValueError(f"no: {model_name}")


def select_optimizer(optimizer_name, net):
    if optimizer_name == 'Adam':
        return Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999),
                    weight_decay=0, amsgrad=False, eps=1e-7)
    
    elif optimizer_name == 'AdaFom':
        return AdaFom(net.parameters(), lr=1e-3, betas=(0.9, 0.999),
                    weight_decay=0, amsgrad=False, eps=1e-7)

    elif optimizer_name == 'AMSGrad':
        return Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999),
                    weight_decay=0, amsgrad=True, eps=1e-7)

    elif optimizer_name == 'Yogi':
        return Yogi(net.parameters(), lr=1e-3,
                    weight_decay=0,  eps=1e-7)
    else:
        raise ValueError(f"no: {optimizer_name}")


def train(net, device, data_loader, optimizer, criterion):

    net.train()
    train_loss, correct, total = 0, 0, 0

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        step_result = optimizer.step()

        stats = None
        if isinstance(step_result, tuple) and len(step_result) == 2:
            _, stats = step_result
        elif isinstance(step_result, dict):
            stats = step_result

        if stats:
            max_lr = stats.get('adaptive_lr_max')
            min_lr = stats.get('adaptive_lr_min')
            print(f"Batch [{batch_idx}] -> Max LR: {max_lr}, Min LR: {min_lr}")

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    avg_train_loss = train_loss / len(data_loader)


    print(f'Train accuracy: {accuracy:.3f}%, Loss: {avg_train_loss:.4f}')
    return avg_train_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train CIFAR-100")
    parser.add_argument('--model', default='ResNet18', type=str)
    parser.add_argument('--epochs', default=60, type=int)

    args = parser.parse_args()

    set_seed(1111)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, test_loader = build_dataset()

    optimizers = ['Adam']

    for optimizer_name in optimizers:
        set_seed(1111)
        print(f"====== Using model: {args.model} with optimizer: {optimizer_name} ======")
        net = select_model(args.model, num_classes=100).to(device)

        optimizer = select_optimizer(optimizer_name, net)
        criterion = nn.CrossEntropyLoss().to(device)

        start_epoch = 0
        for epoch in range(start_epoch, args.epochs):
            print(f"------Epoch {epoch}------")
            start_time = time.time()
            train(net, device, train_loader, optimizer, criterion)
            elapsed_time = (time.time() - start_time) / 60
            print(f"Epoch {epoch} finished in {elapsed_time:.2f} minutes")

if __name__ == '__main__':
    main()