import torch
from torch import nn

import wandb
from tqdm import tqdm

from dataloader import Load_Dataset
from train_utils import train, vaild
from net import ConvNet
from utils import build_optimizer
from make_config import *

DEVICE = torch.device('cuda')

def run(config=None):
    wandb.init(config=config)
    w_config = wandb.config

    criterion = nn.CrossEntropyLoss()
    train_loader, vaild_loader = Load_Dataset(w_config.batch_size)
    model = ConvNet().to(DEVICE)
    optimizer = build_optimizer(model, w_config.optimizer, w_config.learning_rate)

    for epoch in tqdm(range(w_config.epochs), desc='EPOCH'):
        train(model, train_loader, criterion, optimizer, DEVICE, w_config, wandb, epoch)
        vaild(model, vaild_loader, criterion, DEVICE, wandb, epoch)

if __name__ == '__main__':
    sweep_config = sweep_config
    sweep_id = wandb.sweep(sweep_config, project="sweep_tutorial", entity='hmpark1995')
    wandb.agent(sweep_id, run)
    # default_config = hyperparameter_defaults
    # run(default_config)

