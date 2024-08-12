import os
import random
import numpy as np
import torch
from torch.optim import lr_scheduler

def set_seed(seed=42):
    """
    シードを設定して再現性を確保する
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """
    利用可能なデバイス（GPU/CPU）を取得する
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def fetch_scheduler(optimizer, cfg):
    """
    設定に基づいてスケジューラーを取得する
    """
    if cfg.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.T_max,
            eta_min=cfg.min_lr
        )
    elif cfg.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.T_0,
            eta_min=cfg.min_lr
        )
    elif cfg.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=cfg.factor,
            patience=cfg.patience,
            verbose=True,
            eps=cfg.eps
        )
    elif cfg.scheduler == 'ExponentialLR':
        scheduler = lr_scheduler.ExponentialLR(
            optimizer,
            gamma=cfg.gamma
        )
    elif cfg.scheduler == None:
        return None
        
    return scheduler

def save_model(model, path):
    """
    モデルを保存する
    """
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """
    保存されたモデルを読み込む
    """
    model.load_state_dict(torch.load(path))
    return model

def count_parameters(model):
    """
    モデルのパラメータ数を数える
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_lr(optimizer):
    """
    現在の学習率を取得する
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

class AverageMeter:
    """
    平均値を計算し、保持するクラス
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
