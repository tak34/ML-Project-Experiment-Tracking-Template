import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from .augmentation import get_transforms

class ISICDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.file_names = df['file_path'].values
        self.targets = df['target'].values
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path = self.file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = self.targets[index]
        
        if self.transforms:
            img = self.transforms(image=img)["image"]
            
        return {
            'image': img,
            'target': target
        }

def downsample_dataframe(df, positive_ratio=0.5, random_state=42):
    """
    Downsample the majority class to achieve the specified positive ratio.
    """
    positive_samples = df[df['target'] == 1]
    negative_samples = df[df['target'] == 0]
    
    n_positive = len(positive_samples)
    n_negative = int(n_positive * (1 - positive_ratio) / positive_ratio)
    
    downsampled_negative = negative_samples.sample(n=n_negative, random_state=random_state)
    
    return pd.concat([positive_samples, downsampled_negative]).reset_index(drop=True)

def prepare_loaders(df, fold, cfg):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    # Downsample training data
    df_train = downsample_dataframe(df_train, positive_ratio=cfg.data.positive_ratio)
    
    # Optionally downsample validation data
    if cfg.data.downsample_valid:
        df_valid = downsample_dataframe(df_valid, positive_ratio=cfg.data.positive_ratio)
    
    train_dataset = ISICDataset(df_train, transforms=get_transforms(data='train', cfg=cfg))
    valid_dataset = ISICDataset(df_valid, transforms=get_transforms(data='valid', cfg=cfg))

    train_loader = DataLoader(train_dataset, batch_size=cfg.train_batch_size, 
                              num_workers=cfg.num_workers, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.valid_batch_size, 
                              num_workers=cfg.num_workers, shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader


def create_folds(df, cfg):
    skf = StratifiedKFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed)
    for fold, (_, val_idx) in enumerate(skf.split(X=df, y=df.target)):
        df.loc[val_idx, 'kfold'] = fold
    return df