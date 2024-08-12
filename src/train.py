import os
import wandb
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from src.data.dataset import prepare_loaders, create_folds
from src.models.model import ISICModel
from src.utils.utils import set_seed, fetch_scheduler
from src.utils.metrics import comp_score_list
from src.utils.visualization import plot_learning_curves, plot_metrics_summary

def criterion(outputs, targets):
    return nn.BCELoss()(outputs, targets)
#     return nn.CrossEntropyLoss()(outputs, targets) 

def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch, cfg):
    model.train()
    
    dataset_size = 0
    running_loss = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data['image'].to(device, dtype=torch.float)
        targets = data['target'].to(device, dtype=torch.float)
        
        batch_size = images.size(0)
        
        # Mixupを設定されたエポック数まで適用
        if epoch <= CONFIG['mixup_epochs']:
            mixed_images, targets_a, targets_b, lam = mixup(images, targets, alpha=CONFIG['mixup_alpha'])
            outputs = model(mixed_images).squeeze()
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        else:
            outputs = model(images).squeeze()
            loss = criterion(outputs, targets)
            
        loss = loss / CONFIG['n_accumulate']
            
        loss.backward()
    
        if (step + 1) % CONFIG['n_accumulate'] == 0:
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])
    
    gc.collect()
    
    return epoch_loss


@torch.inference_mode()
def valid_one_epoch(model, dataloader, device, epoch, cfg):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    running_auroc = 0.0
    
    # TTAのためのカウンター
    tta_counter = 0
    
    # TTAのための関数を定義
    def apply_tta(model, image):
        outputs = []
        
        # オリジナル画像
        outputs.append(model(image).squeeze())
        
        # 水平フリップ
        outputs.append(model(torch.flip(image, dims=[3])).squeeze())
        
        # 垂直フリップ
        outputs.append(model(torch.flip(image, dims=[2])).squeeze())
        
        # 90度、180度、270度回転
        for k in [1, 2, 3]:
            outputs.append(model(torch.rot90(image, k, dims=[2,3])).squeeze())
        
        return torch.stack(outputs).mean(dim=0)
    
    all_outputs = []
    all_targets = []
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:        
        images = data['image'].to(device, dtype=torch.float)
        targets = data['target'].to(device, dtype=torch.float)
        
        batch_size = images.size(0)

        if CONFIG['use_tta']:
            outputs = apply_tta(model, images).squeeze()
        else:
            outputs = model(images).squeeze()
        loss = criterion(outputs, targets)
        
        all_outputs.extend(outputs.detach().cpu().numpy().flatten())  # 確率に変換して蓄積
        all_targets.extend(targets.detach().cpu().numpy().flatten())  # ラベルを蓄積

#         auroc = binary_auroc(input=outputs.squeeze(), target=targets).item()
        running_loss += (loss.item() * batch_size)
#         running_auroc  += (auroc * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
#         epoch_auroc = running_auroc / dataset_size
        
        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss, 
                        LR=optimizer.param_groups[0]['lr'])   
    
    # epoch毎にauroc算出
    epoch_auroc = comp_score_list(all_targets, all_outputs)
    
    gc.collect()
    
    return epoch_loss, epoch_auroc, all_outputs

def run_training(cfg):
    set_seed(cfg.seed)
    
    if cfg.wandb.use:
        os.environ["WANDB_API_KEY"] = cfg.wandb.api_key
        wandb.init(entity=cfg.wandb.entity,
                   project=cfg.wandb.project,
                   name=cfg.wandb.name,
                   config=cfg)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # データの読み込みと前処理
    df = pd.read_csv(cfg.data.train_metadata)
    df = create_folds(df, cfg)  # Foldの作成

    fold_scores = []
    
    for fold in range(cfg.n_fold):
        print(f"{'='*20} Fold: {fold} {'='*20}")
        
        model = ISICModel(cfg.model.name, pretrained=True)
        model.to(device)
        
        train_loader, valid_loader = prepare_loaders(df, fold, cfg)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, 
                                     weight_decay=cfg.weight_decay)
        scheduler = fetch_scheduler(optimizer, cfg)
        
        best_score = 0
        best_epoch = 0
        
        for epoch in range(1, cfg.epochs + 1):
            train_loss = train_one_epoch(model, optimizer, scheduler, train_loader, device, epoch, cfg)
            val_loss, val_score, val_preds = valid_one_epoch(model, valid_loader, device, epoch, cfg)
            
            if val_score > best_score:
                best_score = val_score
                best_epoch = epoch
                torch.save(model.state_dict(), f"{cfg.save_dir}/best_model_fold{fold}.pth")
            
            print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Score: {val_score:.4f}")
            
            if cfg.wandb.use:
                wandb.log({
                    f"fold_{fold}_train_loss": train_loss,
                    f"fold_{fold}_val_loss": val_loss,
                    f"fold_{fold}_val_score": val_score,
                    f"fold_{fold}_lr": scheduler.get_last_lr()[0]
                })
            history = {
                "Train Loss": train_losses,
                "Valid Loss": valid_losses,
                "Valid pAUC80": valid_pauc80s,
                "lr": learning_rates
            }
            list_history.append(history)
        
        print(f"Best Score for Fold {fold}: {best_score:.4f} at epoch {best_epoch}")
        fold_scores.append(best_score)
    
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    print(f"Mean Score: {mean_score:.4f} ± {std_score:.4f}")
    
    if cfg.wandb.use:
        wandb.log({
            "mean_score": mean_score,
            "std_score": std_score
        })
        wandb.finish()

    # 学習曲線の描画と保存
    save_dir = os.path.join(cfg.save_dir, "plots")
    os.makedirs(save_dir, exist_ok=True)
    plot_learning_curves(list_history, save_dir, cfg.experiment_name)
    plot_metrics_summary(list_history, save_dir, cfg.experiment_name)