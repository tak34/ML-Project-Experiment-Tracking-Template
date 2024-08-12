import matplotlib.pyplot as plt
import numpy as np
import os

def plot_learning_curves(list_history, save_dir, experiment_name):
    """
    学習曲線を描画し、指定されたフォルダに保存する関数
    
    Args:
    list_history (list): 各foldの学習履歴を含むリスト
    save_dir (str): 画像を保存するディレクトリパス
    experiment_name (str): 実験名（ファイル名の一部として使用）
    """
    # Loss curves
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    for i, history in enumerate(list_history):
        axes[0].plot(history["Train Loss"], label=f"fold_{i}")
        axes[1].plot(history["Valid Loss"], label=f"fold_{i}")
    
    axes[0].set_title("Learning Curve (Train Loss)")
    axes[0].set_ylabel("Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    
    axes[1].set_title("Learning Curve (Valid Loss)")
    axes[1].set_ylabel("Loss")
    axes[1].set_xlabel("Epochs")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{experiment_name}_loss_curves.png"))
    plt.close()
    
    # pAUC80 curve
    plt.figure(figsize=(10, 5))
    for i, history in enumerate(list_history):
        plt.plot(history["Valid pAUC80"], label=f"fold_{i}")
    
    plt.title("Learning Curve (Valid pAUC80)")
    plt.ylabel("pAUC80")
    plt.xlabel("Epochs")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.ylim([0, 0.2])  # pAUC80の範囲に合わせて調整
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{experiment_name}_pauc80_curve.png"))
    plt.close()
    
    # Learning rate curve
    plt.figure(figsize=(10, 5))
    for i, history in enumerate(list_history):
        plt.plot(history["lr"], label=f"fold_{i}")
    
    plt.title("Learning Rate")
    plt.ylabel("Learning Rate")
    plt.xlabel("Epochs")
    plt.legend()
    plt.grid(alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{experiment_name}_lr_curve.png"))
    plt.close()

def plot_metrics_summary(list_history, save_dir, experiment_name):
    """
    各foldの最終的なメトリクスをプロットし、保存する関数
    
    Args:
    list_history (list): 各foldの学習履歴を含むリスト
    save_dir (str): 画像を保存するディレクトリパス
    experiment_name (str): 実験名（ファイル名の一部として使用）
    """
    final_pauc80 = [history["Valid pAUC80"][-1] for history in list_history]
    
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(final_pauc80)), final_pauc80)
    plt.title("Final pAUC80 Scores by Fold")
    plt.xlabel("Fold")
    plt.ylabel("pAUC80")
    plt.xticks(range(len(final_pauc80)), [f"Fold {i}" for i in range(len(final_pauc80))])
    
    mean_pauc80 = np.mean(final_pauc80)
    std_pauc80 = np.std(final_pauc80)
    plt.axhline(y=mean_pauc80, color='r', linestyle='--', label=f'Mean: {mean_pauc80:.4f}')
    plt.text(len(final_pauc80)-1, mean_pauc80, f'Mean: {mean_pauc80:.4f} ± {std_pauc80:.4f}', 
             verticalalignment='bottom', horizontalalignment='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{experiment_name}_final_pauc80_summary.png"))
    plt.close()