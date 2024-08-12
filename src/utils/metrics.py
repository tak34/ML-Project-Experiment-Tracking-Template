import numpy as np
from sklearn.metrics import roc_auc_score

def comp_score_list(solution: list, submission: list, min_tpr: float=0.80):
    """
    ISICチャレンジの評価指標を計算する関数
    
    Args:
    solution (list): 正解ラベル
    submission (list): モデルの予測確率
    min_tpr (float): 最小のTrue Positive Rate（感度）。デフォルトは0.80
    
    Returns:
    partial_auc: 部分的AUCスコア
    """
    v_gt = np.abs(np.array(solution)-1)
    v_pred = np.array([1.0 - x for x in submission])
    max_fpr = abs(1-min_tpr)
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    # スケールを [0.5 * max_fpr**2, max_fpr] から [0.5, 1.0] に変更
    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    return partial_auc