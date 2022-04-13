import numpy as np

"""
gt_item: The only item the user has seen
pred_items: K items recommended by the model
users: Number of users
"""

def hit_k(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0

def ndcg_k(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return 1 / np.log2(index+2) # +2 because index begins from 0
    return 0

def precision_recall_k(gt_item, pred_items):
    if gt_item in pred_items:
        return 1 / len(pred_items), 1
    return 0, 0 # precision, recall

def map_k(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return 1 / (index + 1)
    return 0

def get_metrics_k(gt_item, pred_items):
    """
    return mAP@k, nDCG@k, HR@k
    """
    
    metrics_k = np.zeros(3, dtype=float)
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        metrics_k[0] = 1 / (index + 1)
        metrics_k[1] = 1 / np.log2(index+2)
        metrics_k[2] = 1
    return metrics_k


def get_metrics_k_from_rank(rank, k):
    """
    return mAP@k, nDCG@k, HR@k
    """
    
    metrics_k = np.zeros(3, dtype=float)
    if rank < k:
        metrics_k[0] = 1 / (rank + 1)
        metrics_k[1] = 1 / np.log2(rank+2)
        metrics_k[2] = 1
    return metrics_k