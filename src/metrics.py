from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
import numpy as np

def cls_metrics(y_true, y_pred, y_proba=None):
    """
    Calculate the classification metrics.

    Args:
        y_true (array-like of shape (n_samples,)): actual labels
        y_pred (array-like of shape (n_samples,)): predicted labels
        y_proba (array-like of shape (n_samples,), optional): predicted probabilities of the positive class. Defaults to None.

    Returns:
        dict: dictionary of metrics
    """
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        out["roc_auc"] = roc_auc_score(y_true, y_proba)
    return out

def reg_metrics(y_true, y_pred):
    """
    Calculate the regression metrics. 
    
    Args:
        y_true (array-like of shape (n_samples,)): actual labels
        y_pred (array-like of shape (n_samples,)): predicted labels

    Returns:
        dict: dictionary of metrics
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse,
        "r2": r2_score(y_true, y_pred),
    }
    