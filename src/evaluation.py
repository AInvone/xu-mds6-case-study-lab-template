import numpy as np
from sklearn.model_selection import cross_val_predict
from .metrics import cls_metrics, reg_metrics

def evaluate_classification_cv(model, X, y, cv):
    """
    Evaluate the classification model using cross-validation.

    Args:
        model (object): the classification model
        X (array-like of shape (n_samples, n_features)): input features
        y (array-like of shape (n_samples,)): target labels
        cv (object): the cross-validation object

    Returns:
        dict: dictionary of metrics
    """
    # cross_val_predict returns out-of-fold predictions
    y_pred = cross_val_predict(model, X, y, cv=cv, method="predict")
    metrics = cls_metrics(y, y_pred)
    return metrics

def evaluate_regression_cv(model, X, y, cv):
    """
    Evaluate the regression model using cross-validation.

    Args:
        model (object): the regression model
        X (array-like of shape (n_samples, n_features)): input features
        y (array-like of shape (n_samples,)): target labels
        cv (object): the cross-validation object

    Returns:
        dict: dictionary of metrics
    """
    y_pred = cross_val_predict(model, X, y, cv=cv)
    metrics = reg_metrics(y, y_pred)
    return metrics

## TODO: for ROC-AUC in CV, need probabilities (method="predict_proba"). 
# Teach it later without overcomplicating baseline lab.
