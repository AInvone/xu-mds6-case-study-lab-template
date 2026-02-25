from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from .config import RANDOM_STATE, TEST_SIZE, N_SPLITS_CV

def split_classification(X, y):
    """
    Split the data into training and test sets.

    Args:
        X (array-like of shape (n_samples, n_features)): input features
        y (array-like of shape (n_samples,)): target labels

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

def split_regression(X, y):
    """
    Split the data into training and test sets.

    Args:
        X (array-like of shape (n_samples, n_features)): input features
        y (array-like of shape (n_samples,)): target labels

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

def make_cv_classification():
    """
    Create a stratified cross-validation object.

    Returns:
        StratifiedKFold: cross-validation object
    """
    return StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)

def make_cv_regression():
    """
    Create a cross-validation object.

    Returns:
        KFold: cross-validation object
    """
    return KFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)