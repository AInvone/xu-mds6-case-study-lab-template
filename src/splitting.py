from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from .config import RANDOM_STATE, TEST_SIZE, N_SPLITS_CV

def split_classification(X, y):
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

def split_regression(X, y):
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

def make_cv_classification():
    return StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)

def make_cv_regression():
    return KFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)