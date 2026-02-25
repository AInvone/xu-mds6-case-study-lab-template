from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

def make_classification_preprocessor(X):
    """
    Automatically detect categorical vs numerical columns.
    Categoricals are one-hot encoded; numericals are scaled (for stable convergence of e.g. LogisticRegression).
    Returns a ColumnTransformer.
    """
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    numerical_cols = X.select_dtypes(exclude=["object", "category"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numerical_cols),
        ]
    )
    return preprocessor


def make_regression_preprocessor(X):
    """
    For regression: scale numeric features.
    """
    numerical_cols = X.select_dtypes(exclude=["object", "category"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
        ]
    )
    return preprocessor