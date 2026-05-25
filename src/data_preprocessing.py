import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.2


def make_features_target(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def build_preprocessor() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def preprocess_train_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, Pipeline]:
    preprocessor = build_preprocessor()
    X_train_processed = pd.DataFrame(
        preprocessor.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_processed = pd.DataFrame(
        preprocessor.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    return X_train_processed, X_test_processed, preprocessor


def build_leakage_checklist(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_train_processed: pd.DataFrame,
    X_test_processed: pd.DataFrame,
) -> pd.DataFrame:
    checks = [
        {
            "check": "Train/test indices are disjoint",
            "passed": set(X_train.index).isdisjoint(set(X_test.index)),
        },
        {
            "check": "Preprocessing kept train row count unchanged",
            "passed": len(X_train_processed) == len(X_train),
        },
        {
            "check": "Preprocessing kept test row count unchanged",
            "passed": len(X_test_processed) == len(X_test),
        },
        {
            "check": "Preprocessing kept feature count unchanged",
            "passed": X_train_processed.shape[1] == X_train.shape[1]
            and X_test_processed.shape[1] == X_test.shape[1],
        },
        {
            "check": "Test set is transformed only, not resampled",
            "passed": X_test_processed.index.equals(X_test.index),
        },
    ]
    return pd.DataFrame(checks)
