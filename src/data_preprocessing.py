import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def split_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = [
        column for column in X.columns if column not in numeric_features
    ]
    return numeric_features, categorical_features


def build_numeric_preprocessor() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def build_categorical_preprocessor() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", _make_one_hot_encoder()),
        ]
    )


def build_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> Pipeline | ColumnTransformer:
    if numeric_features is None and categorical_features is None:
        return build_numeric_preprocessor()

    transformers = []
    if numeric_features:
        transformers.append(
            ("numeric", build_numeric_preprocessor(), numeric_features)
        )
    if categorical_features:
        transformers.append(
            ("categorical", build_categorical_preprocessor(), categorical_features)
        )
    if not transformers:
        raise ValueError("No feature columns to preprocess.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def preprocess_train_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, Pipeline | ColumnTransformer]:
    numeric_features, categorical_features = split_feature_types(X_train)
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_train_array = preprocessor.fit_transform(X_train)
    X_test_array = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    X_train_processed = pd.DataFrame(
        X_train_array,
        columns=feature_names,
        index=X_train.index,
    )
    X_test_processed = pd.DataFrame(
        X_test_array,
        columns=feature_names,
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
            "check": "Preprocessing kept train/test feature columns aligned",
            "passed": X_train_processed.columns.equals(X_test_processed.columns),
        },
        {
            "check": "Test set is transformed only, not resampled",
            "passed": X_test_processed.index.equals(X_test.index),
        },
    ]
    return pd.DataFrame(checks)
