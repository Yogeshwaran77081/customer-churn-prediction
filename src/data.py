"""Data loading, feature preparation and leakage-free train/val/test splitting."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import config


@dataclass
class Splits:
    """Container for the model-ready arrays."""
    X_train: np.ndarray        # scaled + SMOTE-balanced (training only)
    y_train: np.ndarray
    X_val: np.ndarray          # scaled, real distribution
    y_val: np.ndarray
    X_test: np.ndarray         # scaled, real distribution
    y_test: np.ndarray
    feature_names: list
    scaler: StandardScaler


def load_data(path=config.RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw churn CSV."""
    return pd.read_csv(path)


def prepare_features(df: pd.DataFrame):
    """Drop identifiers, encode categoricals and return (X, y)."""
    df = df.drop(columns=config.ID_COLUMNS)
    for col, mapping in config.BINARY_MAP.items():
        df[col] = df[col].map(mapping)
    df = pd.get_dummies(df, columns=config.CATEGORICAL_ONEHOT, drop_first=False, dtype=int)
    y = df[config.TARGET]
    X = df.drop(columns=config.TARGET)
    return X, y


def make_splits(X: pd.DataFrame, y: pd.Series) -> Splits:
    """Stratified train / validation / test split.

    * The scaler is fitted on the training portion only (no leakage).
    * SMOTE is applied to the training portion only; validation and test
      sets keep the real class distribution.
    """
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=config.VAL_SIZE, stratify=y_tmp, random_state=config.RANDOM_STATE
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s, X_val_s, X_test_s = (scaler.transform(a) for a in (X_train, X_val, X_test))

    X_train_res, y_train_res = SMOTE(random_state=config.RANDOM_STATE).fit_resample(X_train_s, y_train)

    return Splits(
        X_train_res, y_train_res.to_numpy(),
        X_val_s, y_val.to_numpy(),
        X_test_s, y_test.to_numpy(),
        list(X.columns), scaler,
    )
