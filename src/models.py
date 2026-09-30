"""Model definitions: ANN (Keras) and classical baselines."""
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from tensorflow import keras
from xgboost import XGBClassifier

from src import config


def build_ann(input_dim: int) -> keras.Model:
    """Feed-forward network: input -> 10 -> 6 -> 10 -> sigmoid."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(10, activation="relu"),
            keras.layers.Dense(6, activation="relu"),
            keras.layers.Dense(10, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="churn_ann",
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def get_baselines() -> dict:
    """Tree-based baselines with the same hyper-parameters as the original notebook."""
    rs = config.RANDOM_STATE
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=5, random_state=rs
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=4, random_state=rs
        ),
        "XGBoost": XGBClassifier(random_state=rs, eval_metric="logloss"),
    }
