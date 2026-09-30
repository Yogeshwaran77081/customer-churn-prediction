"""End-to-end training script.

Usage (from the repository root):
    python -m src.train
"""
import json
import random

import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from tensorflow import keras

from src import config, visualize
from src.data import load_data, make_splits, prepare_features
from src.models import build_ann, get_baselines


def set_seeds(seed: int = config.RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def score(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
    }


def main() -> None:
    set_seeds()
    df = load_data()
    visualize.plot_class_balance(df)
    visualize.plot_correlation(df)

    X, y = prepare_features(df)
    s = make_splits(X, y)
    print(f"train (SMOTE) {s.X_train.shape} | val {s.X_val.shape} | test {s.X_test.shape}")

    preds, probas, results = {}, {}, {}

    # --- ANN -------------------------------------------------------------
    ann = build_ann(s.X_train.shape[1])
    history = ann.fit(
        s.X_train, s.y_train,
        validation_data=(s.X_val, s.y_val),
        epochs=config.ANN_PARAMS["epochs"],
        batch_size=config.ANN_PARAMS["batch_size"],
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=config.ANN_PARAMS["patience"], restore_best_weights=True)],
        verbose=2,
    )
    visualize.plot_training_history(history)
    p = ann.predict(s.X_test, verbose=0).ravel()
    probas["ANN"], preds["ANN"] = p, (p > config.THRESHOLD).astype(int)
    config.MODELS_DIR.mkdir(exist_ok=True)
    ann.save(config.MODELS_DIR / "churn_ann.keras")

    # --- Baselines ---------------------------------------------------------
    for name, model in get_baselines().items():
        model.fit(s.X_train, s.y_train)
        probas[name] = model.predict_proba(s.X_test)[:, 1]
        preds[name] = (probas[name] > config.THRESHOLD).astype(int)

    # --- Evaluation --------------------------------------------------------
    for name in preds:
        results[name] = score(s.y_test, preds[name], probas[name])
        print(name, results[name])

    visualize.plot_confusion_matrices(s.y_test, preds)
    visualize.plot_roc_curves(s.y_test, probas)

    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"Saved metrics to {config.REPORTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
