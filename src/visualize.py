"""Plotting helpers. All functions save a PNG into reports/figures/."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve

from src import config

sns.set_theme(style="whitegrid")


def _save(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_class_balance(df):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x=config.TARGET, data=df, ax=ax)
    ax.set_title("Class balance (0 = stayed, 1 = churned)")
    return _save(fig, "class_balance.png")


def plot_correlation(df):
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(df.select_dtypes("number").drop(columns=["RowNumber", "CustomerId"]).corr(),
                annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Feature correlation")
    return _save(fig, "correlation_heatmap.png")


def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in zip(axes, ["loss", "accuracy"], ["Loss", "Accuracy"]):
        ax.plot(history.history[key], label="train")
        ax.plot(history.history[f"val_{key}"], label="validation")
        ax.set_title(f"ANN {title}")
        ax.set_xlabel("Epoch")
        ax.legend()
    return _save(fig, "ann_training_history.png")


def plot_confusion_matrices(y_true, preds: dict):
    fig, axes = plt.subplots(1, len(preds), figsize=(4.2 * len(preds), 4))
    for ax, (name, y_pred) in zip(axes, preds.items()):
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name)
        ax.grid(False)
    return _save(fig, "confusion_matrices.png")


def plot_roc_curves(y_true, probas: dict):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (test set)")
    ax.legend()
    return _save(fig, "roc_curves.png")
