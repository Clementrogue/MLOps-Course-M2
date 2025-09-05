from __future__ import annotations
import os
import yaml
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             f1_score, confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay)
import matplotlib.pyplot as plt

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_data(csv_path: str, target: str):
    df = pd.read_csv(csv_path)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

def split_data(X, y, test_size: float, random_state: int):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_model(model, path: str):
    ensure_dir(os.path.dirname(path))
    joblib.dump(model, path)

def plot_and_save_roc(y_true, y_proba, path: str):
    RocCurveDisplay.from_predictions(y_true, y_proba)
    plt.title("ROC Curve")
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def plot_and_save_pr(y_true, y_proba, path: str):
    PrecisionRecallDisplay.from_predictions(y_true, y_proba)
    plt.title("Precision-Recall Curve")
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def plot_and_save_confusion(y_true, y_pred, path: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xlabel='Predicted label', ylabel='True label', title='Confusion Matrix')
    # annotate
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def compute_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }
