import logging
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from src.core.models.Settings import settings

logger = logging.getLogger(__name__)


def predict_many(model, x: NDArray) -> NDArray:
    return np.array([model.predict(row) for row in x])


def accuracy(y_true: NDArray, y_pred: NDArray) -> float:
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: NDArray, y_pred: NDArray):
    labels = np.unique(np.concatenate([y_true, y_pred]))
    idx = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    return labels, cm


def evaluate(model, x: NDArray, y: NDArray, name: str = "test") -> dict:
    y_pred = predict_many(model, x)
    labels, cm = confusion_matrix(y, y_pred)
    acc = accuracy(y, y_pred)

    tp = np.diag(cm)
    col, row = cm.sum(axis=0), cm.sum(axis=1)
    precision = np.divide(tp, col, out=np.zeros(len(labels)), where=col > 0)
    recall = np.divide(tp, row, out=np.zeros(len(labels)), where=row > 0)

    logger.info("[%s] accuracy=%.3f", name, acc)
    for l, p, r in zip(labels, precision, recall):
        logger.info("[%s] class %s: precision=%.3f recall=%.3f", name, l, p, r)

    if settings.show_plots:
        plot_confusion_matrix(labels, cm, name)

    return {"accuracy": acc, "labels": labels, "confusion": cm,
            "precision": precision, "recall": recall}


def plot_confusion_matrix(labels, cm, name: str = "test") -> None:
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix ({name})")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    plt.show()


def plot_feature_importance(importances: NDArray, feature_names=None) -> None:
    if not settings.show_plots:
        return
    names = feature_names or [f"x{i}" for i in range(len(importances))]
    order = np.argsort(importances)[::-1]
    fig, ax = plt.subplots()
    ax.bar([names[i] for i in order], importances[order])
    ax.set_title("Feature importance (share of total information gain)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def depth_sweep(
        make_model: Callable[[int], object],
        x_train, y_train, x_test, y_test,
        depths=range(1, 15),
) -> dict:
    train_acc, test_acc = [], []
    for d in depths:
        model = make_model(d)
        model.fit(x_train, y_train)
        train_acc.append(accuracy(y_train, predict_many(model, x_train)))
        test_acc.append(accuracy(y_test, predict_many(model, x_test)))
        logger.info("depth=%d train=%.3f test=%.3f", d, train_acc[-1], test_acc[-1])

    if settings.show_plots:
        plt.plot(list(depths), train_acc, marker="o", label="train")
        plt.plot(list(depths), test_acc, marker="o", label="test")
        plt.xlabel("max depth")
        plt.ylabel("accuracy")
        plt.legend()
        plt.title("Accuracy vs tree depth")
        plt.show()

    return {"depths": list(depths), "train": train_acc, "test": test_acc}
