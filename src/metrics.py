"""Evaluation metrics, diagnostic plots, and top-2 confusion analysis for EE300148."""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for CPU
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def compute_confusion_matrix(
    targets: np.ndarray,
    preds: np.ndarray,
    class_names: List[str],
) -> Tuple[np.ndarray, List[Dict[str, any]]]:
    """Compute confusion matrix and extract top-2 most confused class pairs (Slide 32)."""
    cm = confusion_matrix(targets, preds, labels=list(range(len(class_names))))

    # Extract all off-diagonal confusion pairs (i != j)
    confusions = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                confusions.append(
                    {
                        "true_class": class_names[i],
                        "pred_class": class_names[j],
                        "true_idx": i,
                        "pred_idx": j,
                        "count": int(cm[i, j]),
                    }
                )

    # Sort descending by error count
    confusions.sort(key=lambda x: x["count"], reverse=True)
    top_2_confused = confusions[:2]

    return cm, top_2_confused


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: str = "./artifacts/figures/confusion_matrix.png",
    title: str = "Test Set Confusion Matrix (ResNet-18 Phase 2)",
) -> None:
    """Export high-DPI confusion matrix heatmap."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Class",
        xlabel="Predicted Class",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold",
            )

    fig.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(
    targets: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
    output_path: str = "./artifacts/figures/roc_curve.png",
) -> Dict[str, float]:
    """Plot multi-class One-vs-Rest ROC curves with chance baseline (AUC = 0.50)."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

    n_classes = len(class_names)
    auc_scores: Dict[str, float] = {}

    for i in range(n_classes):
        binary_targets = (targets == i).astype(int)
        class_probs = probs[:, i]
        fpr, tpr, _ = roc_curve(binary_targets, class_probs)
        roc_auc = auc(fpr, tpr)
        auc_scores[class_names[i]] = float(roc_auc)
        ax.plot(fpr, tpr, lw=2, label=f"{class_names[i]} (AUC = {roc_auc:.3f})")

    # Baseline chance diagonal
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Majority Baseline / Chance (AUC = 0.500)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("One-vs-Rest ROC Curves (Test Set)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return auc_scores


def plot_pr_curve(
    targets: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
    output_path: str = "./artifacts/figures/pr_curve.png",
) -> Dict[str, float]:
    """Plot multi-class Precision-Recall curves with prevalence baseline."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

    n_classes = len(class_names)
    ap_scores: Dict[str, float] = {}

    for i in range(n_classes):
        binary_targets = (targets == i).astype(int)
        class_probs = probs[:, i]
        precision, recall, _ = precision_recall_curve(binary_targets, class_probs)
        pr_auc = auc(recall, precision)
        ap_scores[class_names[i]] = float(pr_auc)
        ax.plot(recall, precision, lw=2, label=f"{class_names[i]} (AUC = {pr_auc:.3f})")

    # Baseline prevalence for balanced 5-class dataset is 0.20
    prevalence = 1.0 / n_classes
    ax.axhline(
        y=prevalence,
        color="k",
        linestyle="--",
        lw=1.5,
        label=f"Baseline Prevalence ({prevalence:.2f})",
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (True Positive Rate)")
    ax.set_ylabel("Precision")
    ax.set_title("One-vs-Rest Precision-Recall Curves (Test Set)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return ap_scores


def plot_scratch_vs_transfer(
    summary: Dict[str, dict],
    output_path: str = "./artifacts/figures/scratch_vs_transfer.png",
) -> None:
    """Export comparison bar chart of mean ± std across models (Slide 62)."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    model_names = list(summary.keys())
    means = [summary[m]["mean"] * 100 for m in model_names]
    stds = [summary[m]["std"] * 100 for m in model_names]

    colors = ["#7f7f7f", "#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]
    bars = ax.bar(
        model_names,
        means,
        yerr=stds,
        capsize=5,
        color=colors[: len(model_names)],
        edgecolor="black",
        linewidth=1,
    )

    for bar, mean, std in zip(bars, means, stds):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + (std if std > 0 else 0) + 1.5,
            f"{mean:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )

    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_ylim([0, 115])
    ax.set_title("Baseline vs. Scratch CNN vs. Transfer Learning (Mean ± Std)")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    ax.grid(axis="y", linestyle=":", alpha=0.7)

    fig.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_results_json(
    data: Optional[dict] = None,
    filepath: str = "./artifacts/results.json",
    results_data: Optional[dict] = None,
) -> None:
    """Save serializable evaluation summary and metric results."""
    payload = data if data is not None else results_data
    if payload is None:
        raise ValueError("Must provide either data or results_data")
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
