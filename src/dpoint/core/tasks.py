# tasks.py
"""
任务类型管理：LabelSpec / LossSpec / MetricSpec。
来自 DpointTrader_deeplearning_Ver1.0/tasks.py，支持 binary/multiclass/regression。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class LabelSpec:
    task_type: str  # binary_classification / multiclass_classification / regression
    label_mode: str  # binary_next_close_up / multiclass_N / regression_return
    horizon_days: int = 1
    thresholds: Optional[dict] = None
    n_classes: Optional[int] = None


@dataclass
class LossSpec:
    loss_name: str  # bce_with_logits / cross_entropy / huber
    output_dim: int
    target_dtype: str
    prediction_key: str  # proba_up / class_id / prediction


@dataclass
class MetricSpec:
    primary_metric: str
    report_metrics: list[str]
    greater_is_better: bool


def infer_task_type(label_mode: str) -> str:
    normalized = str(label_mode).lower()
    if normalized.startswith("regression"):
        return "regression"
    if normalized.startswith("multiclass"):
        return "multiclass_classification"
    return "binary_classification"


def resolve_label_spec(
    label_mode: str = "binary_next_close_up",
    task_type: str = "",
    horizon_days: int = 1,
) -> LabelSpec:
    task_type = task_type or infer_task_type(label_mode)
    n_classes = None
    if label_mode.startswith("multiclass_"):
        try:
            n_classes = int(label_mode.split("_")[-1])
        except ValueError:
            n_classes = None
    return LabelSpec(
        task_type=task_type,
        label_mode=label_mode,
        horizon_days=max(1, horizon_days),
        n_classes=n_classes,
    )


def get_output_dim(task_type: str, n_classes: Optional[int] = None) -> int:
    if task_type == "binary_classification":
        return 1
    if task_type == "multiclass_classification":
        return int(n_classes or 3)
    if task_type == "regression":
        return 1
    raise ValueError(f"Unsupported task_type: {task_type}")


def multiclass_class_values(n_classes: int) -> np.ndarray:
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2")
    return np.linspace(-1.0, 1.0, int(n_classes), dtype=np.float32)


def multiclass_probabilities_to_score(probabilities: Any) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float32)
    if probs.ndim != 2:
        raise ValueError("multiclass probabilities must be 2D")
    class_values = multiclass_class_values(probs.shape[1])
    return probs @ class_values


def resolve_loss_spec(task_type: str, n_classes: Optional[int] = None) -> LossSpec:
    if task_type == "binary_classification":
        return LossSpec("bce_with_logits", 1, "float32", "proba_up")
    if task_type == "multiclass_classification":
        return LossSpec("cross_entropy", get_output_dim(task_type, n_classes), "int64", "class_id")
    if task_type == "regression":
        return LossSpec("huber", 1, "float32", "prediction")
    raise ValueError(f"Unsupported task_type: {task_type}")


def resolve_metric_spec(task_type: str, primary_metric: str = "auto") -> MetricSpec:
    if primary_metric == "auto":
        if task_type == "binary_classification":
            primary_metric = "rank_ic_mean"
        elif task_type == "multiclass_classification":
            primary_metric = "macro_f1"
        else:
            primary_metric = "rank_ic_mean"

    if task_type == "binary_classification":
        return MetricSpec(
            primary_metric,
            ["rank_ic_mean", "auc", "logloss", "topk_return_mean"],
            True,
        )
    if task_type == "multiclass_classification":
        return MetricSpec(
            primary_metric,
            ["rank_ic_mean", "topk_return_mean", "macro_f1", "accuracy"],
            True,
        )
    return MetricSpec(
        primary_metric,
        ["rank_ic_mean", "rmse", "mae", "topk_return_mean"],
        primary_metric not in {"rmse", "mse", "mae"},
    )
