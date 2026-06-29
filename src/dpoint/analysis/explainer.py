# explainer.py
"""
特征解释模块：Permutation Importance / SHAP（可选）。
来自 Ver2.0/explainer.py。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    *,
    n_repeats: int = 5,
    scoring: str = "accuracy",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    计算 Permutation Importance。

    Args:
        model: 训练好的模型（需有 predict 或 predict_proba）
        X: 特征矩阵
        y: 标签
        feature_names: 特征名列表
        n_repeats: 重复次数

    Returns:
        DataFrame with columns: feature, importance_mean, importance_std
    """
    from sklearn.metrics import accuracy_score, log_loss

    rng = np.random.Generator(np.random.PCG64(random_state))

    # 基准分数
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            y_pred = (proba[:, 1] >= 0.5).astype(int)
            base_score = accuracy_score(y, y_pred)
        else:
            base_score = accuracy_score(y, np.argmax(proba, axis=1))
    else:
        y_pred = model.predict(X)
        base_score = accuracy_score(y, y_pred)

    results = []
    for feat_idx, feat_name in enumerate(feature_names):
        scores = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            X_permuted[:, feat_idx] = rng.permutation(X_permuted[:, feat_idx])

            if hasattr(model, "predict_proba"):
                proba_perm = model.predict_proba(X_permuted)
                if proba_perm.ndim == 2 and proba_perm.shape[1] == 2:
                    y_pred_perm = (proba_perm[:, 1] >= 0.5).astype(int)
                else:
                    y_pred_perm = np.argmax(proba_perm, axis=1)
                score_perm = accuracy_score(y, y_pred_perm)
            else:
                y_pred_perm = model.predict(X_permuted)
                score_perm = accuracy_score(y, y_pred_perm)

            scores.append(base_score - score_perm)

        results.append(
            {
                "feature": feat_name,
                "importance_mean": float(np.mean(scores)),
                "importance_std": float(np.std(scores)),
            }
        )

    df = (
        pd.DataFrame(results).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    )
    return df


def shap_importance(
    model,
    X: np.ndarray,
    feature_names: List[str],
    *,
    max_samples: int = 100,
) -> Optional[pd.DataFrame]:
    """
    计算 SHAP 特征重要性（需要 shap 库）。

    Returns:
        DataFrame with columns: feature, shap_mean, shap_std
        or None if shap is not installed.
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed. Install with: pip install shap")
        return None

    X_sample = X[:max_samples] if len(X) > max_samples else X

    try:
        if hasattr(model, "predict_proba"):
            explainer = shap.Explainer(model.predict_proba, X_sample)
        else:
            explainer = shap.Explainer(model.predict, X_sample)

        shap_values = explainer(X_sample)

        if hasattr(shap_values, "values"):
            vals = shap_values.values
            if vals.ndim == 3:
                vals = vals[:, :, 1]  # 二分类取正类
        else:
            vals = np.array(shap_values)

        results = []
        for i, name in enumerate(feature_names):
            if i < vals.shape[1]:
                results.append(
                    {
                        "feature": name,
                        "shap_mean": float(np.abs(vals[:, i]).mean()),
                        "shap_std": float(np.abs(vals[:, i]).std()),
                    }
                )

        return (
            pd.DataFrame(results).sort_values("shap_mean", ascending=False).reset_index(drop=True)
        )

    except Exception as e:
        logger.warning("SHAP computation failed: %s", e)
        return None
