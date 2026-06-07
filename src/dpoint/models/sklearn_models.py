# sklearn_models.py
"""
sklearn / XGBoost 模型工厂。
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np


def create_sklearn_model(model_type: str, config: Dict[str, Any]) -> Any:
    """创建 sklearn/XGBoost 模型。"""
    if model_type == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                C=config.get("C", 1.0),
                penalty=config.get("penalty", "l2"),
                solver="saga" if config.get("penalty") == "l1" else "lbfgs",
                max_iter=1000,
                class_weight=config.get("class_weight", None),
            )),
        ])

    elif model_type == "sgd":
        from sklearn.linear_model import SGDClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", SGDClassifier(
                loss="log_loss",
                alpha=config.get("alpha", 1e-4),
                penalty=config.get("penalty", "l2"),
                max_iter=1000,
                class_weight=config.get("class_weight", None),
            )),
        ])

    elif model_type == "xgb":
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=config.get("n_estimators", 200),
                max_depth=config.get("max_depth", 3),
                learning_rate=config.get("learning_rate", 0.05),
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                use_label_encoder=False,
                eval_metric="logloss",
                verbosity=0,
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=config.get("n_estimators", 200),
                max_depth=config.get("max_depth", 3),
                learning_rate=config.get("learning_rate", 0.05),
                subsample=0.8,
            )

    else:
        raise ValueError(f"Unknown sklearn model type: {model_type}")
