# registry.py
"""
统一模型注册表：make_model 工厂函数。
支持 binary/multiclass/regression 三种 task_type。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from dpoint.core.tasks import LabelSpec, get_output_dim

logger = logging.getLogger(__name__)

ML_MODELS = {"logreg", "sgd", "xgb"}
DL_MODELS = {"mlp", "lstm", "gru", "cnn", "transformer"}
ALL_MODELS = ML_MODELS | DL_MODELS


def make_model(
    model_type: str,
    input_dim: int,
    config: Dict[str, Any],
    label_spec: Optional[LabelSpec] = None,
) -> Tuple[Any, str]:
    """
    模型工厂。

    Args:
        model_type: 模型类型
        input_dim: 输入特征维度
        config: 模型超参数配置
        label_spec: 标签规格（决定 output_dim）

    Returns:
        (模型对象, "sklearn" 或 "torch")
    """
    output_dim = 1
    if label_spec:
        output_dim = get_output_dim(label_spec.task_type, label_spec.n_classes)

    if model_type in ML_MODELS:
        from dpoint.models.sklearn_models import create_sklearn_model
        model = create_sklearn_model(model_type, config)
        return model, "sklearn"

    elif model_type in DL_MODELS:
        from dpoint.models.torch_models import create_dl_model
        model = create_dl_model(model_type, input_dim, config, output_dim=output_dim)
        return model, "torch"

    else:
        raise ValueError(f"Unknown model type: {model_type}. Available: {ALL_MODELS}")
