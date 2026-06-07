# space.py
"""
搜索空间定义与采样。
合并自两个项目的搜索空间设计。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# 模型类型列表
ML_MODELS = ["logreg", "sgd", "xgb"]
DL_MODELS = ["mlp", "lstm", "gru", "cnn", "transformer"]
ALL_MODELS = ML_MODELS + DL_MODELS


def sample_model_config(
    model_type: str,
    rng: np.random.Generator,
    feature_dim: int = 20,
    seq_len: int = 10,
) -> Dict[str, Any]:
    """为指定模型类型采样一组超参数配置。"""
    config = {"model_type": model_type}

    if model_type == "logreg":
        config["C"] = float(rng.choice([0.01, 0.1, 0.3, 1.0, 3.0, 10.0]))
        config["penalty"] = str(rng.choice(["l1", "l2"]))
    elif model_type == "sgd":
        config["alpha"] = float(10 ** rng.uniform(-5, -2))
        config["penalty"] = str(rng.choice(["l1", "l2", "elasticnet"]))
    elif model_type == "xgb":
        config["n_estimators"] = int(rng.choice([100, 200, 400]))
        config["max_depth"] = int(rng.choice([2, 3, 4]))
        config["learning_rate"] = float(rng.choice([0.03, 0.05, 0.1]))
    elif model_type == "mlp":
        config["hidden_dim"] = int(rng.choice([32, 64, 128, 256]))
        config["dropout_rate"] = float(rng.choice([0.1, 0.2, 0.3, 0.5]))
        config["learning_rate"] = float(10 ** rng.uniform(-4, -2.5))
    elif model_type in ("lstm", "gru"):
        config["hidden_dim"] = int(rng.choice([32, 64, 128]))
        config["num_layers"] = int(rng.choice([1, 2, 3]))
        config["dropout_rate"] = float(rng.choice([0.1, 0.2, 0.3]))
        config["bidirectional"] = bool(rng.choice([True, False]))
        config["seq_len"] = int(rng.choice([5, 10, 20]))
        config["learning_rate"] = float(10 ** rng.uniform(-4, -2.5))
    elif model_type == "cnn":
        config["hidden_dim"] = int(rng.choice([32, 64, 128]))
        config["dropout_rate"] = float(rng.choice([0.1, 0.2, 0.3]))
        config["seq_len"] = int(rng.choice([5, 10, 20]))
        config["learning_rate"] = float(10 ** rng.uniform(-4, -2.5))
    elif model_type == "transformer":
        config["hidden_dim"] = int(rng.choice([32, 64, 128]))
        config["num_layers"] = int(rng.choice([1, 2]))
        config["dropout_rate"] = float(rng.choice([0.1, 0.2, 0.3]))
        config["seq_len"] = int(rng.choice([5, 10, 20]))
        config["learning_rate"] = float(10 ** rng.uniform(-4, -2.5))

    return config


def sample_trade_config(rng: np.random.Generator) -> Dict[str, Any]:
    """采样交易参数配置。"""
    return {
        "buy_threshold": float(rng.uniform(0.50, 0.65)),
        "sell_threshold": float(rng.uniform(0.35, 0.50)),
        "confirm_days": int(rng.choice([1, 2, 3])),
        "max_hold_days": int(rng.choice([5, 10, 20, 40])),
    }


def sample_feature_config(rng: np.random.Generator) -> Dict[str, Any]:
    """采样特征配置。"""
    return {
        "use_momentum": bool(rng.choice([True, False])),
        "use_volatility": bool(rng.choice([True, False])),
        "use_volume": bool(rng.choice([True, False])),
        "use_candle": bool(rng.choice([True, False])),
        "use_turnover": bool(rng.choice([True, False])),
        "use_ta_indicators": bool(rng.choice([True, False])),
        "vol_metric": str(rng.choice(["std", "mad"])),
        "liq_transform": str(rng.choice(["ratio", "zscore"])),
    }


def mutate_model_config(
    config: Dict[str, Any],
    rng: np.random.Generator,
    mutation_rate: float = 0.3,
) -> Dict[str, Any]:
    """对已有配置进行小扰动（exploit 模式）。"""
    new_config = config.copy()
    model_type = config.get("model_type", "lstm")

    for key in config:
        if key == "model_type":
            continue
        if rng.random() > mutation_rate:
            continue
        val = config[key]
        if isinstance(val, float):
            new_config[key] = val * rng.uniform(0.8, 1.2)
        elif isinstance(val, int):
            new_config[key] = max(1, val + int(rng.choice([-1, 0, 1])))
        elif isinstance(val, bool):
            new_config[key] = not val

    return new_config
