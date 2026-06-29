# metrics.py
"""
可配置的搜索目标函数。
支持 PnL（回测净值）和 Rank IC（排名信息系数）两种模式。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol

import numpy as np

logger = logging.getLogger(__name__)


class MetricFn(Protocol):
    """搜索目标函数接口。"""

    def __call__(self, fold_results: List[Dict[str, Any]]) -> float: ...


@dataclass
class FoldResult:
    """单折回测结果。"""

    fold_id: int
    geom_mean_ratio: float = 0.0  # 几何均值净值比率
    min_fold_ratio: float = 0.0  # 最差折比率
    n_trades: int = 0
    # Rank IC 相关
    rank_ic: float = 0.0
    topk_return: float = 0.0


def pnl_metric(fold_results: List[Dict[str, Any]]) -> float:
    """
    PnL 目标函数（来自 Ver2.0）。
    0.8 * 几何均值 + 0.2 * 最差折 - 交易频次惩罚
    """
    from dpoint.core.constants import (
        LAMBDA_TRADE_PENALTY,
        MIN_CLOSED_TRADES_PER_FOLD,
        TARGET_CLOSED_TRADES_PER_FOLD,
    )

    if not fold_results:
        return -np.inf

    ratios = [r.get("geom_mean_ratio", 0.0) for r in fold_results]
    n_trades_list = [r.get("n_trades", 0) for r in fold_results]

    # 检查最少交易数
    if any(n < MIN_CLOSED_TRADES_PER_FOLD for n in n_trades_list):
        return -np.inf

    geom_mean = float(np.exp(np.mean(np.log(np.maximum(ratios, 1e-10)))))
    min_fold = float(np.min(ratios))

    # 交易频次惩罚
    avg_trades = float(np.mean(n_trades_list))
    penalty = (
        LAMBDA_TRADE_PENALTY
        * max(0, TARGET_CLOSED_TRADES_PER_FOLD - avg_trades)
        / TARGET_CLOSED_TRADES_PER_FOLD
    )

    score = 0.8 * geom_mean + 0.2 * min_fold - penalty
    return score


def rank_ic_metric(fold_results: List[Dict[str, Any]]) -> float:
    """
    Rank IC 目标函数（来自 Ver1.0）。
    各折 Rank IC 的均值。
    """
    if not fold_results:
        return -np.inf

    ics = [r.get("rank_ic", 0.0) for r in fold_results]
    return float(np.mean(ics))


# 目标函数注册表
METRIC_REGISTRY: Dict[str, MetricFn] = {
    "pnl": pnl_metric,
    "rank_ic": rank_ic_metric,
}


def get_metric_fn(name: str) -> MetricFn:
    """获取目标函数。"""
    if name not in METRIC_REGISTRY:
        raise ValueError(f"Unknown metric: {name}. Available: {list(METRIC_REGISTRY.keys())}")
    return METRIC_REGISTRY[name]
