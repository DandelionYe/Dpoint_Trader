# metrics.py
"""
因子评价指标：Rank IC / ICIR / Top-K Return / 分层收益。
来自 Ver1.0/ranking_metrics.py。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RankingMetrics:
    """因子排名指标汇总。"""

    ic_mean: Optional[float] = None
    ic_std: Optional[float] = None
    ic_ir: Optional[float] = None
    rank_ic_mean: Optional[float] = None
    rank_ic_std: Optional[float] = None
    rank_ic_ir: Optional[float] = None
    topk_return_mean: Optional[float] = None
    topk_return_annual: Optional[float] = None
    layered_returns: Optional[Dict[str, float]] = None


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    x_c = x - x.mean()
    y_c = y - y.mean()
    denom = np.sqrt((x_c * x_c).sum() * (y_c * y_c).sum())
    if denom == 0:
        return float("nan")
    return float((x_c * y_c).sum() / denom)


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = pd.Series(x).rank(method="average").values
    y_rank = pd.Series(y).rank(method="average").values
    return _pearson_corr(x_rank, y_rank)


def compute_daily_ic(
    scores_df: pd.DataFrame,
    score_col: str = "score",
    label_col: str = "label",
    date_col: str = "date",
    method: str = "spearman",
) -> pd.Series:
    """
    计算每日 IC（信息系数）。

    Args:
        scores_df: 含 date/score/label 的 DataFrame
        method: "pearson" 或 "spearman"

    Returns:
        每日 IC 的 Series，index 为日期
    """
    corr_fn = _spearman_corr if method == "spearman" else _pearson_corr
    daily_ics = {}

    for dt, group in scores_df.groupby(date_col):
        scores = group[score_col].values
        labels = group[label_col].values
        mask = np.isfinite(scores) & np.isfinite(labels)
        if mask.sum() < 3:
            continue
        ic = corr_fn(scores[mask], labels[mask])
        if np.isfinite(ic):
            daily_ics[dt] = ic

    return pd.Series(daily_ics)


def compute_rank_ic(
    scores_df: pd.DataFrame,
    score_col: str = "score",
    label_col: str = "label",
    date_col: str = "date",
) -> pd.Series:
    """计算每日 Rank IC（Spearman 秩相关）。"""
    return compute_daily_ic(scores_df, score_col, label_col, date_col, method="spearman")


def compute_ranking_metrics(
    scores_df: pd.DataFrame,
    score_col: str = "score",
    label_col: str = "label",
    date_col: str = "date",
    top_k: int = 5,
    n_layers: int = 5,
) -> RankingMetrics:
    """
    计算完整的因子排名指标。

    Args:
        scores_df: 含 date/score/label 的 DataFrame
        top_k: Top-K 选股数
        n_layers: 分层数

    Returns:
        RankingMetrics
    """
    metrics = RankingMetrics()

    # IC
    ic_series = compute_daily_ic(scores_df, score_col, label_col, date_col, method="pearson")
    if len(ic_series) > 0:
        metrics.ic_mean = float(ic_series.mean())
        metrics.ic_std = float(ic_series.std())
        metrics.ic_ir = metrics.ic_mean / max(metrics.ic_std, 1e-10)

    # Rank IC
    rank_ic_series = compute_rank_ic(scores_df, score_col, label_col, date_col)
    if len(rank_ic_series) > 0:
        metrics.rank_ic_mean = float(rank_ic_series.mean())
        metrics.rank_ic_std = float(rank_ic_series.std())
        metrics.rank_ic_ir = metrics.rank_ic_mean / max(metrics.rank_ic_std, 1e-10)

    # Top-K Return
    topk_returns = []
    for dt, group in scores_df.groupby(date_col):
        if len(group) < top_k:
            continue
        top = group.nlargest(top_k, score_col)
        if label_col in top.columns:
            avg_ret = top[label_col].mean()
            if np.isfinite(avg_ret):
                topk_returns.append(avg_ret)

    if topk_returns:
        metrics.topk_return_mean = float(np.mean(topk_returns))
        metrics.topk_return_annual = float(np.mean(topk_returns)) * 252

    # 分层收益
    layer_returns = {}
    for layer in range(n_layers):
        layer_rets = []
        for dt, group in scores_df.groupby(date_col):
            if len(group) < n_layers:
                continue
            group_sorted = group.sort_values(score_col)
            chunk_size = len(group_sorted) // n_layers
            start = layer * chunk_size
            end = start + chunk_size if layer < n_layers - 1 else len(group_sorted)
            chunk = group_sorted.iloc[start:end]
            if label_col in chunk.columns:
                avg_ret = chunk[label_col].mean()
                if np.isfinite(avg_ret):
                    layer_rets.append(avg_ret)
        layer_returns[f"layer_{layer + 1}"] = float(np.mean(layer_rets)) if layer_rets else 0.0

    metrics.layered_returns = layer_returns

    return metrics
