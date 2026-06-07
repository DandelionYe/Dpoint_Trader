# splitters.py
"""
统一的样本划分模块（日期粒度）。
来自 DpointTrader_deeplearning_Ver1.0/splitters.py，兼容单股/面板两种模式。
支持 Walk-Forward / Embargo Walk-Forward / Nested Walk-Forward / Final Holdout。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SplitSpec:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    embargo_days: int = 0
    fold_id: int = 0


@dataclass
class SplitResult:
    train_dates: List[pd.Timestamp]
    val_dates: List[pd.Timestamp]
    spec: SplitSpec


def _unique_dates(df: pd.DataFrame, date_col: str = "date") -> List[pd.Timestamp]:
    return sorted(pd.to_datetime(df[date_col].unique()))


def walkforward_splits(
    df: pd.DataFrame,
    *,
    n_folds: int = 4,
    train_start_ratio: float = 0.5,
    min_rows: int = 80,
    date_col: str = "date",
) -> List[SplitResult]:
    """
    Walk-Forward expanding window 切分。

    Args:
        df: 输入 DataFrame
        n_folds: 折数
        train_start_ratio: 初始训练集占比
        min_rows: 每折最小行数
        date_col: 日期列名

    Returns:
        SplitResult 列表
    """
    dates = _unique_dates(df, date_col)
    n_dates = len(dates)

    if n_dates < min_rows * 2:
        logger.warning("Not enough dates (%d) for %d folds", n_dates, n_folds)
        return []

    # 计算切分点
    start_idx = int(n_dates * train_start_ratio)
    remaining = n_dates - start_idx
    fold_size = remaining // n_folds

    if fold_size < 1:
        logger.warning("Fold size too small, reducing n_folds")
        n_folds = max(1, remaining)
        fold_size = 1

    results = []
    for i in range(n_folds):
        train_end_idx = start_idx + i * fold_size
        val_end_idx = min(start_idx + (i + 1) * fold_size, n_dates)

        if val_end_idx <= train_end_idx:
            continue

        train_dates = dates[:train_end_idx]
        val_dates = dates[train_end_idx:val_end_idx]

        if len(train_dates) < min_rows or len(val_dates) < min_rows // 4:
            continue

        spec = SplitSpec(
            train_start=dates[0],
            train_end=dates[train_end_idx - 1],
            val_start=dates[train_end_idx],
            val_end=dates[val_end_idx - 1],
            fold_id=i,
        )
        results.append(SplitResult(train_dates=train_dates, val_dates=val_dates, spec=spec))

    logger.info("Walk-forward: %d folds from %d dates", len(results), n_dates)
    return results


def walkforward_splits_with_embargo(
    df: pd.DataFrame,
    *,
    n_folds: int = 4,
    train_start_ratio: float = 0.5,
    embargo_days: int = 5,
    min_rows: int = 80,
    date_col: str = "date",
) -> List[SplitResult]:
    """带 Embargo Gap 的 Walk-Forward 切分。"""
    results = walkforward_splits(df, n_folds=n_folds, train_start_ratio=train_start_ratio,
                                  min_rows=min_rows, date_col=date_col)
    if embargo_days <= 0:
        return results

    dates = _unique_dates(df, date_col)
    embargo_results = []
    for r in results:
        # 从训练集末尾移除 embargo_days
        if len(r.train_dates) <= embargo_days:
            continue
        new_train = r.train_dates[:-embargo_days]
        new_spec = SplitSpec(
            train_start=r.spec.train_start,
            train_end=new_train[-1],
            val_start=r.spec.val_start,
            val_end=r.spec.val_end,
            embargo_days=embargo_days,
            fold_id=r.spec.fold_id,
        )
        embargo_results.append(SplitResult(train_dates=new_train, val_dates=r.val_dates, spec=new_spec))

    logger.info("Embargo walk-forward: %d folds, embargo=%d days", len(embargo_results), embargo_days)
    return embargo_results


def final_holdout_split(
    df: pd.DataFrame,
    *,
    holdout_ratio: float = 0.15,
    gap_days: int = 0,
    date_col: str = "date",
) -> Tuple[List[pd.Timestamp], List[pd.Timestamp]]:
    """
    从末尾切出 holdout 集。

    Returns:
        (search_dates, holdout_dates)
    """
    dates = _unique_dates(df, date_col)
    n = len(dates)
    holdout_size = max(1, int(n * holdout_ratio))
    split_idx = n - holdout_size

    if gap_days > 0:
        split_idx = max(0, split_idx - gap_days)

    search_dates = dates[:split_idx]
    holdout_dates = dates[split_idx:]

    logger.info("Holdout split: %d search dates, %d holdout dates", len(search_dates), len(holdout_dates))
    return search_dates, holdout_dates


def recommend_n_folds(n_dates: int, min_fold_size: int = 60) -> int:
    """根据数据量自适应推荐折数。"""
    available = n_dates * 0.5  # 假设 train_start_ratio=0.5
    folds = max(2, min(8, int(available / min_fold_size)))
    return folds
