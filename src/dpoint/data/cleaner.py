# cleaner.py
"""
共享的 OHLCV 数据清洗逻辑。
两个项目的清洗逻辑高度相似，统一在此处实现。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

import pandas as pd

from dpoint.core.constants import (
    COL_AMOUNT,
    COL_CLOSE,
    COL_DATE,
    COL_HIGH,
    COL_LOW,
    COL_OPEN,
    COL_VOLUME,
)

logger = logging.getLogger(__name__)


@dataclass
class DataReport:
    """数据质量报告。"""
    ticker: str = ""
    source_file: str = ""
    rows_raw: int = 0
    rows_after_clean: int = 0
    missing_optional_cols: List[str] = field(default_factory=list)
    derived_cols: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def clean_ohlcv(df: pd.DataFrame, report: DataReport | None = None) -> pd.DataFrame:
    """
    统一的 OHLCV 清洗流程：
    1. 日期解析与排序
    2. 去重（保留最后一条）
    3. 非正值价格过滤
    4. OHLC 一致性检查
    5. 负成交量过滤
    6. 缺失值处理
    """
    if report is None:
        report = DataReport()
    report.rows_raw = len(df)

    df = df.copy()

    # 1. 日期解析
    if COL_DATE in df.columns:
        df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")
        df = df.dropna(subset=[COL_DATE])
        df = df.sort_values(COL_DATE).reset_index(drop=True)

    # 2. 去重
    if COL_DATE in df.columns:
        n_before = len(df)
        df = df.drop_duplicates(subset=[COL_DATE], keep="last")
        n_dupes = n_before - len(df)
        if n_dupes > 0:
            report.notes.append(f"Removed {n_dupes} duplicate dates")

    # 3. 非正值价格过滤
    price_cols = [c for c in [COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE] if c in df.columns]
    if price_cols:
        mask = (df[price_cols] > 0).all(axis=1)
        n_bad = (~mask).sum()
        if n_bad > 0:
            report.notes.append(f"Filtered {n_bad} rows with non-positive prices")
        df = df[mask]

    # 4. OHLC 一致性检查
    if all(c in df.columns for c in [COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE]):
        ohlc_ok = (
            (df[COL_HIGH] >= df[[COL_OPEN, COL_CLOSE, COL_LOW]].max(axis=1)) &
            (df[COL_LOW] <= df[[COL_OPEN, COL_CLOSE, COL_HIGH]].min(axis=1))
        )
        n_inconsistent = (~ohlc_ok).sum()
        if n_inconsistent > 0:
            report.notes.append(f"Found {n_inconsistent} OHLC-inconsistent rows (kept)")
            # 不删除，只记录警告

    # 5. 负成交量过滤
    if COL_VOLUME in df.columns:
        mask = df[COL_VOLUME] >= 0
        n_neg = (~mask).sum()
        if n_neg > 0:
            report.notes.append(f"Filtered {n_neg} rows with negative volume")
        df = df[mask]

    # 6. 衍生列：amount 缺失时用均价 * 成交量
    required_cols = [COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME]
    if COL_AMOUNT not in df.columns and all(c in df.columns for c in required_cols):
        df[COL_AMOUNT] = ((df[COL_OPEN] + df[COL_HIGH] + df[COL_LOW] + df[COL_CLOSE]) / 4) * df[COL_VOLUME]
        report.derived_cols.append(COL_AMOUNT)
        report.notes.append("Derived 'amount' from OHLC avg * volume")

    report.rows_after_clean = len(df)
    return df.reset_index(drop=True)
