# csv_loader.py
"""
单股 CSV 数据加载器。
来自 DpointTrader_deeplearning_Ver1.0/csv_loader.py。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from dpoint.core.constants import (
    DEFAULT_COLUMN_MAP,
    OPTIONAL_COLS,
    REQUIRED_COLS_SINGLE,
)
from dpoint.data.cleaner import DataReport, clean_ohlcv

logger = logging.getLogger(__name__)


def standardize_columns(
    df: pd.DataFrame,
    column_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """标准化列名。"""
    if column_map is None:
        column_map = DEFAULT_COLUMN_MAP
    col_mapping = {}
    for orig_col in df.columns:
        cleaned = orig_col.strip()
        col_mapping[orig_col] = column_map.get(cleaned, column_map.get(orig_col, cleaned))
    return df.rename(columns=col_mapping)


def extract_ticker_from_filename(file_path: Path) -> str:
    """从文件名提取股票代码，如 '600036_19930304.csv' -> '600036'。"""
    stem = file_path.stem
    parts = stem.split("_")
    return parts[0] if parts else stem


def load_single_csv(
    file_path: str | Path,
    ticker: str = "",
    column_map: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, DataReport]:
    """
    从 CSV 文件加载单只股票数据。

    Args:
        file_path: CSV 文件路径
        ticker: 股票代码（若为空则从文件名推断）
        column_map: 列名映射字典

    Returns:
        (清洗后的 DataFrame, 数据质量报告)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    if not ticker:
        ticker = extract_ticker_from_filename(file_path)

    report = DataReport(ticker=ticker, source_file=str(file_path))

    # 读取 CSV
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    logger.info("Loaded %d rows from %s", len(df), file_path.name)

    # 列名标准化
    df = standardize_columns(df, column_map)

    # 检查必需列
    missing = [c for c in REQUIRED_COLS_SINGLE if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after mapping: {missing}")

    # 检查可选列
    missing_opt = [c for c in OPTIONAL_COLS if c not in df.columns]
    report.missing_optional_cols = missing_opt

    # 清洗
    df = clean_ohlcv(df, report)

    return df, report
