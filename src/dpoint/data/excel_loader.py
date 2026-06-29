# excel_loader.py
"""
单股 Excel 数据加载器。
来自 Ashare_DpointTrader_deeplearning_Ver2.0/data_loader.py。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

from dpoint.core.constants import (
    COL_AMOUNT,
    COL_CLOSE,
    COL_DATE,
    COL_HIGH,
    COL_LOW,
    COL_OPEN,
    COL_TURNOVER,
    COL_VOLUME,
    OPTIONAL_COLS,
    REQUIRED_COLS_SINGLE,
)
from dpoint.data.cleaner import DataReport, clean_ohlcv

logger = logging.getLogger(__name__)


def load_stock_excel(
    file_path: str | Path,
    ticker: str = "",
) -> Tuple[pd.DataFrame, DataReport]:
    """
    从 Excel 文件加载单只股票数据。

    Args:
        file_path: Excel 文件路径
        ticker: 股票代码（可选，用于报告）

    Returns:
        (清洗后的 DataFrame, 数据质量报告)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    report = DataReport(ticker=ticker or file_path.stem, source_file=str(file_path))

    # 读取 Excel
    df = pd.read_excel(file_path, engine="openpyxl")
    logger.info("Loaded %d rows from %s", len(df), file_path.name)

    # 检查必需列
    missing = [c for c in REQUIRED_COLS_SINGLE if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # 检查可选列
    missing_opt = [c for c in OPTIONAL_COLS if c not in df.columns]
    report.missing_optional_cols = missing_opt
    if missing_opt:
        logger.info("Missing optional columns: %s", missing_opt)

    # 清洗
    df = clean_ohlcv(df, report)

    # 设置日期索引
    if COL_DATE in df.columns:
        df = df.set_index(COL_DATE)

    return df, report
