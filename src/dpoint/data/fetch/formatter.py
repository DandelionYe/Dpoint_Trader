"""
数据格式转换：将 QMT DataFrame 转换为 Dpoint_Trader 所需格式。

QMT 返回格式:
    列: time(ms), open, high, low, close, volume, amount

Dpoint_Trader 单股格式:
    列: date, open_qfq, high_qfq, low_qfq, close_qfq, volume, amount

Dpoint_Trader 篮子 CSV 格式:
    列: Date, Open (CNY, qfq), High (CNY, qfq), Low (CNY, qfq), Close (CNY, qfq), Volume (shares)
    日期格式: YYYY/M/D
"""
from __future__ import annotations

import logging
import sys

import pandas as pd

from dpoint.core.constants import (
    COL_AMOUNT,
    COL_CLOSE,
    COL_DATE,
    COL_HIGH,
    COL_LOW,
    COL_OPEN,
    COL_VOLUME,
    DEFAULT_COLUMN_MAP,
)

logger = logging.getLogger(__name__)

# QMT 返回的必需列
_QMT_REQUIRED = ["time", "open", "high", "low", "close", "volume"]

# Windows strftime 不支持 %-m/%-d，需用 %#m/%#d
_DATE_FMT = "%Y/%#m/%#d" if sys.platform == "win32" else "%Y/%-m/%-d"

# 内部标准列名（来自 constants.py）
_INTERNAL_COLS = [COL_DATE, COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME, COL_AMOUNT]

# 外部 CSV 列名（来自 DEFAULT_COLUMN_MAP 的 keys）
_CSV_COLS = list(DEFAULT_COLUMN_MAP.keys())


def qmt_to_dpoint_single(df: pd.DataFrame) -> pd.DataFrame:
    """
    转换 QMT DataFrame 为 Dpoint_Trader 单股格式。

    Args:
        df: QMT 返回的 DataFrame，列名为 time/open/high/low/close/volume/amount

    Returns:
        Dpoint_Trader 格式 DataFrame，列为 date/open_qfq/high_qfq/low_qfq/close_qfq/volume/amount
    """
    missing = set(_QMT_REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"缺少必需列: {missing}")

    if df.empty:
        return pd.DataFrame(columns=_INTERNAL_COLS)

    dates = pd.to_datetime(df["time"], unit="ms", errors="coerce")

    # 检查并记录无效时间戳
    nat_count = dates.isna().sum()
    if nat_count > 0:
        logger.warning("发现 %d 个无效时间戳（已转为 NaT）", nat_count)

    result = pd.DataFrame({
        COL_DATE: dates,
        COL_OPEN: df["open"].values,
        COL_HIGH: df["high"].values,
        COL_LOW: df["low"].values,
        COL_CLOSE: df["close"].values,
        COL_VOLUME: df["volume"].values,
        COL_AMOUNT: df["amount"].values if "amount" in df.columns else 0.0,
    })

    # 按日期排序，NaT 排到末尾
    result = result.sort_values(COL_DATE).reset_index(drop=True)
    return result


def qmt_to_dpoint_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    转换 QMT DataFrame 为 Dpoint_Trader 篮子 CSV 格式。

    列名使用 Dpoint_Trader 的外部映射名（参见 core/constants.py DEFAULT_COLUMN_MAP）。
    日期格式为 YYYY/M/D（无前导零），与现有 basket_1/*.csv 一致。

    Args:
        df: QMT 返回的 DataFrame

    Returns:
        篮子 CSV 格式 DataFrame
    """
    missing = set(_QMT_REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"缺少必需列: {missing}")

    if df.empty:
        return pd.DataFrame(columns=_CSV_COLS)

    dates = pd.to_datetime(df["time"], unit="ms", errors="coerce")

    # 检查并记录无效时间戳
    nat_count = dates.isna().sum()
    if nat_count > 0:
        logger.warning("发现 %d 个无效时间戳（已转为 NaT）", nat_count)

    # NaT 的 strftime 会产生 NaN，用空字符串替换
    date_strs = dates.dt.strftime(_DATE_FMT)
    date_strs = date_strs.fillna("")

    result = pd.DataFrame({
        "Date": date_strs,
        "Open (CNY, qfq)": df["open"].values,
        "High (CNY, qfq)": df["high"].values,
        "Low (CNY, qfq)": df["low"].values,
        "Close (CNY, qfq)": df["close"].values,
        "Volume (shares)": df["volume"].values,
    })

    return result


def generate_csv_filename(stock_code: str, start_date: str) -> str:
    """
    生成篮子 CSV 文件名。

    格式: {6位代码}_{日期}.csv
    示例: 000001_19910403.csv

    Args:
        stock_code: 股票代码，如 "000001.SZ" 或 "000001"
        start_date: 起始日期，格式 "YYYYMMDD"

    Returns:
        文件名字符串
    """
    code = stock_code.split(".")[0] if "." in stock_code else stock_code
    return f"{code}_{start_date}.csv"
