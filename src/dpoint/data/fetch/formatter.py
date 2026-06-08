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

import sys

import pandas as pd

# Windows strftime 不支持 %-m/%-d，需用 %#m/%#d
_DATE_FMT = "%Y/%#m/%#d" if sys.platform == "win32" else "%Y/%-m/%-d"


def qmt_to_dpoint_single(df: pd.DataFrame) -> pd.DataFrame:
    """
    转换 QMT DataFrame 为 Dpoint_Trader 单股格式。

    Args:
        df: QMT 返回的 DataFrame，列名为 time/open/high/low/close/volume/amount

    Returns:
        Dpoint_Trader 格式 DataFrame，列为 date/open_qfq/high_qfq/low_qfq/close_qfq/volume/amount
    """
    if df.empty:
        return pd.DataFrame(
            columns=["date", "open_qfq", "high_qfq", "low_qfq", "close_qfq", "volume", "amount"]
        )

    result = pd.DataFrame()
    result["date"] = pd.to_datetime(df["time"], unit="ms", errors="coerce")
    result["open_qfq"] = df["open"].values
    result["high_qfq"] = df["high"].values
    result["low_qfq"] = df["low"].values
    result["close_qfq"] = df["close"].values
    result["volume"] = df["volume"].values
    if "amount" in df.columns:
        result["amount"] = df["amount"].values

    # 按日期排序
    result = result.sort_values("date").reset_index(drop=True)
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
    if df.empty:
        return pd.DataFrame(
            columns=[
                "Date",
                "Open (CNY, qfq)",
                "High (CNY, qfq)",
                "Low (CNY, qfq)",
                "Close (CNY, qfq)",
                "Volume (shares)",
            ]
        )

    dates = pd.to_datetime(df["time"], unit="ms", errors="coerce")

    result = pd.DataFrame()
    result["Date"] = dates.dt.strftime(_DATE_FMT)
    result["Open (CNY, qfq)"] = df["open"].values
    result["High (CNY, qfq)"] = df["high"].values
    result["Low (CNY, qfq)"] = df["low"].values
    result["Close (CNY, qfq)"] = df["close"].values
    result["Volume (shares)"] = df["volume"].astype(float).values

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
