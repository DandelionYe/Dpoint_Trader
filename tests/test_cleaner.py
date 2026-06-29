# test_cleaner.py
"""数据清洗测试。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dpoint.data.cleaner import clean_ohlcv, DataReport


def test_clean_basic(sample_single_df):
    report = DataReport()
    result = clean_ohlcv(sample_single_df, report)
    assert report.rows_after_clean > 0
    assert report.rows_after_clean <= report.rows_raw
    assert "date" in result.columns or result.index.name == "date"


def test_clean_filters_negative_prices():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2021-01-01", periods=5),
            "open_qfq": [10, -1, 10, 10, 10],
            "high_qfq": [11, 11, 11, 11, 11],
            "low_qfq": [9, 9, 9, 9, 9],
            "close_qfq": [10, 10, 10, 10, 10],
            "volume": [1000] * 5,
        }
    )
    report = DataReport()
    result = clean_ohlcv(df, report)
    assert report.rows_after_clean == 4


def test_clean_derives_amount():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2021-01-01", periods=5),
            "open_qfq": [10] * 5,
            "high_qfq": [11] * 5,
            "low_qfq": [9] * 5,
            "close_qfq": [10] * 5,
            "volume": [1000] * 5,
        }
    )
    report = DataReport()
    result = clean_ohlcv(df, report)
    assert "amount" in result.columns
    assert "amount" in report.derived_cols
