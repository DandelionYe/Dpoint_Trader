# conftest.py
"""测试公共 fixtures。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# 确保 src 在路径中
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_single_df():
    """单股测试数据（600698 格式）。"""
    rng = np.random.Generator(np.random.PCG64(42))
    n = 200
    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    close = 10.0 * np.exp(np.cumsum(rng.normal(0.001, 0.02, n)))
    return pd.DataFrame(
        {
            "date": dates,
            "open_qfq": close * (1 + rng.normal(0, 0.005, n)),
            "high_qfq": close * (1 + np.abs(rng.normal(0, 0.01, n))),
            "low_qfq": close * (1 - np.abs(rng.normal(0, 0.01, n))),
            "close_qfq": close,
            "volume": rng.integers(100000, 1000000, n).astype(float),
            "amount": rng.uniform(1e6, 1e7, n),
            "turnover_rate": rng.uniform(0.01, 0.1, n),
        }
    )


@pytest.fixture
def sample_panel_df():
    """面板测试数据（3只股票）。"""
    rng = np.random.Generator(np.random.PCG64(42))
    n_dates = 200
    tickers = ["600036", "600519", "000858"]
    dates = pd.date_range("2021-01-01", periods=n_dates, freq="B")

    frames = []
    for ticker in tickers:
        close = (10 + hash(ticker) % 50) * np.exp(np.cumsum(rng.normal(0.001, 0.02, n_dates)))
        df = pd.DataFrame(
            {
                "date": dates,
                "ticker": ticker,
                "open_qfq": close * (1 + rng.normal(0, 0.005, n_dates)),
                "high_qfq": close * (1 + np.abs(rng.normal(0, 0.01, n_dates))),
                "low_qfq": close * (1 - np.abs(rng.normal(0, 0.01, n_dates))),
                "close_qfq": close,
                "volume": rng.integers(100000, 1000000, n_dates).astype(float),
            }
        )
        frames.append(df)

    return pd.concat(frames, ignore_index=True)
