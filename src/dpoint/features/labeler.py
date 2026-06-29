# labeler.py
"""
标签构建器：支持 binary / multiclass / regression 三种模式。
来自 DpointTrader_deeplearning_Ver1.0/labeler.py。
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_label(
    df: pd.DataFrame,
    *,
    close_col: str = "close_qfq",
    ticker_col: str = "ticker",
    label_mode: str = "binary_next_close_up",
    horizon_days: int = 1,
) -> pd.Series:
    """
    构建标签列。

    Args:
        df: 面板 DataFrame（需包含 close_col 和 ticker_col）
        close_col: 收盘价列名
        ticker_col: ticker 列名
        label_mode: 标签模式
            - binary_next_close_up: 次日收盘价上涨为1，否则为0
            - multiclass_N: N分类（涨/平/跌）
            - regression_return: 次日收益率
        horizon_days: 预测天数（默认1）

    Returns:
        标签 Series
    """
    if ticker_col in df.columns:
        future_close = df.groupby(ticker_col)[close_col].shift(-horizon_days)
    else:
        future_close = df[close_col].shift(-horizon_days)

    if label_mode == "binary_next_close_up":
        label = (future_close > df[close_col]).astype(float)
    elif label_mode.startswith("multiclass_"):
        n_classes = int(label_mode.split("_")[-1])
        ret = future_close / df[close_col] - 1
        # 将收益率等分为 N 类
        quantiles = np.linspace(0, 1, n_classes + 1)
        bins = ret.quantile(quantiles[1:-1]).values
        label = np.digitize(ret, bins).astype(float)
    elif label_mode in ("regression_return", "regression"):
        label = future_close / df[close_col] - 1
    else:
        raise ValueError(f"Unknown label_mode: {label_mode}")

    n_valid = label.notna().sum()
    n_total = len(label)
    logger.info("Label '%s': %d/%d valid samples", label_mode, n_valid, n_total)

    return label
