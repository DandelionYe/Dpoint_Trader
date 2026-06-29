# cross_sectional.py
"""
横截面排名特征：在每个交易日内跨股票计算相对排名。
来自 DpointTrader_deeplearning_Ver1.0/cross_sectional_features.py。
篮子模式独占，单股模式跳过。
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_cross_sectional_features(
    df: pd.DataFrame,
    feature_names: List[str],
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    methods: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    为每个时序特征添加横截面排名变体。

    Args:
        df: 包含时序特征的面板 DataFrame
        feature_names: 需要计算横截面特征的列名列表
        methods: 排名方法列表，默认 ["rank", "zscore"]

    Returns:
        (添加了横截面特征的 DataFrame, 新特征名列表)
    """
    if methods is None:
        methods = ["rank", "zscore"]

    new_feature_names = []
    result_df = df.copy()

    for method in methods:
        for feat in feature_names:
            if feat not in result_df.columns:
                continue
            new_col = f"{feat}_cs_{method}"
            if method == "rank":
                result_df[new_col] = result_df.groupby(date_col)[feat].rank(pct=True)
            elif method == "zscore":
                grouped = result_df.groupby(date_col)[feat]
                result_df[new_col] = (
                    result_df[feat] - grouped.transform("mean")
                ) / grouped.transform("std").replace(0, np.nan)
            new_feature_names.append(new_col)

    logger.info(
        "Added %d cross-sectional features (%d base × %d methods)",
        len(new_feature_names),
        len(feature_names),
        len(methods),
    )

    return result_df, new_feature_names
