# pipeline.py
"""
特征工程统一编排入口。
兼容单股/面板两种模式：单股模式退化为单 ticker 面板。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from dpoint.core.config import FeatureConfig
from dpoint.features.cross_sectional import add_cross_sectional_features
from dpoint.features.groups import add_all_features
from dpoint.features.labeler import build_label

logger = logging.getLogger(__name__)


@dataclass
class FeatureMeta:
    """特征元数据。"""
    feature_names: List[str] = field(default_factory=list)
    ts_feature_names: List[str] = field(default_factory=list)  # 时序特征
    cs_feature_names: List[str] = field(default_factory=list)  # 横截面特征
    n_features: int = 0
    n_samples: int = 0
    n_tickers: int = 1
    label_mode: str = ""
    params: Dict[str, Any] = field(default_factory=dict)


def build_features_and_labels(
    df: pd.DataFrame,
    config: FeatureConfig,
    *,
    label_mode: str = "binary_next_close_up",
    horizon_days: int = 1,
    ticker_col: str = "ticker",
    date_col: str = "date",
    close_col: str = "close_qfq",
    mode: str = "single",  # single / basket
) -> Tuple[pd.DataFrame, pd.Series, FeatureMeta]:
    """
    统一的特征工程管道。

    Args:
        df: 输入 DataFrame（单股或面板）
        config: 特征配置
        label_mode: 标签模式
        horizon_days: 预测天数
        mode: "single"（单股）或 "basket"（篮子）

    Returns:
        (带特征的 DataFrame, 标签 Series, 元数据)
    """
    # 单股模式：如果没有 ticker 列，添加一个虚拟的
    if ticker_col not in df.columns:
        df = df.copy()
        df[ticker_col] = "SINGLE"

    # 1. 时序特征
    df, ts_meta = add_all_features(
        df,
        ticker_col=ticker_col,
        windows=config.windows,
        use_momentum=config.use_momentum,
        use_volatility=config.use_volatility,
        use_volume=config.use_volume,
        use_candle=config.use_candle,
        use_turnover=config.use_turnover,
        use_ta_indicators=config.use_ta_indicators,
        vol_metric=config.vol_metric,
        liq_transform=config.liq_transform,
    )
    ts_feature_names = ts_meta.feature_names

    # 2. 横截面特征（篮子模式独占）
    cs_feature_names = []
    if mode == "basket" and config.include_cross_section:
        df, cs_feature_names = add_cross_sectional_features(
            df, ts_feature_names, date_col=date_col, ticker_col=ticker_col,
        )

    all_feature_names = ts_feature_names + cs_feature_names

    # 3. 标签
    y = build_label(
        df, close_col=close_col, ticker_col=ticker_col,
        label_mode=label_mode, horizon_days=horizon_days,
    )

    # 4. 过滤 NaN 标签
    valid_mask = y.notna()
    df = df[valid_mask].copy()
    y = y[valid_mask]

    # 5. 过滤特征全 NaN 的行
    feature_nan_mask = df[all_feature_names].isna().all(axis=1)
    if feature_nan_mask.any():
        df = df[~feature_nan_mask].copy()
        y = y[~feature_nan_mask]

    n_tickers = df[ticker_col].nunique() if ticker_col in df.columns else 1

    meta = FeatureMeta(
        feature_names=all_feature_names,
        ts_feature_names=ts_feature_names,
        cs_feature_names=cs_feature_names,
        n_features=len(all_feature_names),
        n_samples=len(df),
        n_tickers=n_tickers,
        label_mode=label_mode,
        params={
            "windows": config.windows,
            "include_cross_section": config.include_cross_section and mode == "basket",
            "vol_metric": config.vol_metric,
            "liq_transform": config.liq_transform,
        },
    )

    logger.info(
        "Features built: %d samples, %d features (%d ts + %d cs), %d tickers",
        meta.n_samples, meta.n_features, len(ts_feature_names), len(cs_feature_names), n_tickers,
    )

    return df, y, meta
