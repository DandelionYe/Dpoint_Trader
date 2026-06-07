# groups.py
"""
时序特征组：动量、波动率、成交量、K线、换手率、技术指标。
来自 DpointTrader_deeplearning_Ver1.0/feature_groups.py。
所有 rolling 操作在 groupby(ticker) 内进行，避免跨 ticker 泄露。
单股模式下传入单 ticker 面板即可复用。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeatureGroupMeta:
    feature_names: List[str] = field(default_factory=list)
    n_features: int = 0


# =========================================================
# 工具函数
# =========================================================

def _apply_per_ticker(df: pd.DataFrame, ticker_col: str, func) -> pd.DataFrame:
    """在每个 ticker 组内独立应用函数。"""
    pieces = []
    for _, group in df.groupby(ticker_col, sort=False):
        pieces.append(func(group))
    if not pieces:
        return pd.DataFrame(index=df.index)
    return pd.concat(pieces).sort_index()


def _rolling_mad(x: pd.Series, window: int) -> pd.Series:
    med = x.rolling(window, min_periods=window).median()
    return (x - med).abs().rolling(window, min_periods=window).median()


def _rolling_zscore(x: pd.Series, window: int) -> pd.Series:
    mu = x.rolling(window, min_periods=window).mean()
    sd = x.rolling(window, min_periods=window).std()
    return (x - mu) / sd.replace(0, np.nan)


# =========================================================
# 动量特征
# =========================================================

def add_momentum_features(
    df: pd.DataFrame, *, ticker_col: str = "ticker", close_col: str = "close_qfq",
    windows: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    if windows is None:
        windows = [5, 10, 20]
    feature_names = []

    def calc(group):
        close = group[close_col]
        result = pd.DataFrame(index=group.index)
        for w in windows:
            name = f"ret_{w}"
            result[name] = close.pct_change(w)
            feature_names.append(name)
            name_ma = f"ma_{w}_ratio"
            result[name_ma] = close / close.rolling(w, min_periods=w).mean() - 1
            feature_names.append(name_ma)
        return result

    features = _apply_per_ticker(df, ticker_col, calc)
    return pd.concat([df, features], axis=1), feature_names


# =========================================================
# 波动率特征
# =========================================================

def add_volatility_features(
    df: pd.DataFrame, *, ticker_col: str = "ticker",
    high_col: str = "high_qfq", low_col: str = "low_qfq",
    close_col: str = "close_qfq", open_col: str = "open_qfq",
    windows: Optional[List[int]] = None, vol_metric: str = "std",
) -> Tuple[pd.DataFrame, List[str]]:
    if windows is None:
        windows = [5, 10, 20]
    feature_names = []

    def calc(group):
        high, low, close, opn = group[high_col], group[low_col], group[close_col], group[open_col]
        result = pd.DataFrame(index=group.index)
        # 振幅
        result["hl_range"] = (high - low) / close.replace(0, np.nan)
        feature_names.append("hl_range")
        # 真实波幅
        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        result["true_range_norm"] = tr / close.replace(0, np.nan)
        feature_names.append("true_range_norm")
        # 滚动波动率
        ret = close.pct_change()
        for w in windows:
            if vol_metric == "mad":
                name = f"vol_mad_{w}"
                result[name] = _rolling_mad(ret, w)
            else:
                name = f"vol_std_{w}"
                result[name] = ret.rolling(w, min_periods=w).std()
            feature_names.append(name)
        return result

    features = _apply_per_ticker(df, ticker_col, calc)
    return pd.concat([df, features], axis=1), feature_names


# =========================================================
# 成交量特征
# =========================================================

def add_volume_features(
    df: pd.DataFrame, *, ticker_col: str = "ticker",
    volume_col: str = "volume", amount_col: str = "amount",
    windows: Optional[List[int]] = None, liq_transform: str = "ratio",
) -> Tuple[pd.DataFrame, List[str]]:
    if windows is None:
        windows = [5, 10, 20]
    feature_names = []

    def calc(group):
        vol = group[volume_col].astype(float)
        result = pd.DataFrame(index=group.index)
        result["log_volume"] = np.log1p(vol.clip(lower=0))
        feature_names.append("log_volume")
        for w in windows:
            ma = vol.rolling(w, min_periods=w).mean()
            if liq_transform == "zscore":
                name = f"vol_zscore_{w}"
                result[name] = _rolling_zscore(vol, w)
            else:
                name = f"vol_ma_{w}_ratio"
                result[name] = vol / ma.replace(0, np.nan) - 1
            feature_names.append(name)
        # amount 特征
        if amount_col in group.columns:
            amt = group[amount_col].astype(float)
            result["log_amount"] = np.log1p(amt.clip(lower=0))
            feature_names.append("log_amount")
        return result

    features = _apply_per_ticker(df, ticker_col, calc)
    return pd.concat([df, features], axis=1), feature_names


# =========================================================
# K线形态特征
# =========================================================

def add_candle_features(
    df: pd.DataFrame, *, ticker_col: str = "ticker",
    open_col: str = "open_qfq", high_col: str = "high_qfq",
    low_col: str = "low_qfq", close_col: str = "close_qfq",
) -> Tuple[pd.DataFrame, List[str]]:
    feature_names = []

    def calc(group):
        opn, high, low, close = group[open_col], group[high_col], group[low_col], group[close_col]
        result = pd.DataFrame(index=group.index)
        body = close - opn
        hl = (high - low).replace(0, np.nan)
        result["candle_body"] = body / hl
        result["candle_upper_shadow"] = (high - close.clip(upper=opn.where(close > opn, close))) / hl
        result["candle_lower_shadow"] = (close.clip(lower=opn.where(close < opn, close)) - low) / hl
        feature_names.extend(["candle_body", "candle_upper_shadow", "candle_lower_shadow"])
        return result

    features = _apply_per_ticker(df, ticker_col, calc)
    return pd.concat([df, features], axis=1), feature_names


# =========================================================
# 换手率特征
# =========================================================

def add_turnover_features(
    df: pd.DataFrame, *, ticker_col: str = "ticker",
    turnover_col: str = "turnover_rate",
    windows: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    if windows is None:
        windows = [5, 10, 20]
    feature_names = []
    if turnover_col not in df.columns:
        return df, feature_names

    def calc(group):
        tr = group[turnover_col].astype(float)
        result = pd.DataFrame(index=group.index)
        for w in windows:
            ma = tr.rolling(w, min_periods=w).mean()
            std = tr.rolling(w, min_periods=w).std()
            result[f"turnover_ma_{w}"] = ma
            result[f"turnover_zscore_{w}"] = (tr - ma) / std.replace(0, np.nan)
            feature_names.extend([f"turnover_ma_{w}", f"turnover_zscore_{w}"])
        return result

    features = _apply_per_ticker(df, ticker_col, calc)
    return pd.concat([df, features], axis=1), feature_names


# =========================================================
# 技术指标特征（P3-19）
# =========================================================

def add_ta_indicators(
    df: pd.DataFrame, *, ticker_col: str = "ticker",
    close_col: str = "close_qfq", high_col: str = "high_qfq",
    low_col: str = "low_qfq", volume_col: str = "volume",
    windows: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """RSI、MACD、布林带宽、OBV — 均不引入前向偏差。"""
    if windows is None:
        windows = [14, 28]
    feature_names = []

    def calc(group):
        close = group[close_col]
        high = group[high_col]
        low = group[low_col]
        vol = group[volume_col].astype(float)
        result = pd.DataFrame(index=group.index)

        # RSI
        for w in windows:
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(w, min_periods=w).mean()
            loss = (-delta.clip(upper=0)).rolling(w, min_periods=w).mean()
            rs = gain / loss.replace(0, np.nan)
            result[f"rsi_{w}"] = 100 - 100 / (1 + rs)
            result[f"rsi_{w}"] = result[f"rsi_{w}"] / 100  # 归一化到 [0,1]
            feature_names.append(f"rsi_{w}")

        # MACD (固定 12/26/9)
        ema12 = close.ewm(span=12, min_periods=12).mean()
        ema26 = close.ewm(span=26, min_periods=26).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, min_periods=9).mean()
        result["macd_line"] = _rolling_zscore(macd_line, 60)
        result["macd_hist"] = _rolling_zscore(macd_line - signal_line, 60)
        feature_names.extend(["macd_line", "macd_hist"])

        # 布林带宽
        for w in windows:
            sma = close.rolling(w, min_periods=w).mean()
            std = close.rolling(w, min_periods=w).std()
            result[f"bband_width_{w}"] = 2 * std / sma.replace(0, np.nan)
            feature_names.append(f"bband_width_{w}")

        # OBV
        sign = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv = (vol * sign).cumsum()
        result["obv_zscore"] = _rolling_zscore(obv, 60)
        feature_names.append("obv_zscore")

        return result

    features = _apply_per_ticker(df, ticker_col, calc)
    return pd.concat([df, features], axis=1), feature_names


# =========================================================
# 汇总：添加所有特征
# =========================================================

def add_all_features(
    df: pd.DataFrame,
    *,
    ticker_col: str = "ticker",
    windows: Optional[List[int]] = None,
    use_momentum: bool = True,
    use_volatility: bool = True,
    use_volume: bool = True,
    use_candle: bool = True,
    use_turnover: bool = True,
    use_ta_indicators: bool = True,
    vol_metric: str = "std",
    liq_transform: str = "ratio",
) -> Tuple[pd.DataFrame, FeatureGroupMeta]:
    """添加所有时序特征，返回 (df, meta)。"""
    all_feature_names = []

    if use_momentum:
        df, names = add_momentum_features(df, ticker_col=ticker_col, windows=windows)
        all_feature_names.extend(names)
    if use_volatility:
        df, names = add_volatility_features(df, ticker_col=ticker_col, windows=windows, vol_metric=vol_metric)
        all_feature_names.extend(names)
    if use_volume:
        df, names = add_volume_features(df, ticker_col=ticker_col, windows=windows, liq_transform=liq_transform)
        all_feature_names.extend(names)
    if use_candle:
        df, names = add_candle_features(df, ticker_col=ticker_col)
        all_feature_names.extend(names)
    if use_turnover:
        df, names = add_turnover_features(df, ticker_col=ticker_col, windows=windows)
        all_feature_names.extend(names)
    if use_ta_indicators:
        df, names = add_ta_indicators(df, ticker_col=ticker_col, windows=windows)
        all_feature_names.extend(names)

    meta = FeatureGroupMeta(feature_names=all_feature_names, n_features=len(all_feature_names))
    return df, meta
