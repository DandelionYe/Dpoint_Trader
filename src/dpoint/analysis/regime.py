# regime.py
"""
市场状态检测与分层评估模块。
来自 Ver2.0/regime.py。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RegimeConfig:
    """Regime 检测参数。"""

    ma_short: int = 5
    ma_long: int = 20
    vol_window: int = 20
    vol_high_threshold: float = 0.20
    vol_low_threshold: float = 0.10


class RegimeDetector:
    """
    市场状态检测器。

    支持两种维度:
    - trend / non_trend: 基于均线斜率
    - high_vol / low_vol / medium_vol: 基于波动率
    - combined: trend + volatility 组合
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()

    def detect_trend(self, close: pd.Series) -> pd.Series:
        """检测趋势状态。trend=短期均线在长期均线之上。"""
        ma_s = close.rolling(self.config.ma_short, min_periods=self.config.ma_short).mean()
        ma_l = close.rolling(self.config.ma_long, min_periods=self.config.ma_long).mean()
        return pd.Series(
            np.where(ma_s > ma_l, "trend", "non_trend"),
            index=close.index,
        )

    def detect_volatility(self, close: pd.Series) -> pd.Series:
        """检测波动率状态。"""
        returns = close.pct_change()
        vol = returns.rolling(
            self.config.vol_window, min_periods=self.config.vol_window
        ).std() * np.sqrt(252)

        def classify(v):
            if np.isnan(v):
                return "unknown"
            if v >= self.config.vol_high_threshold:
                return "high_vol"
            elif v <= self.config.vol_low_threshold:
                return "low_vol"
            else:
                return "medium_vol"

        return vol.apply(classify)

    def detect_combined(self, close: pd.Series) -> pd.Series:
        """组合趋势和波动率。如 'trend_high_vol', 'non_trend_low_vol'。"""
        trend = self.detect_trend(close)
        vol = self.detect_volatility(close)
        return trend + "_" + vol

    def detect(self, close: pd.Series, mode: str = "combined") -> pd.Series:
        """统一入口。mode: trend / volatility / combined。"""
        if mode == "trend":
            return self.detect_trend(close)
        elif mode == "volatility":
            return self.detect_volatility(close)
        else:
            return self.detect_combined(close)


def compute_regime_metrics(
    equity_curve: pd.DataFrame,
    close: pd.Series,
    *,
    regime_mode: str = "combined",
    config: Optional[RegimeConfig] = None,
) -> Dict[str, Dict[str, float]]:
    """
    按 regime 分层计算回测指标。

    Args:
        equity_curve: 含 total_equity 的净值曲线
        close: 收盘价序列
        regime_mode: regime 检测模式

    Returns:
        {regime_name: {metric_name: value}}
    """
    detector = RegimeDetector(config)
    regimes = detector.detect(close, mode=regime_mode)

    # 对齐
    if len(regimes) != len(equity_curve):
        regimes = regimes.reindex(equity_curve.index)

    equity_curve = equity_curve.copy()
    equity_curve["regime"] = regimes.values

    results = {}
    for regime_name, group in equity_curve.groupby("regime"):
        if len(group) < 5:
            continue
        equity = group["total_equity"].values
        returns = np.diff(equity) / equity[:-1]
        returns = returns[np.isfinite(returns)]

        if len(returns) == 0:
            continue

        total_ret = equity[-1] / equity[0] - 1 if equity[0] > 0 else 0
        ann_vol = float(np.std(returns) * np.sqrt(252))
        sharpe = float(np.mean(returns) / max(np.std(returns), 1e-10) * np.sqrt(252))

        cum_max = np.maximum.accumulate(equity)
        max_dd = float(np.min(equity / cum_max - 1))

        results[str(regime_name)] = {
            "n_days": len(group),
            "total_return": round(total_ret, 6),
            "annual_vol": round(ann_vol, 6),
            "sharpe": round(sharpe, 4),
            "max_drawdown": round(max_dd, 6),
        }

    return results
