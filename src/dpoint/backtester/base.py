# base.py
"""
统一的 BacktestResult 数据类和风险指标计算。
单股和组合回测共用此接口。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ExecutionStats:
    """执行统计。"""

    orders_submitted: int = 0
    orders_filled: int = 0
    orders_rejected: int = 0
    reject_reasons: Dict[str, int] = field(default_factory=dict)
    total_slippage_cost: float = 0.0

    def add_reject(self, reason: str):
        self.orders_rejected += 1
        self.reject_reasons[reason] = self.reject_reasons.get(reason, 0) + 1

    def add_fill(self, slippage_cost: float):
        self.orders_filled += 1
        self.total_slippage_cost += slippage_cost

    def to_dict(self) -> dict:
        return {
            "orders_submitted": self.orders_submitted,
            "orders_filled": self.orders_filled,
            "orders_rejected": self.orders_rejected,
            "reject_reasons": self.reject_reasons,
            "total_slippage_cost": self.total_slippage_cost,
        }


@dataclass
class BacktestResult:
    """统一的回测结果。"""

    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    risk_metrics: Dict[str, float]
    execution_stats: ExecutionStats
    notes: List[str] = field(default_factory=list)
    # 篮子模式额外字段
    orders: Optional[pd.DataFrame] = None
    positions: Optional[pd.DataFrame] = None
    benchmark: Optional[pd.DataFrame] = None


def compute_risk_metrics(
    equity_curve: pd.DataFrame, initial_cash: float = 100_000.0
) -> Dict[str, float]:
    """
    从净值曲线计算风险指标。

    Returns:
        dict with keys: total_return, annual_return, annual_vol, sharpe, sortino,
        max_drawdown, max_drawdown_days, calmar, win_rate, profit_factor
    """
    if equity_curve.empty or "total_equity" not in equity_curve.columns:
        return {}

    equity = equity_curve["total_equity"].values
    n_days = len(equity)
    if n_days < 2:
        return {}

    # 日收益率
    returns = np.diff(equity) / equity[:-1]
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return {}

    # 总收益
    total_return = equity[-1] / initial_cash - 1.0

    # 年化（假设252个交易日）
    years = n_days / 252.0
    annual_return = (1 + total_return) ** (1 / max(years, 0.01)) - 1 if total_return > -1 else -1.0

    # 年化波动率
    annual_vol = float(np.std(returns) * np.sqrt(252))

    # 夏普比率（假设无风险利率2%）
    rf_daily = 0.02 / 252
    excess = returns - rf_daily
    sharpe = float(np.mean(excess) / np.std(excess) * np.sqrt(252)) if np.std(excess) > 0 else 0.0

    # Sortino
    downside = returns[returns < 0]
    downside_std = float(np.std(downside)) if len(downside) > 1 else 1e-10
    sortino = float((np.mean(returns) - rf_daily) / max(downside_std, 1e-10) * np.sqrt(252))

    # 最大回撤
    cum_max = np.maximum.accumulate(equity)
    drawdown = equity / cum_max - 1.0
    max_dd = float(np.min(drawdown))

    # 最大回撤持续天数
    dd_end = int(np.argmin(drawdown))
    dd_start = int(np.argmax(equity[: dd_end + 1])) if dd_end > 0 else 0
    max_dd_days = dd_end - dd_start

    # Calmar
    calmar = annual_return / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0

    # 胜率和盈亏比（从交易记录）
    win_rate = 0.0
    profit_factor = 0.0
    if "trades" in equity_curve.columns:
        pass  # 需要 trades DataFrame

    return {
        "total_return": round(total_return, 6),
        "annual_return": round(annual_return, 6),
        "annual_vol": round(annual_vol, 6),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_drawdown": round(max_dd, 6),
        "max_drawdown_days": max_dd_days,
        "calmar": round(calmar, 4),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4),
        "n_days": n_days,
    }
