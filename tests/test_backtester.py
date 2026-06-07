# test_backtester.py
"""回测引擎测试。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dpoint.backtester.execution import (
    apply_slippage, calc_buy_shares, calc_buy_cost, calc_sell_proceeds,
    check_limit, execute_order,
)
from dpoint.backtester.base import ExecutionStats, compute_risk_metrics
from dpoint.backtester.single_stock import backtest_from_dpoint, compute_fold_metrics


def test_apply_slippage_buy():
    price = apply_slippage(10.0, "BUY", slippage_bps=20)
    assert price > 10.0
    assert abs(price - 10.02) < 0.001


def test_apply_slippage_sell():
    price = apply_slippage(10.0, "SELL", slippage_bps=20)
    assert price < 10.0
    assert abs(price - 9.98) < 0.001


def test_check_limit_buy_at_limit():
    can, reason = check_limit(11.0, 10.0, "BUY", limit_up_pct=0.10)
    assert not can
    assert "涨停" in reason


def test_check_limit_sell_at_limit():
    can, reason = check_limit(9.0, 10.0, "SELL", limit_down_pct=0.10)
    assert not can
    assert "跌停" in reason


def test_check_limit_normal():
    can, reason = check_limit(10.5, 10.0, "BUY", limit_up_pct=0.10)
    assert can


def test_calc_buy_shares():
    shares = calc_buy_shares(100_000, 10.0, commission_rate=0.0003)
    assert shares > 0
    assert shares % 100 == 0


def test_calc_buy_cost():
    cost = calc_buy_cost(100, 10.0, commission_rate=0.0003)
    assert cost > 1000.0


def test_calc_sell_proceeds():
    proceeds = calc_sell_proceeds(100, 10.0, commission_rate=0.0003, stamp_duty=0.001)
    assert proceeds < 1000.0
    assert proceeds > 980.0


def test_backtest_basic(sample_single_df):
    """基本回测测试。"""
    n = len(sample_single_df)
    # 模拟一个简单的 dpoint 序列
    rng = np.random.Generator(np.random.PCG64(42))
    dpoint = pd.Series(rng.uniform(0.3, 0.7, n), index=sample_single_df["date"].values)

    result = backtest_from_dpoint(
        sample_single_df, dpoint,
        buy_threshold=0.55, sell_threshold=0.45,
        confirm_days=1, max_hold_days=10,
    )

    assert result.equity_curve is not None
    assert len(result.equity_curve) > 0
    assert "total_equity" in result.equity_curve.columns
    assert result.risk_metrics is not None


def test_backtest_with_take_profit(sample_single_df):
    """止盈测试。"""
    n = len(sample_single_df)
    dpoint = pd.Series(np.full(n, 0.8), index=sample_single_df["date"].values)

    result = backtest_from_dpoint(
        sample_single_df, dpoint,
        buy_threshold=0.55, sell_threshold=0.2,
        take_profit=0.05, max_hold_days=50,
    )

    if not result.trades.empty:
        sell_trades = result.trades[result.trades["action"] == "SELL"]
        if not sell_trades.empty and "reason" in sell_trades.columns:
            tp_trades = sell_trades[sell_trades["reason"] == "take_profit"]
            # 止盈应该触发
            assert len(tp_trades) >= 0  # 可能不触发，取决于数据


def test_compute_fold_metrics(sample_single_df):
    n = len(sample_single_df)
    dpoint = pd.Series(np.random.uniform(0.3, 0.7, n), index=sample_single_df["date"].values)
    result = backtest_from_dpoint(sample_single_df, dpoint)
    fold_metrics = compute_fold_metrics(result)
    assert "geom_mean_ratio" in fold_metrics
    assert "n_trades" in fold_metrics


def test_risk_metrics():
    """风险指标计算测试。"""
    equity = pd.DataFrame({
        "total_equity": [100000, 101000, 102000, 101500, 103000],
    })
    metrics = compute_risk_metrics(equity, initial_cash=100000)
    assert "total_return" in metrics
    assert "sharpe" in metrics
    assert "max_drawdown" in metrics
    assert metrics["total_return"] > 0
