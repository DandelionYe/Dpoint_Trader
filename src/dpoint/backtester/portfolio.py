# portfolio.py
"""
组合回测引擎：多股票 Top-K 选股 + 调仓。
来自 Ver1.0 的 portfolio_backtester.py。
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from dpoint.core.constants import (
    DEFAULT_BUY_COMMISSION_RATE, DEFAULT_CASH_BUFFER, DEFAULT_LIMIT_DOWN_PCT,
    DEFAULT_LIMIT_UP_PCT, DEFAULT_MAX_WEIGHT, DEFAULT_REBALANCE_FREQ,
    DEFAULT_SELL_COMMISSION_RATE, DEFAULT_SELL_STAMP_DUTY_RATE,
    DEFAULT_SLIPPAGE_BPS, DEFAULT_TOP_K, DEFAULT_WEIGHTING,
)
from dpoint.backtester.base import BacktestResult, ExecutionStats, compute_risk_metrics
from dpoint.backtester.execution import apply_slippage, calc_buy_shares, calc_buy_cost, calc_sell_proceeds

logger = logging.getLogger(__name__)


def select_top_k(
    scores: pd.DataFrame,
    top_k: int = DEFAULT_TOP_K,
    weighting: str = DEFAULT_WEIGHTING,
    max_weight: float = DEFAULT_MAX_WEIGHT,
) -> Dict[str, float]:
    """
    从横截面分数中选择 Top-K 股票并分配权重。

    Args:
        scores: 含 ticker 和 score 列的 DataFrame
        top_k: 选股数
        weighting: equal / score / vol_inv
        max_weight: 单票最大权重

    Returns:
        {ticker: weight} 字典
    """
    if scores.empty or "score" not in scores.columns:
        return {}

    ranked = scores.nlargest(top_k, "score")

    if weighting == "equal":
        w = 1.0 / len(ranked)
        weights = {row["ticker"]: w for _, row in ranked.iterrows()}
    elif weighting == "score":
        total_score = ranked["score"].sum()
        if total_score <= 0:
            w = 1.0 / len(ranked)
            weights = {row["ticker"]: w for _, row in ranked.iterrows()}
        else:
            weights = {row["ticker"]: row["score"] / total_score for _, row in ranked.iterrows()}
    else:  # vol_inv
        w = 1.0 / len(ranked)
        weights = {row["ticker"]: w for _, row in ranked.iterrows()}

    # 限制最大权重
    for t in weights:
        weights[t] = min(weights[t], max_weight)

    # 归一化
    total = sum(weights.values())
    if total > 0:
        weights = {t: w / total for t, w in weights.items()}

    return weights


def backtest_from_scores(
    panel_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    *,
    initial_cash: float = 100_000.0,
    top_k: int = DEFAULT_TOP_K,
    weighting: str = DEFAULT_WEIGHTING,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    rebalance_freq: str = DEFAULT_REBALANCE_FREQ,
    cash_buffer: float = DEFAULT_CASH_BUFFER,
    commission_rate_buy: float = DEFAULT_BUY_COMMISSION_RATE,
    commission_rate_sell: float = DEFAULT_SELL_COMMISSION_RATE,
    stamp_duty: float = DEFAULT_SELL_STAMP_DUTY_RATE,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> BacktestResult:
    """
    组合回测：按日横截面选股，调仓执行。

    Args:
        panel_df: 面板行情数据（含 date/ticker/open_qfq/close_qfq）
        scores_df: 预测分数（含 date/ticker/score）

    Returns:
        BacktestResult
    """
    stats = ExecutionStats()
    notes = []

    # 准备日期列表
    dates = sorted(panel_df["date"].unique())
    scores_df = scores_df.copy()
    scores_df["date"] = pd.to_datetime(scores_df["date"])

    # 持仓簿
    positions = {}  # ticker -> {"shares": int, "entry_price": float}
    cash = initial_cash
    equity_rows = []

    # 判断调仓日
    if rebalance_freq == "daily":
        rebalance_dates = set(dates)
    elif rebalance_freq == "weekly":
        rebalance_dates = set(dates[::5])
    else:  # monthly
        rebalance_dates = set(dates[::21])

    for dt in dates:
        day_data = panel_df[panel_df["date"] == dt].set_index("ticker")
        day_scores = scores_df[scores_df["date"] == dt]

        if dt in rebalance_dates and not day_scores.empty:
            # 选股
            target_weights = select_top_k(day_scores, top_k, weighting, max_weight)

            # 计算当前总市值
            portfolio_value = cash
            for ticker, pos in positions.items():
                if ticker in day_data.index:
                    close = float(day_data.loc[ticker, "close_qfq"])
                    portfolio_value += pos["shares"] * close

            # 先卖出不在目标中的持仓
            for ticker in list(positions.keys()):
                if ticker not in target_weights:
                    if ticker in day_data.index:
                        open_p = float(day_data.loc[ticker, "open_qfq"])
                        exec_price = apply_slippage(open_p, "SELL", slippage_bps)
                        proceeds = calc_sell_proceeds(positions[ticker]["shares"], exec_price, commission_rate_sell, stamp_duty)
                        cash += proceeds
                        stats.add_fill(abs(exec_price - open_p) * positions[ticker]["shares"])
                    del positions[ticker]

            # 买入目标持仓
            available_cash = cash * (1 - cash_buffer)
            for ticker, weight in target_weights.items():
                target_value = portfolio_value * weight
                current_value = 0
                if ticker in positions and ticker in day_data.index:
                    current_value = positions[ticker]["shares"] * float(day_data.loc[ticker, "close_qfq"])

                if target_value > current_value + 100:
                    buy_value = target_value - current_value
                    if ticker in day_data.index and buy_value > 0:
                        open_p = float(day_data.loc[ticker, "open_qfq"])
                        exec_price = apply_slippage(open_p, "BUY", slippage_bps)
                        shares_to_buy = calc_buy_shares(min(buy_value, available_cash), exec_price, commission_rate_buy)
                        if shares_to_buy > 0:
                            cost = calc_buy_cost(shares_to_buy, exec_price, commission_rate_buy)
                            if cost <= available_cash:
                                if ticker in positions:
                                    # 加仓：更新均价
                                    old = positions[ticker]
                                    total_shares = old["shares"] + shares_to_buy
                                    avg_price = (old["entry_price"] * old["shares"] + exec_price * shares_to_buy) / total_shares
                                    positions[ticker] = {"shares": total_shares, "entry_price": avg_price}
                                else:
                                    positions[ticker] = {"shares": shares_to_buy, "entry_price": exec_price}
                                cash -= cost
                                available_cash -= cost
                                stats.add_fill(abs(exec_price - open_p) * shares_to_buy)

        # 每日净值
        portfolio_value = cash
        for ticker, pos in positions.items():
            if ticker in day_data.index:
                close = float(day_data.loc[ticker, "close_qfq"])
                portfolio_value += pos["shares"] * close

        equity_rows.append({
            "date": dt,
            "cash": cash,
            "n_positions": len(positions),
            "total_equity": portfolio_value,
        })

    # 组装结果
    equity_curve = pd.DataFrame(equity_rows)
    if not equity_curve.empty:
        equity_curve["cum_max_equity"] = equity_curve["total_equity"].cummax()
        equity_curve["drawdown"] = equity_curve["total_equity"] / equity_curve["cum_max_equity"] - 1.0

    risk_metrics = compute_risk_metrics(equity_curve, initial_cash)

    return BacktestResult(
        equity_curve=equity_curve,
        trades=pd.DataFrame(),
        risk_metrics=risk_metrics,
        execution_stats=stats,
        notes=notes,
    )
