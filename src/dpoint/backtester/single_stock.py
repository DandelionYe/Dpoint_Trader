# single_stock.py
"""
单股回测引擎：全仓/空仓切换模式。
来自 Ver2.0 的 backtester_engine.py，简化但保留核心逻辑。
"""

from __future__ import annotations

import logging
from typing import Dict

import pandas as pd

from dpoint.backtester.base import BacktestResult, ExecutionStats, compute_risk_metrics
from dpoint.backtester.execution import (
    apply_slippage,
    calc_buy_cost,
    calc_buy_shares,
    calc_sell_proceeds,
    check_limit,
)
from dpoint.core.constants import (
    DEFAULT_BUY_COMMISSION_RATE,
    DEFAULT_LIMIT_DOWN_PCT,
    DEFAULT_LIMIT_UP_PCT,
    DEFAULT_SELL_COMMISSION_RATE,
    DEFAULT_SELL_STAMP_DUTY_RATE,
    DEFAULT_SLIPPAGE_BPS,
)

logger = logging.getLogger(__name__)


def build_signal_frame(
    df: pd.DataFrame,
    dpoint: pd.Series,
    buy_threshold: float = 0.55,
    sell_threshold: float = 0.45,
    confirm_days: int = 1,
) -> pd.DataFrame:
    """
    构建信号帧（无状态）。
    信号在 t 日收盘后生成，t+1 日开盘执行。
    """
    df = df.copy()
    if "date" in df.columns:
        df = df.set_index("date", drop=False)

    # 对齐 dpoint 到 df index
    dpoint_aligned = dpoint.reindex(df.index)

    df["dpoint"] = dpoint_aligned
    df["raw_buy_signal"] = (df["dpoint"] >= buy_threshold).astype(int)
    df["raw_sell_signal"] = (df["dpoint"] <= sell_threshold).astype(int)

    # confirm_days 连续确认
    if confirm_days > 1:
        df["buy_confirmed"] = (
            df["raw_buy_signal"].rolling(confirm_days).sum() == confirm_days
        ).astype(int)
        df["sell_confirmed"] = (
            df["raw_sell_signal"].rolling(confirm_days).sum() == confirm_days
        ).astype(int)
    else:
        df["buy_confirmed"] = df["raw_buy_signal"]
        df["sell_confirmed"] = df["raw_sell_signal"]

    return df


def backtest_from_dpoint(
    df: pd.DataFrame,
    dpoint: pd.Series,
    *,
    initial_cash: float = 100_000.0,
    buy_threshold: float = 0.55,
    sell_threshold: float = 0.45,
    confirm_days: int = 1,
    max_hold_days: int = 20,
    min_hold_days: int = 1,
    take_profit: float = 0.0,
    stop_loss: float = 0.0,
    commission_rate_buy: float = DEFAULT_BUY_COMMISSION_RATE,
    commission_rate_sell: float = DEFAULT_SELL_COMMISSION_RATE,
    stamp_duty: float = DEFAULT_SELL_STAMP_DUTY_RATE,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    limit_up_pct: float = DEFAULT_LIMIT_UP_PCT,
    limit_down_pct: float = DEFAULT_LIMIT_DOWN_PCT,
) -> BacktestResult:
    """
    单股回测：信号在 t 日收盘生成，t+1 日开盘价执行。

    Args:
        df: 含 date/open_qfq/close_qfq 的行情 DataFrame
        dpoint: 预测概率序列，index 为日期

    Returns:
        BacktestResult
    """
    df = df.sort_values("date").reset_index(drop=True).copy()
    df["date"] = pd.to_datetime(df["date"])

    dpoint = dpoint.copy()
    dpoint.index = pd.to_datetime(dpoint.index)

    # 构建信号帧
    signal_df = build_signal_frame(df, dpoint, buy_threshold, sell_threshold, confirm_days)

    # 逐日执行模拟
    cash = initial_cash
    shares = 0
    entry_price = 0.0
    hold_days = 0
    trade_rows = []
    equity_rows = []
    stats = ExecutionStats()
    notes = []

    for i in range(1, len(signal_df)):
        row = signal_df.iloc[i]
        prev_row = signal_df.iloc[i - 1]
        dt = row["date"]
        open_price = float(row.get("open_qfq", 0))
        close_price = float(row.get("close_qfq", 0))
        prev_close = float(prev_row.get("close_qfq", open_price))

        # t-1 日的信号决定今日操作
        buy_signal = bool(prev_row.get("buy_confirmed", 0))
        sell_signal = bool(prev_row.get("sell_confirmed", 0))

        # 持仓状态
        in_position = shares > 0
        if in_position:
            hold_days += 1

        # 决策
        action = None
        if in_position:
            # 卖出条件
            should_sell = False
            sell_reason = ""

            if sell_signal:
                should_sell = True
                sell_reason = "sell_signal"
            elif max_hold_days > 0 and hold_days >= max_hold_days:
                should_sell = True
                sell_reason = "max_hold"
            elif take_profit > 0 and (open_price / entry_price - 1) >= take_profit:
                should_sell = True
                sell_reason = "take_profit"
            elif stop_loss > 0 and (open_price / entry_price - 1) <= -stop_loss:
                should_sell = True
                sell_reason = "stop_loss"

            if should_sell and hold_days >= min_hold_days:
                action = "SELL"
        else:
            # 买入条件
            if buy_signal:
                action = "BUY"

        # 执行
        if action == "BUY":
            can_exec, reason = check_limit(
                open_price, prev_close, "BUY", limit_up_pct, limit_down_pct
            )
            if can_exec:
                exec_price = apply_slippage(open_price, "BUY", slippage_bps)
                max_shares = calc_buy_shares(cash, exec_price, commission_rate_buy)
                if max_shares > 0:
                    cost = calc_buy_cost(max_shares, exec_price, commission_rate_buy)
                    if cost <= cash:
                        cash -= cost
                        shares = max_shares
                        entry_price = exec_price
                        hold_days = 0
                        slippage_cost = abs(exec_price - open_price) * max_shares
                        stats.add_fill(slippage_cost)
                        trade_rows.append(
                            {
                                "date": dt,
                                "action": "BUY",
                                "price": exec_price,
                                "shares": max_shares,
                                "cost": cost,
                                "reason": "buy_signal",
                            }
                        )
                    else:
                        stats.add_reject("现金不足")
                else:
                    stats.add_reject("现金不足")
            else:
                stats.add_reject(reason)

        elif action == "SELL":
            can_exec, reason = check_limit(
                open_price, prev_close, "SELL", limit_up_pct, limit_down_pct
            )
            if can_exec:
                exec_price = apply_slippage(open_price, "SELL", slippage_bps)
                proceeds = calc_sell_proceeds(shares, exec_price, commission_rate_sell, stamp_duty)
                pnl = proceeds - calc_buy_cost(shares, entry_price, commission_rate_buy)
                cash += proceeds
                slippage_cost = abs(exec_price - open_price) * shares
                stats.add_fill(slippage_cost)
                trade_rows.append(
                    {
                        "date": dt,
                        "action": "SELL",
                        "price": exec_price,
                        "shares": shares,
                        "proceeds": proceeds,
                        "pnl": pnl,
                        "reason": sell_reason if "sell_reason" in dir() else "sell_signal",
                    }
                )
                shares = 0
                entry_price = 0.0
                hold_days = 0
            else:
                stats.add_reject(reason)

        # 每日净值快照
        equity = cash + shares * close_price
        equity_rows.append(
            {
                "date": dt,
                "cash": cash,
                "shares": shares,
                "close_price": close_price,
                "total_equity": equity,
            }
        )

    # 组装结果
    equity_curve = pd.DataFrame(equity_rows)
    trades = pd.DataFrame(trade_rows)

    if not equity_curve.empty:
        equity_curve["cum_max_equity"] = equity_curve["total_equity"].cummax()
        equity_curve["drawdown"] = (
            equity_curve["total_equity"] / equity_curve["cum_max_equity"] - 1.0
        )

    risk_metrics = compute_risk_metrics(equity_curve, initial_cash)

    # 计算交易统计
    if not trades.empty:
        sell_trades = trades[trades["action"] == "SELL"]
        if not sell_trades.empty and "pnl" in sell_trades.columns:
            wins = (sell_trades["pnl"] > 0).sum()
            total = len(sell_trades)
            risk_metrics["n_trades"] = total
            risk_metrics["win_rate"] = round(wins / total, 4) if total > 0 else 0.0
            gross_profit = sell_trades.loc[sell_trades["pnl"] > 0, "pnl"].sum()
            gross_loss = abs(sell_trades.loc[sell_trades["pnl"] < 0, "pnl"].sum())
            risk_metrics["profit_factor"] = (
                round(gross_profit / gross_loss, 4) if gross_loss > 0 else float("inf")
            )

    return BacktestResult(
        equity_curve=equity_curve,
        trades=trades,
        risk_metrics=risk_metrics,
        execution_stats=stats,
        notes=notes,
    )


def compute_fold_metrics(result: BacktestResult) -> Dict[str, float]:
    """
    从回测结果计算单折评估指标。
    供搜索引擎的 evaluate_fn 使用。
    """
    metrics = result.risk_metrics
    n_trades = metrics.get("n_trades", 0)
    total_return = metrics.get("total_return", 0.0)

    # 几何均值净值比率（简化：用总收益 + 1）
    geom_mean_ratio = 1.0 + total_return if total_return > -1 else 0.001

    return {
        "geom_mean_ratio": geom_mean_ratio,
        "min_fold_ratio": geom_mean_ratio,  # 单折时等于自身
        "n_trades": n_trades,
        "total_return": total_return,
        "sharpe": metrics.get("sharpe", 0.0),
        "max_drawdown": metrics.get("max_drawdown", 0.0),
        "win_rate": metrics.get("win_rate", 0.0),
    }
