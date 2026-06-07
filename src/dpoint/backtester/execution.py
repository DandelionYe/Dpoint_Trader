# execution.py
"""
A股执行引擎：涨跌停、T+1、佣金、滑点、手数约束。
来自 Ver2.0 和 Ver1.0 的执行层实现。
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from dpoint.core.constants import (
    DEFAULT_BOARD_LOT, DEFAULT_BUY_COMMISSION_RATE, DEFAULT_LIMIT_DOWN_PCT,
    DEFAULT_LIMIT_DOWN_PCT_ST, DEFAULT_LIMIT_UP_PCT, DEFAULT_LIMIT_UP_PCT_ST,
    DEFAULT_MAX_PARTICIPATION_RATE, DEFAULT_SELL_COMMISSION_RATE,
    DEFAULT_SELL_STAMP_DUTY_RATE, DEFAULT_SLIPPAGE_BPS,
)
from dpoint.backtester.base import ExecutionStats


def apply_slippage(price: float, action: str, slippage_bps: float = DEFAULT_SLIPPAGE_BPS) -> float:
    """应用固定滑点。买入上浮，卖出下浮。"""
    if price <= 0:
        return price
    slippage = price * slippage_bps / 10000.0
    return price + slippage if action == "BUY" else price - slippage


def check_limit(
    open_price: float, prev_close: float, action: str,
    limit_up_pct: float = DEFAULT_LIMIT_UP_PCT,
    limit_down_pct: float = DEFAULT_LIMIT_DOWN_PCT,
) -> Tuple[bool, str]:
    """检查涨跌停限制。返回 (可执行, 原因)。"""
    if prev_close <= 0:
        return True, ""
    limit_up = prev_close * (1 + limit_up_pct)
    limit_down = prev_close * (1 - limit_down_pct)
    if action == "BUY" and open_price >= limit_up:
        return False, "涨停买不到"
    if action == "SELL" and open_price <= limit_down:
        return False, "跌停卖不掉"
    return True, ""


def calc_buy_shares(
    cash: float, price: float, commission_rate: float = DEFAULT_BUY_COMMISSION_RATE,
    board_lot: int = DEFAULT_BOARD_LOT,
) -> int:
    """计算可买入手数（100股整数倍）。"""
    if price <= 0:
        return 0
    cost_per_share = price * (1 + commission_rate)
    max_shares = int(cash / cost_per_share)
    return (max_shares // board_lot) * board_lot


def calc_buy_cost(shares: int, price: float, commission_rate: float = DEFAULT_BUY_COMMISSION_RATE) -> float:
    """计算买入总成本。"""
    return shares * price * (1 + commission_rate)


def calc_sell_proceeds(
    shares: int, price: float,
    commission_rate: float = DEFAULT_SELL_COMMISSION_RATE,
    stamp_duty: float = DEFAULT_SELL_STAMP_DUTY_RATE,
) -> float:
    """计算卖出实收。"""
    return shares * price * (1 - commission_rate - stamp_duty)


def execute_order(
    action: str,
    shares: int,
    price: float,
    prev_close: float,
    cash: float,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    commission_rate_buy: float = DEFAULT_BUY_COMMISSION_RATE,
    commission_rate_sell: float = DEFAULT_SELL_COMMISSION_RATE,
    stamp_duty: float = DEFAULT_SELL_STAMP_DUTY_RATE,
    limit_up_pct: float = DEFAULT_LIMIT_UP_PCT,
    limit_down_pct: float = DEFAULT_LIMIT_DOWN_PCT,
    stats: ExecutionStats | None = None,
) -> Tuple[float, int, float, str]:
    """
    执行订单。

    Returns:
        (成交价, 成交股数, 滑点成本, 状态)
    """
    if stats is None:
        stats = ExecutionStats()
    stats.orders_submitted += 1

    # 检查涨跌停
    can_exec, reason = check_limit(price, prev_close, action, limit_up_pct, limit_down_pct)
    if not can_exec:
        stats.add_reject(reason)
        return 0.0, 0, 0.0, reason

    # 应用滑点
    exec_price = apply_slippage(price, action, slippage_bps)
    slippage_cost = abs(exec_price - price) * shares

    if action == "BUY":
        # 检查现金
        actual_shares = calc_buy_shares(cash, exec_price, commission_rate_buy)
        if actual_shares == 0:
            stats.add_reject("现金不足")
            return 0.0, 0, 0.0, "现金不足"
        actual_shares = min(actual_shares, shares)
        cost = calc_buy_cost(actual_shares, exec_price, commission_rate_buy)
        if cost > cash:
            actual_shares = calc_buy_shares(cash, exec_price, commission_rate_buy)
        stats.add_fill(slippage_cost)
        return exec_price, actual_shares, slippage_cost, "filled"

    else:  # SELL
        proceeds = calc_sell_proceeds(shares, exec_price, commission_rate_sell, stamp_duty)
        stats.add_fill(slippage_cost)
        return exec_price, shares, slippage_cost, "filled"
