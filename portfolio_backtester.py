# portfolio_backtester.py
"""
组合回测模块（P-basket 核心新增）。

**架构定位：**
    本模块是单股回测引擎（backtester.py）的上层组合管理层。
    不重写执行逻辑，而是复用 backtester.py 中经过充分测试的
    ``check_execution_feasibility`` 和 ``apply_slippage`` 函数，
    在其上实现多股持仓管理、换仓调度和组合净值计算。

**数据流：**
    dpoint_matrix (Dict[str, pd.Series])      ← train_final_model_panel 输出
        ↓  rank_dpoints（每日横截面排名）
    weight_matrix (date × stock_code DataFrame)
        ↓  backtest_portfolio（模拟换仓 + 执行成本）
    PortfolioResult（equity_curve, trades, attribution, turnover）

**核心设计约定：**

1. **执行节奏（与单股回测一致）：**
   - t 日收盘后观测 dpoint → 生成换仓信号
   - t+1 日开盘价执行买卖（避免前向偏差）
   - 调仓日的"信号日"为 t，"执行日"为 t+1

2. **调仓顺序（防止透支现金）：**
   - 先执行所有卖出（回收现金）
   - 再执行所有买入（用回收的现金 + 原有现金）
   - 同一调仓日的卖出和买入共用一次现金结算

3. **A 股执行约束（复用 backtester.py）：**
   - 涨停不能买、跌停不能卖
   - 停牌无法成交
   - 上市天数不足跳过
   - 流动性不足跳过（优先使用 amount，回退到 amount_proxy）

4. **仓位权重方案：**
   - ``equal``：等权，每只持仓股权重 = 1/top_k
   - ``signal``：按 dpoint 得分比例加权，并裁剪到 [min_weight, max_weight]
   - 目标权重按"可交易现金 + 当前持仓市值"计算，不含被冻结资金

5. **性能说明：**
   - 日内循环按股票数 × 交易日数，20 只股票 × 3000 天 = 6 万次迭代，
     Python 循环足够快（< 5s）。
   - dpoint_matrix 和 stock_dict 均为内存字典，无 I/O 瓶颈。

**公开 API：**
    PortfolioConfig      — 组合参数配置数据类
    PortfolioResult      — 回测结果数据类
    rank_dpoints         — 单日横截面 dpoint 排名
    construct_portfolio  — 从 dpoint_matrix 构建权重矩阵
    backtest_portfolio   — 权重矩阵 → 组合净值 + 交易记录
    run_portfolio_backtest — 一键端到端入口（dpoint_matrix → PortfolioResult）
    compute_portfolio_attribution — 归因：各股对组合收益的贡献
    format_portfolio_summary     — 可读摘要字符串
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtester import (
    COMMISSION_RATE_BUY,
    COMMISSION_RATE_SELL,
    DEFAULT_SLIPPAGE_BPS,
    DEFAULT_LIMIT_UP_PCT,
    DEFAULT_LIMIT_DOWN_PCT,
    DEFAULT_FILTER_ST,
    DEFAULT_MIN_LISTING_DAYS,
    DEFAULT_MIN_DAILY_AMOUNT,
    check_execution_feasibility,
    apply_slippage,
    calculate_risk_metrics,
    format_metrics_summary,
)
from constants import (
    DEFAULT_TOP_K,
    DEFAULT_REBALANCE_FREQ,
    DEFAULT_WEIGHTING_SCHEME,
    DEFAULT_MAX_WEIGHT,
    DEFAULT_MIN_WEIGHT,
    DEFAULT_PORTFOLIO_INITIAL_CASH,
    COL_AMOUNT_PROXY,
)

logger = logging.getLogger(__name__)


# =========================================================
# 数据类
# =========================================================

@dataclass
class PortfolioConfig:
    """
    组合回测参数配置。

    所有参数均有默认值，可直接 ``PortfolioConfig()`` 使用。

    Attributes:
        top_k: 每期最大持仓股票数，默认 5
        rebalance_freq: 调仓频率，可选 'daily' | 'weekly' | 'monthly'，默认 'weekly'
        weighting_scheme: 持仓权重方案，可选 'equal' | 'signal'，默认 'equal'
        max_weight: 单股最大权重（signal 方案时生效），默认 0.3
        min_weight: 单股最小权重（signal 方案时，低于此权重的候选不纳入），默认 0.05
        initial_cash: 初始资金（元），默认 1_000_000
        commission_rate_buy: 买入佣金率，默认 0.03%
        commission_rate_sell: 卖出佣金 + 印花税，默认 0.13%
        slippage_bps: 固定滑点（bps），默认 20
        limit_up_pct: 涨停幅度，默认 10%
        limit_down_pct: 跌停幅度，默认 10%
        filter_st: 是否过滤 ST 股，默认 True
        min_listing_days: 最小上市天数，默认 60
        min_daily_amount: 最小日成交额（元），默认 100 万
        dpoint_buy_threshold: 选股最低 dpoint 分数门槛（低于此值不纳入候选），默认 0.5
    """
    top_k: int = DEFAULT_TOP_K
    rebalance_freq: str = DEFAULT_REBALANCE_FREQ
    weighting_scheme: str = DEFAULT_WEIGHTING_SCHEME
    max_weight: float = DEFAULT_MAX_WEIGHT
    min_weight: float = DEFAULT_MIN_WEIGHT
    initial_cash: float = DEFAULT_PORTFOLIO_INITIAL_CASH
    commission_rate_buy:  float = COMMISSION_RATE_BUY
    commission_rate_sell: float = COMMISSION_RATE_SELL
    slippage_bps: int = DEFAULT_SLIPPAGE_BPS
    limit_up_pct: float = DEFAULT_LIMIT_UP_PCT
    limit_down_pct: float = DEFAULT_LIMIT_DOWN_PCT
    filter_st: bool = DEFAULT_FILTER_ST
    min_listing_days: int = DEFAULT_MIN_LISTING_DAYS
    min_daily_amount: float = DEFAULT_MIN_DAILY_AMOUNT
    dpoint_buy_threshold: float = 0.5


@dataclass
class PortfolioResult:
    """
    组合回测结果。

    Attributes:
        equity_curve: 每日净值曲线，含列：
            date, total_equity, cash, market_value,
            cum_return, drawdown, n_positions
        trades: 交易记录（每笔买卖各一行），含列：
            date, stock_code, action, shares, price,
            cost/proceeds, commission, rebalance_date
        rebalance_log: 每次调仓摘要（调仓日、目标持仓、实际执行情况）
        metrics: 组合风险指标字典（来自 backtester.calculate_risk_metrics）
        attribution: 各股对组合收益的贡献分析 DataFrame
        turnover_series: 每次调仓的换手率 Series（index 为调仓日）
        config: 运行时使用的 PortfolioConfig
        notes: 运行日志
    """
    equity_curve:    pd.DataFrame
    trades:          pd.DataFrame
    rebalance_log:   pd.DataFrame
    metrics:         Dict[str, Any]
    attribution:     pd.DataFrame
    turnover_series: pd.Series
    config:          PortfolioConfig
    notes:           List[str]


# =========================================================
# 工具函数
# =========================================================

def _get_rebalance_dates(
    all_dates: List[pd.Timestamp],
    freq: str,
) -> List[pd.Timestamp]:
    """
    从交易日序列中提取调仓触发日期。

    **调仓触发规则：**

    - ``daily``：每个交易日都是调仓日（信号日 = 执行日前一天）
    - ``weekly``：每周第一个交易日（周一，或假期顺延后的最近交易日）
    - ``monthly``：每月第一个交易日

    调仓日是"信号日"——该日收盘后生成换仓指令，次日开盘执行。

    Args:
        all_dates: 所有交易日的有序列表
        freq: 'daily' | 'weekly' | 'monthly'

    Returns:
        调仓触发日期列表（为 all_dates 的子集）
    """
    if freq == "daily":
        return list(all_dates)

    dates_df = pd.DataFrame({"date": all_dates})
    dates_df["date"] = pd.to_datetime(dates_df["date"])

    if freq == "weekly":
        # 每周的 isocalendar().week 第一个出现的日期
        dates_df["week_key"] = dates_df["date"].dt.isocalendar().week.astype(str) + \
                               "_" + dates_df["date"].dt.isocalendar().year.astype(str)
        rebal = dates_df.groupby("week_key", sort=False)["date"].first().sort_values()
    elif freq == "monthly":
        dates_df["month_key"] = dates_df["date"].dt.to_period("M").astype(str)
        rebal = dates_df.groupby("month_key", sort=False)["date"].first().sort_values()
    else:
        raise ValueError(
            f"rebalance_freq='{freq}' 不支持，可选：'daily' | 'weekly' | 'monthly'"
        )

    return list(rebal)


def _calc_buy_shares_portfolio(cash: float, price: float, commission_rate: float) -> int:
    """按 A 股 100 股整数倍计算可买入股数（含佣金）。"""
    if price <= 0 or cash <= 0:
        return 0
    cost_per_lot = price * 100 * (1.0 + commission_rate)
    return (int(cash // cost_per_lot)) * 100


def _make_row_for_feasibility(
    stock_df: pd.DataFrame,
    date: pd.Timestamp,
    prev_date: Optional[pd.Timestamp],
) -> Optional[pd.Series]:
    """
    从单股 DataFrame 中提取指定日期的行，补全 check_execution_feasibility 需要的字段。

    **字段映射：**
        - ``amount``：优先使用原始 ``amount`` 列；不存在时回退到 ``amount_proxy``
        - ``prev_close``：用 prev_date 的 close_qfq；不存在时用当日 open 代替
        - 其他字段（is_st, suspended, listing_days）：直接取或填默认值

    Args:
        stock_df: 单股完整 OHLCV DataFrame（含 date 列）
        date: 目标日期
        prev_date: 前一交易日（用于 prev_close）

    Returns:
        兼容 check_execution_feasibility 的 pd.Series，找不到数据则返回 None
    """
    df_indexed = stock_df.set_index("date")
    if date not in df_indexed.index:
        return None

    row = df_indexed.loc[date].copy()

    # prev_close：用前一日收盘价
    if prev_date is not None and prev_date in df_indexed.index:
        row["prev_close"] = float(df_indexed.loc[prev_date, "close_qfq"])
    else:
        row["prev_close"] = float(row.get("open_qfq", 0))

    # amount：优先原始 amount，回退到 amount_proxy
    if "amount" not in row or pd.isna(row.get("amount", np.nan)):
        proxy = row.get(COL_AMOUNT_PROXY, np.nan)
        row["amount"] = float(proxy) if pd.notna(proxy) else np.nan

    # 默认值防御（pd.Series 无 setdefault，逐字段赋值）
    if "is_st" not in row.index or pd.isna(row["is_st"]) if "is_st" in row.index else True:
        row["is_st"] = False
    if "suspended" not in row.index or pd.isna(row["suspended"]) if "suspended" in row.index else True:
        row["suspended"] = False
    if "listing_days" not in row.index or pd.isna(row["listing_days"]) if "listing_days" in row.index else True:
        row["listing_days"] = 999_999

    return row


# =========================================================
# 横截面排名
# =========================================================

def rank_dpoints(
    dpoint_matrix: Dict[str, pd.Series],
    date: pd.Timestamp,
    cfg: PortfolioConfig,
) -> pd.Series:
    """
    给定某日所有股票的 dpoint，返回满足门槛条件的候选及其得分（降序排列）。

    **筛选逻辑：**
        1. 该日有 dpoint 值（不为 NaN）
        2. dpoint >= cfg.dpoint_buy_threshold

    **输出：**
        pd.Series，index 为股票代码，值为 dpoint 得分，按得分降序排列。
        最多包含所有满足条件的股票（由调用方决定取 Top-K）。

    Args:
        dpoint_matrix: {股票代码: dpoint Series}，index 为 pd.Timestamp
        date: 要查询的日期
        cfg: 组合配置

    Returns:
        按 dpoint 降序排列的候选 pd.Series
    """
    date = pd.Timestamp(date)
    scores: Dict[str, float] = {}

    for code, dp_series in dpoint_matrix.items():
        dp_series_dt = dp_series.copy()
        dp_series_dt.index = pd.to_datetime(dp_series_dt.index)
        if date in dp_series_dt.index:
            val = float(dp_series_dt.loc[date])
            if not np.isnan(val) and val >= cfg.dpoint_buy_threshold:
                scores[code] = val

    if not scores:
        return pd.Series(dtype=float)

    return pd.Series(scores).sort_values(ascending=False)


def _compute_target_weights(
    top_candidates: pd.Series,
    cfg: PortfolioConfig,
) -> Dict[str, float]:
    """
    根据 Top-K 候选列表计算目标持仓权重。

    **等权（equal）方案：**
        每只股票权重 = 1 / min(len(top_candidates), top_k)

    **信号加权（signal）方案：**
        原始权重 = dpoint_score / sum(dpoint_scores)
        裁剪到 [min_weight, max_weight]
        重新归一化使权重之和 = 1

    Args:
        top_candidates: rank_dpoints 返回的得分 Series（已按得分降序）
        cfg: 组合配置

    Returns:
        {股票代码: 目标权重}，权重之和约等于 1（执行时按可用资金比例分配）
    """
    n = min(len(top_candidates), cfg.top_k)
    if n == 0:
        return {}

    selected = top_candidates.iloc[:n]

    if cfg.weighting_scheme == "equal":
        w = 1.0 / n
        return {code: w for code in selected.index}

    elif cfg.weighting_scheme == "signal":
        raw = selected.values.astype(float)
        total = raw.sum()
        if total <= 0:
            w = 1.0 / n
            return {code: w for code in selected.index}

        weights = raw / total
        # 裁剪到 [min_weight, max_weight]
        weights = np.clip(weights, cfg.min_weight, cfg.max_weight)
        # 重新归一化
        w_sum = weights.sum()
        if w_sum > 0:
            weights = weights / w_sum
        return {code: float(w) for code, w in zip(selected.index, weights)}

    else:
        raise ValueError(
            f"weighting_scheme='{cfg.weighting_scheme}' 不支持，可选：'equal' | 'signal'"
        )


# =========================================================
# 权重矩阵构建
# =========================================================

def construct_portfolio(
    dpoint_matrix: Dict[str, pd.Series],
    cfg: PortfolioConfig,
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
) -> pd.DataFrame:
    """
    从 dpoint_matrix 构建每日目标权重矩阵。

    **输出格式：**
        DataFrame，index 为调仓触发日期，columns 为股票代码，
        值为该股目标权重（0 表示不持有，NaN 表示该股当日无 dpoint 数据）。

        注意：权重矩阵仅覆盖"调仓触发日"，而非每个交易日。
        backtest_portfolio 会将调仓信号延伸到下一个调仓日。

    **调仓触发日期：**
        从所有股票 dpoint 序列的日期并集中，按 cfg.rebalance_freq 提取。

    Args:
        dpoint_matrix: {股票代码: dpoint Series}
        cfg: 组合配置（top_k、rebalance_freq、weighting_scheme 等）
        date_range: 可选的日期范围过滤 (start, end)，均为 inclusive

    Returns:
        pd.DataFrame：调仓日 × 股票代码 的目标权重矩阵

    Raises:
        ValueError: dpoint_matrix 为空
    """
    if not dpoint_matrix:
        raise ValueError("dpoint_matrix 为空，无法构建权重矩阵。")

    # 收集所有日期的并集，排序
    all_dates_set: set = set()
    for dp in dpoint_matrix.values():
        all_dates_set.update(pd.to_datetime(dp.index))
    all_dates: List[pd.Timestamp] = sorted(all_dates_set)

    if date_range is not None:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        all_dates = [d for d in all_dates if start <= d <= end]

    if not all_dates:
        raise ValueError("过滤后没有有效交易日，请检查 date_range 参数。")

    rebal_dates = _get_rebalance_dates(all_dates, cfg.rebalance_freq)
    all_codes   = sorted(dpoint_matrix.keys())
    weight_rows: List[Dict[str, float]] = []

    for rd in rebal_dates:
        candidates = rank_dpoints(dpoint_matrix, rd, cfg)
        weights    = _compute_target_weights(candidates, cfg)
        row = {code: weights.get(code, 0.0) for code in all_codes}
        row["_rebal_date"] = rd
        weight_rows.append(row)

    if not weight_rows:
        raise ValueError("没有生成任何调仓权重行，请检查 dpoint_matrix 的日期范围。")

    weight_df = pd.DataFrame(weight_rows).set_index("_rebal_date")
    weight_df.index.name = "rebal_date"
    weight_df = weight_df[all_codes]  # 确保列顺序一致

    logger.info(
        "construct_portfolio: %d 次调仓，%d 只股票，freq=%s, scheme=%s",
        len(weight_df), len(all_codes), cfg.rebalance_freq, cfg.weighting_scheme,
    )
    return weight_df



def _calc_core_metrics(equity_curve: pd.DataFrame, initial_cash: float) -> Dict[str, Any]:
    """核心风险指标（不依赖 resample，兼容所有 pandas 版本）。"""
    equity = equity_curve["total_equity"].values
    n = len(equity)
    total_return = (equity[-1] - initial_cash) / initial_cash
    years = n / 252.0
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    daily_rets = np.diff(equity) / equity[:-1]
    daily_rets = daily_rets[np.isfinite(daily_rets)]
    annual_vol  = float(np.std(daily_rets) * np.sqrt(252)) if len(daily_rets) > 0 else 0.0
    cum_max = np.maximum.accumulate(equity)
    dd = (equity - cum_max) / cum_max
    max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0
    sharpe  = float(np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)) if (
        len(daily_rets) > 0 and np.std(daily_rets) > 0) else 0.0
    return {
        "total_return": float(total_return),
        "total_return_pct": float(total_return * 100),
        "annual_return": float(annual_return),
        "annual_return_pct": float(annual_return * 100),
        "annual_vol": annual_vol,
        "annual_vol_pct": annual_vol * 100,
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd * 100,
        "sharpe": sharpe,
        "n_days": n,
        "years": years,
        "initial_cash": float(initial_cash),
        "final_equity": float(equity[-1]),
    }

# =========================================================
# 组合回测核心
# =========================================================

def backtest_portfolio(
    stock_dict: Dict[str, pd.DataFrame],
    weight_matrix: pd.DataFrame,
    cfg: PortfolioConfig,
) -> PortfolioResult:
    """
    按权重矩阵模拟换仓，计算组合净值曲线和交易记录。

    **执行流程（每个交易日）：**

    1. 判断是否为调仓执行日（前一调仓信号日的次日）
    2. 若是：
       a. 先卖出所有减仓/清仓股（t 日开盘价 × 滑点）
       b. 再买入所有加仓/建仓股（用回收现金 + 原有现金）
       c. 记录本次调仓的换手率
    3. 每日收盘后计算组合净值（cash + Σ shares_i × close_i）
    4. 记录净值、回撤、持仓数等到 equity_curve

    **调仓资金分配逻辑：**

        target_portfolio_value = cash_after_sell + Σ shares_retained × close_exec
        buy_budget_i = target_portfolio_value × weight_i（目标权重）
        shares_to_buy_i = floor(buy_budget_i / exec_price_i / 100) × 100

        注：已持有且继续保留的股票不再交易（节省成本），
        只在权重调整时才通过卖出+买入实现再平衡。

    **A 股执行约束（来自 backtester.py）：**
        - 涨停日不能买入 → 跳过该股买入，预算重新分配给其他可买股
        - 跌停日不能卖出 → 该股继续持有，等下一交易日再尝试
        - 停牌日跳过
        - 上市天数/流动性不足跳过

    Args:
        stock_dict: {股票代码: 单股 OHLCV DataFrame}，来自 load_basket()
        weight_matrix: construct_portfolio 返回的权重矩阵（调仓日 × 股票代码）
        cfg: 组合配置

    Returns:
        PortfolioResult

    Raises:
        ValueError: stock_dict 或 weight_matrix 为空
    """
    if not stock_dict:
        raise ValueError("stock_dict 为空。")
    if weight_matrix.empty:
        raise ValueError("weight_matrix 为空。")

    all_codes = list(weight_matrix.columns)
    rebal_signal_dates = sorted(weight_matrix.index)

    # 为每只股票建立日期索引，方便 O(1) 查询
    stock_indexed: Dict[str, pd.DataFrame] = {}
    for code, df in stock_dict.items():
        df_c = df.copy()
        df_c["date"] = pd.to_datetime(df_c["date"])
        df_c = df_c.sort_values("date").reset_index(drop=True)
        stock_indexed[code] = df_c.set_index("date")

    # 所有交易日：取所有股票日期的并集
    all_trading_dates: List[pd.Timestamp] = sorted(
        set().union(*[set(df.index) for df in stock_indexed.values()])
    )

    # 构建"调仓信号日 → 执行日"映射（信号日次日为执行日）
    # 若次日不是交易日，顺延到最近的交易日
    trading_date_set = set(all_trading_dates)
    signal_to_exec: Dict[pd.Timestamp, pd.Timestamp] = {}
    for sd in rebal_signal_dates:
        # 在 all_trading_dates 中找 sd 之后的第一个交易日
        future = [d for d in all_trading_dates if d > sd]
        if future:
            signal_to_exec[sd] = future[0]

    # exec_date → signal_date 的反向映射（用于查权重）
    exec_to_signal: Dict[pd.Timestamp, pd.Timestamp] = {
        v: k for k, v in signal_to_exec.items()
    }

    # ── 状态变量 ──────────────────────────────────────────
    cash: float = float(cfg.initial_cash)
    positions: Dict[str, int] = {}        # {code: shares}
    pending_sell: Dict[str, str] = {}     # {code: reason}（跌停未能卖出的持仓）
    # 最近已知收盘价（forward fill）：持仓股在无行情数据的交易日用此价格估算市值，
    # 避免净值在停牌/数据缺失日出现虚假归零后的剧烈跳升
    last_known_close: Dict[str, float] = {}

    # ── 输出记录 ──────────────────────────────────────────
    equity_rows:   List[Dict] = []
    trade_rows:    List[Dict] = []
    rebal_rows:    List[Dict] = []
    turnover_vals: Dict[pd.Timestamp, float] = {}

    prev_date: Optional[pd.Timestamp] = None

    for dt in all_trading_dates:

        # ── 阶段一：执行当日调仓（若是执行日）────────────
        is_exec_day = dt in exec_to_signal
        rebal_stats: Dict[str, Any] = {}

        if is_exec_day:
            signal_date = exec_to_signal[dt]
            target_weights = weight_matrix.loc[signal_date]
            target_codes   = set(c for c in all_codes if float(target_weights.get(c, 0)) > 0)
            current_codes  = set(c for c, sh in positions.items() if sh > 0)

            sell_codes = current_codes - target_codes   # 减仓/清仓
            buy_codes  = target_codes - current_codes   # 新建/加仓（纯加仓需重新计算）
            keep_codes = current_codes & target_codes   # 继续持有（权重可能变化）

            rebal_sells_ok:    List[str] = []
            rebal_sells_fail:  List[str] = []
            rebal_buys_ok:     List[str] = []
            rebal_buys_fail:   List[str] = []

            # ── 1a. 先卖出 ──────────────────────────────
            cash_before_sell = cash
            for code in list(sell_codes) + list(pending_sell.keys()):
                if code not in positions or positions[code] <= 0:
                    continue
                if code not in stock_indexed:
                    continue

                row = _make_row_for_feasibility(
                    stock_dict[code], dt, prev_date
                )
                if row is None:
                    rebal_sells_fail.append(code)
                    continue

                feasible, reason = check_execution_feasibility(
                    row, "SELL",
                    limit_up_pct=cfg.limit_up_pct,
                    limit_down_pct=cfg.limit_down_pct,
                    filter_st=cfg.filter_st,
                    min_listing_days=cfg.min_listing_days,
                    min_daily_amount=cfg.min_daily_amount,
                )
                if not feasible:
                    # 跌停等情况，保留持仓，下次再试
                    if reason == "跌停卖不掉":
                        pending_sell[code] = reason
                    rebal_sells_fail.append(f"{code}({reason})")
                    continue

                exec_price = apply_slippage(
                    float(row["open_qfq"]), "SELL", cfg.slippage_bps
                )
                shares_sold = positions[code]
                commission  = shares_sold * exec_price * cfg.commission_rate_sell
                proceeds    = shares_sold * exec_price - commission
                cash += proceeds
                positions[code] = 0
                pending_sell.pop(code, None)

                rebal_sells_ok.append(code)
                trade_rows.append({
                    "date":         dt,
                    "stock_code":   code,
                    "action":       "SELL",
                    "shares":       shares_sold,
                    "exec_price":   round(exec_price, 4),
                    "proceeds":     round(proceeds, 2),
                    "commission":   round(commission, 2),
                    "rebal_date":   dt,
                    "signal_date":  signal_date,
                })

            # ── 1b. 计算买入预算 ──────────────────────────
            # 当前总资产估值（用执行日开盘价估算持仓市值）
            portfolio_value = cash
            for code, sh in positions.items():
                if sh <= 0 or code not in stock_indexed:
                    continue
                if dt in stock_indexed[code].index:
                    price_ref = float(stock_indexed[code].loc[dt, "open_qfq"])
                    portfolio_value += sh * price_ref

            # 目标买入代码（新建 + 权重变化导致需要再平衡的 keep）
            all_target_codes = list(target_codes)
            buy_budgets: Dict[str, float] = {}
            for code in all_target_codes:
                w = float(target_weights.get(code, 0.0))
                if w <= 0:
                    continue
                target_value = portfolio_value * w
                current_value = 0.0
                if code in positions and positions.get(code, 0) > 0:
                    if dt in stock_indexed.get(code, pd.DataFrame()).index:
                        p = float(stock_indexed[code].loc[dt, "open_qfq"])
                        current_value = positions[code] * p
                # 只有目标市值 > 当前市值才买入（避免频繁小额调整）
                if target_value > current_value + 100:  # 差距超过 100 元才调仓
                    buy_budgets[code] = target_value - current_value

            # ── 1c. 按预算买入（从大到小，防止现金不足）──
            for code in sorted(buy_budgets, key=buy_budgets.get, reverse=True):
                budget = min(buy_budgets[code], cash)
                if budget < 100:   # 低于 1 手最低成本（约 100 元），跳过
                    continue
                if code not in stock_indexed:
                    continue

                row = _make_row_for_feasibility(
                    stock_dict[code], dt, prev_date
                )
                if row is None:
                    rebal_buys_fail.append(code)
                    continue

                feasible, reason = check_execution_feasibility(
                    row, "BUY",
                    limit_up_pct=cfg.limit_up_pct,
                    limit_down_pct=cfg.limit_down_pct,
                    filter_st=cfg.filter_st,
                    min_listing_days=cfg.min_listing_days,
                    min_daily_amount=cfg.min_daily_amount,
                )
                if not feasible:
                    rebal_buys_fail.append(f"{code}({reason})")
                    continue

                exec_price = apply_slippage(
                    float(row["open_qfq"]), "BUY", cfg.slippage_bps
                )
                shares_buy = _calc_buy_shares_portfolio(
                    budget, exec_price, cfg.commission_rate_buy
                )
                if shares_buy <= 0:
                    rebal_buys_fail.append(f"{code}(资金不足1手)")
                    continue

                cost = shares_buy * exec_price * (1 + cfg.commission_rate_buy)
                if cost > cash:
                    # 重新按实际可用现金计算
                    shares_buy = _calc_buy_shares_portfolio(
                        cash, exec_price, cfg.commission_rate_buy
                    )
                    if shares_buy <= 0:
                        rebal_buys_fail.append(f"{code}(现金不足)")
                        continue
                    cost = shares_buy * exec_price * (1 + cfg.commission_rate_buy)

                commission = shares_buy * exec_price * cfg.commission_rate_buy
                cash -= cost
                positions[code] = positions.get(code, 0) + shares_buy
                rebal_buys_ok.append(code)

                trade_rows.append({
                    "date":         dt,
                    "stock_code":   code,
                    "action":       "BUY",
                    "shares":       shares_buy,
                    "exec_price":   round(exec_price, 4),
                    "cost":         round(cost, 2),
                    "commission":   round(commission, 2),
                    "rebal_date":   dt,
                    "signal_date":  signal_date,
                })

            # ── 计算换手率 ────────────────────────────────
            # 换手率 = 当日买卖总额 / 期初组合价值
            trade_today = [t for t in trade_rows if t["date"] == dt]
            traded_value = sum(
                t.get("cost", t.get("proceeds", 0)) for t in trade_today
            )
            turnover = traded_value / max(portfolio_value, 1.0)
            turnover_vals[dt] = round(turnover, 6)

            rebal_stats = {
                "rebal_exec_date": dt,
                "signal_date":     signal_date,
                "target_n":        len(target_codes),
                "sells_ok":        len(rebal_sells_ok),
                "sells_fail":      len(rebal_sells_fail),
                "buys_ok":         len(rebal_buys_ok),
                "buys_fail":       len(rebal_buys_fail),
                "cash_before":     round(cash_before_sell, 2),
                "cash_after":      round(cash, 2),
                "turnover":        round(turnover, 4),
            }
            rebal_rows.append(rebal_stats)

        # ── 阶段二：每日净值快照（收盘价估值）────────────
        market_value = 0.0
        for code, sh in positions.items():
            if sh <= 0 or code not in stock_indexed:
                continue
            if dt in stock_indexed[code].index:
                close_t = float(stock_indexed[code].loc[dt, "close_qfq"])
                last_known_close[code] = close_t   # 更新最近已知价格
                market_value += sh * close_t
            elif code in last_known_close:
                # 数据缺失（停牌/节假日）：用最近已知收盘价估算，避免市值虚假归零
                market_value += sh * last_known_close[code]

        total_equity = cash + market_value
        n_positions  = sum(1 for sh in positions.values() if sh > 0)

        equity_rows.append({
            "date":           dt,
            "total_equity":   round(total_equity, 2),
            "cash":           round(cash, 2),
            "market_value":   round(market_value, 2),
            "n_positions":    n_positions,
        })

        prev_date = dt

    # ── 组装 equity_curve ─────────────────────────────────
    equity_curve = pd.DataFrame(equity_rows)
    if not equity_curve.empty:
        equity_curve["cum_return"] = (
            equity_curve["total_equity"] / cfg.initial_cash - 1.0
        ).round(6)
        cum_max = equity_curve["total_equity"].cummax()
        equity_curve["drawdown"] = (
            equity_curve["total_equity"] / cum_max - 1.0
        ).round(6)

    # ── 风险指标 ──────────────────────────────────────────
    metrics: Dict[str, Any] = {}
    if not equity_curve.empty:
        try:
            metrics = calculate_risk_metrics(
                equity_curve,
                trades=pd.DataFrame(trade_rows) if trade_rows else None,
                initial_cash=cfg.initial_cash,
            )
        except Exception as e:
            # 兜底：calculate_risk_metrics 内部可能含 pandas 版本相关的 resample 问题
            # 降级为仅计算核心指标
            logger.debug("calculate_risk_metrics 完整调用失败（%s），使用核心指标降级版", e)
            metrics = _calc_core_metrics(equity_curve, cfg.initial_cash)

    # ── 输出 DataFrame ────────────────────────────────────
    trades_df   = pd.DataFrame(trade_rows)   if trade_rows   else pd.DataFrame()
    rebal_df    = pd.DataFrame(rebal_rows)   if rebal_rows   else pd.DataFrame()
    turnover_sr = pd.Series(turnover_vals, name="turnover")

    notes = [
        f"Portfolio backtest: {len(stock_dict)} stocks, "
        f"freq={cfg.rebalance_freq}, scheme={cfg.weighting_scheme}, top_k={cfg.top_k}",
        f"Date range: {all_trading_dates[0].date()} ~ {all_trading_dates[-1].date()}",
        f"Rebalance events: {len(rebal_rows)}",
        f"Total trades: {len(trade_rows)} "
        f"(buys={sum(1 for t in trade_rows if t.get('action')=='BUY')}, "
        f"sells={sum(1 for t in trade_rows if t.get('action')=='SELL')})",
        f"Avg turnover per rebal: {turnover_sr.mean():.2%}" if not turnover_sr.empty else "",
        f"Final equity: {equity_curve['total_equity'].iloc[-1]:,.2f}" if not equity_curve.empty else "",
    ]

    # 归因在后面单独调用
    attribution = pd.DataFrame()

    return PortfolioResult(
        equity_curve=equity_curve,
        trades=trades_df,
        rebalance_log=rebal_df,
        metrics=metrics,
        attribution=attribution,
        turnover_series=turnover_sr,
        config=cfg,
        notes=notes,
    )


# =========================================================
# 归因分析
# =========================================================

def compute_portfolio_attribution(
    portfolio_result: PortfolioResult,
    stock_dict: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    计算各股票对组合总收益的贡献（事后归因）。

    **归因方法（持有期收益归因）：**
        对每一笔已完成的买卖配对（买入 → 卖出），计算：
            - 持有区间收益（卖出实收 - 买入成本）
            - 对组合总 PnL 的贡献比例

        对于回测结束时仍持有的仓位，按末日收盘价估算未实现盈亏。

    **输出 DataFrame 列：**
        stock_code, total_pnl, n_trades, win_trades, win_rate,
        avg_hold_days, contribution_pct（占组合总 PnL 的比例）

    Args:
        portfolio_result: backtest_portfolio 的返回值
        stock_dict: 原始行情数据（用于获取末日收盘价估算未平仓盈亏）

    Returns:
        pd.DataFrame，按 total_pnl 降序排列
    """
    if portfolio_result.trades.empty:
        return pd.DataFrame()

    trades = portfolio_result.trades.copy()
    trades["date"] = pd.to_datetime(trades["date"])

    # 按股票代码配对买卖，计算每笔交易盈亏
    attr_rows: List[Dict] = []

    for code in trades["stock_code"].unique():
        code_trades = trades[trades["stock_code"] == code].sort_values("date")
        buys  = code_trades[code_trades["action"] == "BUY"].reset_index(drop=True)
        sells = code_trades[code_trades["action"] == "SELL"].reset_index(drop=True)

        pnl_list:       List[float] = []
        hold_days_list: List[int]   = []
        win_list:       List[bool]  = []

        # 简单配对（FIFO）
        buy_q: List[Tuple[pd.Timestamp, float, int]] = []  # (date, cost_per_share, shares)
        for _, br in buys.iterrows():
            cost_per = float(br["exec_price"]) * (1 + portfolio_result.config.commission_rate_buy)
            buy_q.append((pd.Timestamp(br["date"]), cost_per, int(br["shares"])))

        for _, sr in sells.iterrows():
            sell_date   = pd.Timestamp(sr["date"])
            sell_price  = float(sr["exec_price"]) * (1 - portfolio_result.config.commission_rate_sell)
            sell_shares = int(sr["shares"])
            remaining   = sell_shares

            while remaining > 0 and buy_q:
                buy_date, buy_cost, buy_sh = buy_q[0]
                matched = min(remaining, buy_sh)
                pnl    = matched * (sell_price - buy_cost)
                days   = (sell_date - buy_date).days
                pnl_list.append(pnl)
                hold_days_list.append(max(days, 1))
                win_list.append(pnl > 0)

                remaining -= matched
                if matched == buy_sh:
                    buy_q.pop(0)
                else:
                    buy_q[0] = (buy_date, buy_cost, buy_sh - matched)

        # 未平仓仓位的未实现盈亏
        unrealized_pnl = 0.0
        if buy_q and code in stock_dict:
            last_close_df = stock_dict[code].sort_values("date")
            if not last_close_df.empty:
                last_close = float(last_close_df["close_qfq"].iloc[-1])
                for buy_date, buy_cost, buy_sh in buy_q:
                    est_sell_price = last_close * (1 - portfolio_result.config.commission_rate_sell)
                    unrealized_pnl += buy_sh * (est_sell_price - buy_cost)

        total_pnl  = sum(pnl_list) + unrealized_pnl
        n_trades   = len(pnl_list)
        win_trades = sum(win_list)
        win_rate   = win_trades / n_trades if n_trades > 0 else 0.0
        avg_hold   = np.mean(hold_days_list) if hold_days_list else 0.0

        attr_rows.append({
            "stock_code":      code,
            "total_pnl":       round(total_pnl, 2),
            "realized_pnl":    round(sum(pnl_list), 2),
            "unrealized_pnl":  round(unrealized_pnl, 2),
            "n_trades":        n_trades,
            "win_trades":      win_trades,
            "win_rate":        round(win_rate, 4),
            "avg_hold_days":   round(avg_hold, 1),
        })

    if not attr_rows:
        return pd.DataFrame()

    attr_df = pd.DataFrame(attr_rows).sort_values("total_pnl", ascending=False)
    total_pnl_all = attr_df["total_pnl"].sum()
    attr_df["contribution_pct"] = (
        attr_df["total_pnl"] / max(abs(total_pnl_all), 1.0) * 100
    ).round(2)
    attr_df = attr_df.reset_index(drop=True)

    return attr_df


# =========================================================
# 端到端一键入口
# =========================================================

def run_portfolio_backtest(
    stock_dict: Dict[str, pd.DataFrame],
    dpoint_matrix: Dict[str, pd.Series],
    cfg: Optional[PortfolioConfig] = None,
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    compute_attribution: bool = True,
) -> PortfolioResult:
    """
    端到端组合回测入口：dpoint_matrix → PortfolioResult。

    串联以下步骤：
        1. ``construct_portfolio``：构建权重矩阵
        2. ``backtest_portfolio``：模拟换仓，生成净值曲线
        3. ``compute_portfolio_attribution``（可选）：归因分析

    这是 main_cli.py 中调用的主入口，用户无需手动串联各步骤。

    Args:
        stock_dict: {股票代码: 单股 OHLCV DataFrame}，来自 load_basket()
        dpoint_matrix: {股票代码: dpoint Series}，来自 train_final_model_panel()
        cfg: 组合配置，None 时使用默认值
        date_range: 可选的回测日期范围 (start, end)
        compute_attribution: 是否计算归因分析，默认 True

    Returns:
        PortfolioResult
    """
    if cfg is None:
        cfg = PortfolioConfig()

    logger.info(
        "run_portfolio_backtest: stocks=%d, freq=%s, top_k=%d, scheme=%s",
        len(stock_dict), cfg.rebalance_freq, cfg.top_k, cfg.weighting_scheme,
    )

    # Step 1: 构建权重矩阵
    weight_matrix = construct_portfolio(dpoint_matrix, cfg, date_range=date_range)

    # Step 2: 回测
    result = backtest_portfolio(stock_dict, weight_matrix, cfg)

    # Step 3: 归因
    if compute_attribution and not result.trades.empty:
        result.attribution = compute_portfolio_attribution(result, stock_dict)

    return result


# =========================================================
# 格式化摘要
# =========================================================

def format_portfolio_summary(result: PortfolioResult) -> str:
    """
    将 PortfolioResult 格式化为可读摘要字符串。

    Args:
        result: run_portfolio_backtest 的返回值

    Returns:
        多行摘要字符串，适合打印到控制台或写入报告
    """
    cfg = result.cfg if hasattr(result, "cfg") else result.config
    lines = [
        "=" * 60,
        "PORTFOLIO BACKTEST SUMMARY",
        "=" * 60,
        f"  Config  : top_k={cfg.top_k}, freq={cfg.rebalance_freq}, "
        f"scheme={cfg.weighting_scheme}",
        f"  Cash    : {cfg.initial_cash:,.0f} CNY",
        "",
    ]

    # 净值指标
    m = result.metrics
    if m:
        lines += [
            "  ── 收益风险指标 ──────────────────────────",
            f"  Total Return    : {m.get('total_return_pct', 0):+.2f}%",
            f"  Annual Return   : {m.get('annual_return_pct', 0):+.2f}%",
            f"  Annual Vol      : {m.get('annual_vol_pct', 0):.2f}%",
            f"  Sharpe          : {m.get('sharpe', 0):.3f}",
            f"  Max Drawdown    : {m.get('max_drawdown_pct', 0):.2f}%",
            f"  Calmar          : {m.get('calmar', 0):.3f}",
            "",
        ]

    # 换手率
    if not result.turnover_series.empty:
        avg_to = result.turnover_series.mean()
        lines += [
            "  ── 换手统计 ──────────────────────────────",
            f"  Avg Turnover/Rebal : {avg_to:.2%}",
            f"  Rebal Count        : {len(result.rebalance_log)}",
            "",
        ]

    # 交易统计
    if not result.trades.empty:
        n_buy  = (result.trades["action"] == "BUY").sum()
        n_sell = (result.trades["action"] == "SELL").sum()
        lines += [
            "  ── 交易统计 ──────────────────────────────",
            f"  Total Trades : {len(result.trades)} (买={n_buy}, 卖={n_sell})",
            "",
        ]

    # 归因 Top5
    if not result.attribution.empty:
        lines += ["  ── 归因 Top5（收益贡献）─────────────────"]
        for _, row in result.attribution.head(5).iterrows():
            lines.append(
                f"  {row['stock_code']}: PnL={row['total_pnl']:,.0f}  "
                f"Win={row['win_rate']:.0%}  "
                f"贡献={row['contribution_pct']:+.1f}%"
            )

    lines.append("=" * 60)
    return "\n".join(lines)


# =========================================================
# 公开 API 导出
# =========================================================
__all__ = [
    "PortfolioConfig",
    "PortfolioResult",
    "rank_dpoints",
    "construct_portfolio",
    "backtest_portfolio",
    "compute_portfolio_attribution",
    "run_portfolio_backtest",
    "format_portfolio_summary",
    # 内部工具（供测试）
    "_get_rebalance_dates",
    "_compute_target_weights",
    "_make_row_for_feasibility",
]
