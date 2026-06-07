# panel_builder.py
"""
面板数据构建器：将多只股票合并为 date × ticker 面板。
来自 DpointTrader_deeplearning_Ver1.0/panel_builder.py。
"""
from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd

from dpoint.core.constants import COL_DATE, COL_TICKER, REQUIRED_COLS_PANEL

logger = logging.getLogger(__name__)


def add_ticker_column(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """为 DataFrame 添加 ticker 列。"""
    df = df.copy()
    df[COL_TICKER] = ticker
    return df


def build_panel(
    stock_frames: List[pd.DataFrame],
    basket_name: str = "",
) -> pd.DataFrame:
    """
    将多只股票的 DataFrame 合并为面板数据。

    Args:
        stock_frames: 已包含 ticker 列的 DataFrame 列表
        basket_name: 篮子名称（用于日志）

    Returns:
        合并后的面板 DataFrame
    """
    if not stock_frames:
        raise ValueError("No stock frames provided")

    panel = pd.concat(stock_frames, ignore_index=True)

    # 检查必需列
    missing = [c for c in REQUIRED_COLS_PANEL if c not in panel.columns]
    if missing:
        raise ValueError(f"Panel missing required columns: {missing}")

    # 排序
    panel = panel.sort_values([COL_DATE, COL_TICKER]).reset_index(drop=True)

    n_tickers = panel[COL_TICKER].nunique()
    n_dates = panel[COL_DATE].nunique()
    logger.info(
        "Built panel '%s': %d rows, %d tickers, %d dates",
        basket_name or "unnamed", len(panel), n_tickers, n_dates,
    )

    return panel


def align_calendar(
    panel_df: pd.DataFrame,
    date_col: str = COL_DATE,
    ticker_col: str = COL_TICKER,
    method: str = "inner",
) -> pd.DataFrame:
    """
    日历对齐：确保所有股票有相同的交易日。

    Args:
        method: "inner" (交集) / "outer" (并集) / "majority" (>50%)
    """
    if method == "outer":
        return panel_df

    all_dates = set(panel_df[date_col].unique())
    ticker_dates = {}
    for ticker in panel_df[ticker_col].unique():
        ticker_dates[ticker] = set(panel_df[panel_df[ticker_col] == ticker][date_col].unique())

    if method == "inner":
        for td in ticker_dates.values():
            all_dates &= td
    elif method == "majority":
        date_counts = {}
        for td in ticker_dates.values():
            for d in td:
                date_counts[d] = date_counts.get(d, 0) + 1
        threshold = len(ticker_dates) * 0.5
        all_dates = {d for d, c in date_counts.items() if c >= threshold}

    if not all_dates:
        logger.warning("Calendar alignment resulted in no common dates")
        return panel_df.iloc[:0]

    result = panel_df[panel_df[date_col].isin(all_dates)].copy()
    logger.info("Calendar alignment (%s): %d -> %d dates", method, panel_df[date_col].nunique(), len(all_dates))
    return result.reset_index(drop=True)
