# basket_loader.py
"""
篮子文件夹加载器：从目录中发现 CSV 文件并构建面板。
来自 DpointTrader_deeplearning_Ver1.0/basket_loader.py。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from dpoint.core.constants import COL_TICKER
from dpoint.data.csv_loader import extract_ticker_from_filename, load_single_csv
from dpoint.data.panel_builder import align_calendar, build_panel

logger = logging.getLogger(__name__)


@dataclass
class BasketReport:
    """篮子数据质量报告。"""
    basket_name: str = ""
    basket_path: str = ""
    n_files_found: int = 0
    n_files_loaded: int = 0
    n_files_failed: int = 0
    total_rows: int = 0
    failed_files: List[str] = field(default_factory=list)
    per_stock_reports: Dict[str, any] = field(default_factory=dict)


def discover_csv_files(
    basket_dir: str | Path,
    pattern: str = "*.csv",
) -> List[Path]:
    """发现篮子目录中的 CSV 文件。"""
    basket_dir = Path(basket_dir)
    if not basket_dir.is_dir():
        raise FileNotFoundError(f"Basket directory not found: {basket_dir}")
    files = sorted(basket_dir.glob(pattern))
    logger.info("Discovered %d CSV files in %s", len(files), basket_dir)
    return files


def load_basket_folder(
    basket_dir: str | Path,
    pattern: str = "*.csv",
    column_map: Optional[Dict[str, str]] = None,
    calendar_align: str = "none",  # none / inner / outer / majority
) -> Tuple[pd.DataFrame, BasketReport]:
    """
    加载篮子文件夹中的所有 CSV 文件并构建面板。

    Args:
        basket_dir: 篮子目录路径
        pattern: 文件匹配模式
        column_map: 列名映射
        calendar_align: 日历对齐方法

    Returns:
        (面板 DataFrame, 篮子报告)
    """
    basket_dir = Path(basket_dir)
    basket_name = basket_dir.name
    report = BasketReport(basket_name=basket_name, basket_path=str(basket_dir))

    csv_files = discover_csv_files(basket_dir, pattern)
    report.n_files_found = len(csv_files)

    stock_frames = []
    for f in csv_files:
        ticker = extract_ticker_from_filename(f)
        try:
            df, stock_report = load_single_csv(f, ticker=ticker, column_map=column_map)
            # 添加 ticker 列
            df[COL_TICKER] = ticker
            stock_frames.append(df)
            report.n_files_loaded += 1
            report.total_rows += len(df)
            report.per_stock_reports[ticker] = stock_report
        except Exception as e:
            logger.warning("Failed to load %s: %s", f.name, e)
            report.n_files_failed += 1
            report.failed_files.append(f.name)

    if not stock_frames:
        raise ValueError(f"No valid CSV files loaded from {basket_dir}")

    # 构建面板
    panel = build_panel(stock_frames, basket_name=basket_name)

    # 日历对齐
    if calendar_align != "none":
        panel = align_calendar(panel, method=calendar_align)

    return panel, report
