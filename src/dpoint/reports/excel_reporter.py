# excel_reporter.py
"""
Excel 报告生成器。
合并自两个项目的 reporter.py，统一输出格式。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def escape_excel_formulas(df: pd.DataFrame) -> pd.DataFrame:
    """防止 Excel 将字符串误认为公式。"""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(
                lambda v: ("'" + v) if isinstance(v, str) and v[:1] in ("=", "+", "-", "@") else v
            )
    return df


def save_excel_report(
    output_path: str | Path,
    *,
    equity_curve: Optional[pd.DataFrame] = None,
    trades: Optional[pd.DataFrame] = None,
    risk_metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
    search_log: Optional[List[Dict[str, Any]]] = None,
    ranking_metrics: Optional[Dict[str, float]] = None,
    notes: Optional[List[str]] = None,
    fold_results: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    """
    保存 Excel 报告。

    Args:
        output_path: 输出文件路径
        equity_curve: 净值曲线
        trades: 交易记录
        risk_metrics: 风险指标
        config: 运行配置
        search_log: 搜索日志
        ranking_metrics: 因子排名指标
        notes: 注释
        fold_results: 各折结果

    Returns:
        输出文件路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        # Sheet 1: Risk Metrics（最重要，放第一个）
        if risk_metrics:
            metrics_df = pd.DataFrame([risk_metrics])
            metrics_df = escape_excel_formulas(metrics_df)
            metrics_df.to_excel(writer, sheet_name="RiskMetrics", index=False)
            _auto_width(writer.sheets["RiskMetrics"], metrics_df)

        # Sheet 2: Equity Curve
        if equity_curve is not None and not equity_curve.empty:
            ec = escape_excel_formulas(equity_curve)
            ec.to_excel(writer, sheet_name="EquityCurve", index=False)
            _auto_width(writer.sheets["EquityCurve"], ec)

        # Sheet 3: Trades
        if trades is not None and not trades.empty:
            trades_esc = escape_excel_formulas(trades)
            trades_esc.to_excel(writer, sheet_name="Trades", index=False)
            _auto_width(writer.sheets["Trades"], trades_esc)

        # Sheet 4: Ranking Metrics（篮子模式）
        if ranking_metrics:
            rm_df = pd.DataFrame([ranking_metrics])
            rm_df = escape_excel_formulas(rm_df)
            rm_df.to_excel(writer, sheet_name="RankingMetrics", index=False)
            _auto_width(writer.sheets["RankingMetrics"], rm_df)

        # Sheet 5: Search Log
        if search_log:
            log_df = pd.DataFrame(search_log)
            log_df = escape_excel_formulas(log_df)
            log_df.to_excel(writer, sheet_name="SearchLog", index=False)
            _auto_width(writer.sheets["SearchLog"], log_df)

        # Sheet 6: Fold Results
        if fold_results:
            fold_df = pd.DataFrame(fold_results)
            fold_df = escape_excel_formulas(fold_df)
            fold_df.to_excel(writer, sheet_name="FoldResults", index=False)
            _auto_width(writer.sheets["FoldResults"], fold_df)

        # Sheet 7: Config
        if config:
            config_df = pd.DataFrame(
                [{"key": k, "value": str(v)} for k, v in _flatten_dict(config).items()]
            )
            config_df = escape_excel_formulas(config_df)
            config_df.to_excel(writer, sheet_name="Config", index=False)
            _auto_width(writer.sheets["Config"], config_df)

        # Sheet 8: Notes
        if notes:
            notes_df = pd.DataFrame({"note": notes})
            notes_df.to_excel(writer, sheet_name="Notes", index=False)
            _auto_width(writer.sheets["Notes"], notes_df)

    logger.info("Excel report saved: %s", output_path)
    return output_path


def _auto_width(worksheet, df: pd.DataFrame):
    """自动调整列宽。"""
    for i, col in enumerate(df.columns):
        max_len = max(df[col].astype(str).map(len).max(), len(str(col))) + 2
        worksheet.set_column(i, i, min(max_len, 50))


def _flatten_dict(d: dict, prefix: str = "") -> dict:
    """展平嵌套字典。"""
    items = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, key))
        else:
            items[key] = v
    return items
