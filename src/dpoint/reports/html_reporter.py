# html_reporter.py
"""
HTML 报告生成器（可选，依赖 plotly）。
来自 Ver1.0 的 html_reporter.py。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def save_html_report(
    output_path: str | Path,
    *,
    equity_curve: Optional[pd.DataFrame] = None,
    risk_metrics: Optional[Dict[str, float]] = None,
    trades: Optional[pd.DataFrame] = None,
    title: str = "Dpoint_Trader Report",
) -> Path:
    """
    保存 HTML 报告。需要 plotly 库。

    Args:
        output_path: 输出文件路径
        equity_curve: 净值曲线
        risk_metrics: 风险指标
        trades: 交易记录
        title: 报告标题

    Returns:
        输出文件路径
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.warning("plotly not installed, skipping HTML report. Install with: pip install plotly")
        return Path(output_path)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Equity Curve", "Drawdown"],
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
    )

    # 净值曲线
    if equity_curve is not None and not equity_curve.empty and "total_equity" in equity_curve.columns:
        dates = equity_curve["date"] if "date" in equity_curve.columns else equity_curve.index
        fig.add_trace(
            go.Scatter(x=dates, y=equity_curve["total_equity"], name="Strategy", line=dict(color="blue")),
            row=1, col=1,
        )
        # Buy & Hold 基准
        if "bnh_equity" in equity_curve.columns:
            fig.add_trace(
                go.Scatter(x=dates, y=equity_curve["bnh_equity"], name="Buy & Hold", line=dict(color="gray", dash="dash")),
                row=1, col=1,
            )

    # 回撤
    if equity_curve is not None and "drawdown" in equity_curve.columns:
        dates = equity_curve["date"] if "date" in equity_curve.columns else equity_curve.index
        fig.add_trace(
            go.Scatter(x=dates, y=equity_curve["drawdown"], name="Drawdown", fill="tozeroy", line=dict(color="red")),
            row=2, col=1,
        )

    fig.update_layout(title=title, height=800, showlegend=True)

    # 指标摘要
    html_content = f"<html><head><title>{title}</title></head><body>"
    html_content += f"<h1>{title}</h1>"

    if risk_metrics:
        html_content += "<h2>Risk Metrics</h2><table border='1'>"
        for k, v in risk_metrics.items():
            html_content += f"<tr><td>{k}</td><td>{v}</td></tr>"
        html_content += "</table>"

    html_content += fig.to_html(full_html=False, include_plotlyjs="cdn")
    html_content += "</body></html>"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info("HTML report saved: %s", output_path)
    return output_path
