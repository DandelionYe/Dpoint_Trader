"""实验结果详情页面。

展示单个实验的配置、权益曲线、交易记录、搜索日志等。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from nicegui import ui

from gui.components.layout import create_page_layout
from gui.components.charts import equity_curve_chart, drawdown_chart
from gui.state import app_state
from gui.utils import validate_output_subpath

logger = logging.getLogger(__name__)


def _load_experiment(exp_name: str) -> dict | None:
    """加载实验数据。"""
    exp_dir = validate_output_subpath(app_state.output_dir, exp_name)
    if exp_dir is None or not exp_dir.exists():
        return None

    data = {"name": exp_name, "path": str(exp_dir)}

    for filename, key in [("manifest.json", "manifest"), ("config.json", "config")]:
        fpath = exp_dir / filename
        if fpath.exists():
            try:
                data[key] = json.loads(fpath.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("读取 %s 失败: %s", filename, e)

    report_path = exp_dir / "report.xlsx"
    if report_path.exists():
        for sheet_name, key in [
            ("EquityCurve", "equity_df"),
            ("RiskMetrics", "risk_df"),
            ("Trades", "trades_df"),
            ("SearchLog", "search_df"),
        ]:
            try:
                data[key] = pd.read_excel(report_path, sheet_name=sheet_name)
            except ValueError:
                pass
            except Exception as e:
                logger.warning("读取 %s 的 %s 工作表失败: %s", report_path, sheet_name, e)

    return data


def _render_df_table(df: pd.DataFrame, empty_msg: str = "无数据") -> None:
    """将 DataFrame 渲染为 NiceGUI 表格（最多 200 行）。"""
    if df is None or df.empty:
        ui.label(empty_msg).classes("text-grey-6")
        return
    columns = [
        {"name": str(c), "label": str(c), "field": str(c), "align": "left"}
        for c in df.columns
    ]
    rows = [{str(k): str(v) for k, v in r.items()} for r in df.head(200).to_dict("records")]
    ui.table(columns=columns, rows=rows, row_key=str(df.columns[0])).classes("w-full")


@ui.page("/results/{exp_name}")
def results_page(exp_name: str):
    create_page_layout()

    data = _load_experiment(exp_name)

    with ui.column().classes("w-full p-4 gap-4"):
        with ui.row().classes("items-center gap-4"):
            ui.label(f"实验详情: {exp_name}").classes("text-h4")
            ui.button(
                "🔄 恢复搜索",
                on_click=lambda: ui.navigate.to("/resume"),
                color="orange",
                icon="play_arrow",
            ).classes("q-ml-auto")

        if data is None:
            ui.label("实验不存在或无法读取。").classes("text-negative")
            return

        # ---- 配置摘要 ----
        config = data.get("config", {})
        if config:
            with ui.expansion("实验配置", icon="settings").classes("w-full"):
                ui.code(json.dumps(config, indent=2, ensure_ascii=False), language="json").classes(
                    "w-full"
                )

        # ---- 风险指标 ----
        risk_df = data.get("risk_df")
        if risk_df is not None and not risk_df.empty:
            with ui.card().classes("w-full"):
                ui.label("风险指标").classes("text-h6")
                ui.separator()
                with ui.row().classes("gap-4 flex-wrap"):
                    row = risk_df.iloc[0]
                    for col_name in risk_df.columns:
                        with ui.card().classes("flex-1 min-w-[120px]"):
                            ui.label(str(col_name)).classes("text-caption text-grey-6")
                            ui.label(str(row[col_name])).classes("text-h6")

        # ---- 图表 Tabs ----
        equity_df = data.get("equity_df")

        with ui.tabs() as tabs:
            tab_equity = ui.tab("权益曲线", icon="show_chart")
            tab_trades = ui.tab("交易记录", icon="receipt")
            tab_search = ui.tab("搜索日志", icon="search")

        with ui.tab_panels(tabs, value=tab_equity).classes("w-full"):
            with ui.tab_panel(tab_equity):
                if equity_df is not None and not equity_df.empty:
                    cols = equity_df.columns.tolist()
                    date_col = next((c for c in cols if "date" in str(c).lower() or "日期" in str(c)), cols[0])
                    equity_col = next((c for c in cols if "equity" in str(c).lower() or "净值" in str(c) or "cumulative" in str(c).lower()), cols[1] if len(cols) > 1 else cols[0])
                    dd_col = next((c for c in cols if "drawdown" in str(c).lower() or "回撤" in str(c)), None)

                    dates = equity_df[date_col].astype(str).tolist()
                    equity_values = equity_df[equity_col].tolist()
                    equity_curve_chart(dates, equity_values, title=f"{exp_name} 权益曲线")

                    if dd_col:
                        drawdown_chart(dates, equity_df[dd_col].tolist(), title=f"{exp_name} 回撤曲线")
                else:
                    ui.label("无权益曲线数据。").classes("text-grey-6")

            with ui.tab_panel(tab_trades):
                _render_df_table(data.get("trades_df"), "无交易记录数据。")

            with ui.tab_panel(tab_search):
                _render_df_table(data.get("search_df"), "无搜索日志数据。")

        # ---- 下载（在 data is not None 的 guard 内） ----
        report_path = Path(data["path"]) / "report.xlsx"
        if report_path.exists():
            ui.button(
                "📥 下载 Excel 报告",
                on_click=lambda: ui.download(str(report_path)),
                icon="download",
            )
