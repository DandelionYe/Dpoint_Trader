"""实验对比页面。

选择多个实验进行关键指标并列对比和权益曲线叠加。
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from nicegui import ui

from gui.components.layout import create_page_layout
from gui.components.charts import comparison_equity_chart
from gui.state import app_state
from gui.utils import scan_experiments, validate_output_subpath

logger = logging.getLogger(__name__)


def _load_risk_metrics(exp_path: str) -> dict:
    """加载单个实验的风险指标。RiskMetrics 是列方向：指标名为列头，一行为值。"""
    report_path = Path(exp_path) / "report.xlsx"
    metrics = {}
    if report_path.exists():
        try:
            df = pd.read_excel(report_path, sheet_name="RiskMetrics")
            if not df.empty:
                row = df.iloc[0]
                for col_name in df.columns:
                    metrics[str(col_name)] = str(row[col_name])
        except ValueError:
            pass
        except Exception as e:
            logger.warning("读取 RiskMetrics 失败: %s", e)
    return metrics


def _load_equity_curve(exp_path: str) -> tuple[list[str], list[float]]:
    """加载单个实验的权益曲线。"""
    report_path = Path(exp_path) / "report.xlsx"
    if not report_path.exists():
        return [], []
    try:
        df = pd.read_excel(report_path, sheet_name="EquityCurve")
        cols = df.columns.tolist()
        date_col = cols[0]
        equity_col = cols[1] if len(cols) > 1 else cols[0]
        for c in cols:
            cl = str(c).lower()
            if "date" in cl or "日期" in cl:
                date_col = c
            elif "equity" in cl or "净值" in cl:
                equity_col = c
        return df[date_col].astype(str).tolist(), df[equity_col].tolist()
    except Exception:
        return [], []


@ui.page("/compare")
def compare_page():
    create_page_layout()

    with ui.column().classes("w-full p-4 gap-4"):
        ui.label("实验对比").classes("text-h4")
        ui.label("选择多个实验进行指标对比和权益曲线叠加。").classes("text-grey-6")

        output_dir = Path(app_state.output_dir)
        experiments = scan_experiments(output_dir, require_report=True)

        if not experiments:
            ui.label("没有可对比的实验（需要有 Excel 报告）。").classes("text-grey-6")
            return

        # 选择实验
        exp_options = {e["path"]: f"{e['name']} ({e['mode']})" for e in experiments}
        selected = ui.select(
            exp_options,
            label="选择要对比的实验（可多选）",
            multiple=True,
            value=[],
        ).classes("w-full")

        # 对比结果容器
        result_container = ui.column().classes("w-full gap-4")

        async def on_compare():
            paths = selected.value
            if not paths or len(paths) < 2:
                ui.notify("请至少选择 2 个实验", type="warning")
                return

            # 校验路径
            for p in paths:
                if validate_output_subpath(app_state.output_dir, Path(p).name) is None:
                    ui.notify(f"非法路径: {p}", type="negative")
                    return

            # 清空之前的结果
            result_container.clear()

            # 加载数据
            all_metrics = {}
            all_curves = {}
            all_dates = {}
            for path in paths:
                name = Path(path).name
                metrics = _load_risk_metrics(path)
                all_metrics[name] = metrics
                dates, equity = _load_equity_curve(path)
                if dates:
                    all_dates[name] = dates
                    all_curves[name] = equity

            with result_container:
                # 对比表格
                if all_metrics:
                    ui.label("风险指标对比").classes("text-h6")
                    all_keys = set()
                    for m in all_metrics.values():
                        all_keys.update(m.keys())

                    columns = [{"name": "metric", "label": "指标", "field": "metric", "align": "left"}]
                    for name in all_metrics:
                        columns.append(
                            {"name": name, "label": name, "field": name, "align": "center"}
                        )

                    rows = []
                    for key in sorted(all_keys):
                        row = {"metric": key}
                        for name, m in all_metrics.items():
                            row[name] = m.get(key, "-")
                        rows.append(row)

                    ui.table(columns=columns, rows=rows, row_key="metric").classes("w-full")

                # 权益曲线叠加 — 使用所有实验日期的并集，对齐数据
                if all_curves:
                    ui.label("权益曲线叠加").classes("text-h6")
                    # 收集所有日期的并集并排序
                    all_dates_union = sorted(set(d for dates in all_dates.values() for d in dates))
                    # 对齐每个曲线到并集日期
                    aligned_curves = {}
                    for name, dates in all_dates.items():
                        date_to_val = dict(zip(dates, all_curves[name]))
                        aligned = [date_to_val.get(d, None) for d in all_dates_union]
                        # 去掉前导 None（用第一个有效值之前的 None）
                        aligned_curves[name] = aligned
                    comparison_equity_chart(all_dates_union, aligned_curves)

        ui.button(
            "📊 开始对比",
            on_click=on_compare,
            color="blue",
            icon="compare",
        )
