"""实验对比页面。

选择多个实验进行关键指标并列对比和权益曲线叠加。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from nicegui import ui

from gui.components.layout import create_page_layout
from gui.components.charts import comparison_equity_chart
from gui.state import app_state


def _list_experiments_for_compare(output_dir: Path) -> list[dict]:
    """列出有报告的实验。"""
    experiments = []
    if not output_dir.exists():
        return experiments
    for exp_dir in sorted(output_dir.iterdir(), reverse=True):
        if not exp_dir.is_dir():
            continue
        report_path = exp_dir / "report.xlsx"
        config_path = exp_dir / "config.json"
        mode = "unknown"
        if config_path.exists():
            try:
                cfg = json.loads(config_path.read_text(encoding="utf-8"))
                mode = cfg.get("mode", "unknown")
            except Exception:
                pass
        experiments.append({
            "name": exp_dir.name,
            "path": str(exp_dir),
            "mode": mode,
            "has_report": report_path.exists(),
        })
    return experiments


def _load_risk_metrics(exp_path: str) -> dict:
    """加载单个实验的风险指标。"""
    report_path = Path(exp_path) / "report.xlsx"
    metrics = {}
    if report_path.exists():
        try:
            df = pd.read_excel(report_path, sheet_name="RiskMetrics")
            for _, row in df.iterrows():
                if len(row) >= 2:
                    metrics[str(row.iloc[0])] = str(row.iloc[1])
        except Exception:
            pass
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
        experiments = _list_experiments_for_compare(output_dir)
        available = [e for e in experiments if e["has_report"]]

        if not available:
            ui.label("没有可对比的实验（需要有 Excel 报告）。").classes("text-grey-6")
            return

        # 选择实验
        exp_options = {e["path"]: f"{e['name']} ({e['mode']})" for e in available}
        selected = ui.select(
            exp_options,
            label="选择要对比的实验（可多选）",
            multiple=True,
            value=[],
        ).classes("w-full")

        async def on_compare():
            paths = selected.value
            if not paths or len(paths) < 2:
                ui.notify("请至少选择 2 个实验", type="warning")
                return

            # 加载数据
            all_metrics = {}
            all_curves = {}
            for path in paths:
                name = Path(path).name
                metrics = _load_risk_metrics(path)
                all_metrics[name] = metrics
                dates, equity = _load_equity_curve(path)
                if dates:
                    all_curves[name] = equity

            # 对比表格
            if all_metrics:
                ui.label("风险指标对比").classes("text-h6 q-mt-md")
                # 收集所有指标名
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

            # 权益曲线叠加
            if all_curves:
                ui.label("权益曲线叠加").classes("text-h6 q-mt-md")
                # 使用最长的日期序列
                longest_dates = max(all_curves.values(), key=len) if all_curves else []
                # 对齐日期（简化：使用第一个实验的日期）
                first_path = paths[0]
                dates, _ = _load_equity_curve(first_path)
                if dates:
                    comparison_equity_chart(dates, all_curves)

        ui.button(
            "📊 开始对比",
            on_click=on_compare,
            color="blue",
            icon="compare",
        )
