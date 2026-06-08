"""运行篮子策略页面。

提供完整的篮子策略配置表单，启动运行并实时显示日志。
"""

from __future__ import annotations

import json
import sys
import tempfile

from nicegui import ui

from gui.components.layout import create_page_layout
from gui.components.config_forms import (
    feature_config_form,
    model_config_form,
    trade_config_form,
    search_config_form,
    split_config_form,
    portfolio_config_form,
    collect_form_values,
)
from gui.components.log_panel import create_log_panel, stream_subprocess_output
from gui.state import app_state


def _safe_int(value, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@ui.page("/run/basket")
def run_basket_page():
    create_page_layout()

    with ui.column().classes("w-full p-4 gap-4"):
        ui.label("运行篮子策略").classes("text-h4")
        ui.label(
            "配置参数后点击「开始运行」，系统将调用 dpoint basket 命令执行。"
        ).classes("text-grey-6")

        # ---- 基本参数 ----
        with ui.card().classes("w-full"):
            ui.label("基本参数").classes("text-h6")
            ui.separator()
            with ui.row().classes("gap-4 w-full"):
                basket_path = ui.input(
                    label="篮子数据目录",
                    placeholder="如: data/basket/",
                ).tooltip("包含多个 CSV 文件的目录，每个文件对应一只股票")

                output_dir = ui.input(
                    label="输出目录", value=app_state.output_dir
                )

            with ui.row().classes("gap-4 w-full"):
                device = ui.select(
                    ["auto", "cpu", "cuda"],
                    label="计算设备",
                    value="auto",
                )
                calendar_align = ui.select(
                    ["none", "inner", "outer", "majority"],
                    label="日历对齐",
                    value="none",
                ).tooltip(
                    "none=不对齐, inner=取交集, outer=取并集, majority=取多数"
                )

        # ---- Tab 配置面板 ----
        with ui.tabs() as tabs:
            tab_feature = ui.tab("特征工程", icon="tune")
            tab_model = ui.tab("模型", icon="smart_toy")
            tab_search = ui.tab("搜索", icon="search")
            tab_portfolio = ui.tab("组合", icon="pie_chart")
            tab_trade = ui.tab("交易/分割", icon="show_chart")

        with ui.tab_panels(tabs, value=tab_feature).classes("w-full"):
            with ui.tab_panel(tab_feature):
                feature_widgets = feature_config_form()
            with ui.tab_panel(tab_model):
                model_widgets = model_config_form()
            with ui.tab_panel(tab_search):
                search_widgets = search_config_form({"metric": "rank_ic"})
            with ui.tab_panel(tab_portfolio):
                portfolio_widgets = portfolio_config_form()
            with ui.tab_panel(tab_trade):
                trade_widgets = trade_config_form()
                split_widgets = split_config_form()

        # ---- 运行按钮和状态 ----
        run_button = ui.button(
            "▶ 开始运行",
            color="green",
            icon="play_arrow",
        ).classes("text-white")
        is_running = {"value": False}

        async def on_run():
            if is_running["value"]:
                ui.notify("已有实验正在运行，请等待完成", type="warning")
                return
            if not basket_path.value or not basket_path.value.strip():
                ui.notify("请填写篮子数据目录", type="warning")
                return

            is_running["value"] = True
            run_button.disable()

            try:
                feat_vals = collect_form_values(feature_widgets)
                model_vals = collect_form_values(model_widgets)
                trade_vals = collect_form_values(trade_widgets)
                search_vals = collect_form_values(search_widgets)
                split_vals = collect_form_values(split_widgets)
                portfolio_vals = collect_form_values(portfolio_widgets)

                config_dict = {
                    "mode": "basket",
                    "basket_path": basket_path.value.strip(),
                    "output_dir": output_dir.value or "output",
                    "seed": _safe_int(search_vals.get("seed"), 42),
                    "device": device.value,
                    "feature": feat_vals,
                    "model": model_vals,
                    "trade": trade_vals,
                    "search": {
                        "n_candidates": _safe_int(search_vals.get("n_candidates"), 100),
                        "n_rounds": _safe_int(search_vals.get("n_rounds"), 4),
                        "metric": search_vals.get("metric", "rank_ic"),
                        "seed": _safe_int(search_vals.get("seed"), 42),
                    },
                    "split": split_vals,
                    "portfolio": portfolio_vals,
                }

                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False, encoding="utf-8"
                ) as f:
                    json.dump(config_dict, f, ensure_ascii=False, indent=2)
                    config_path = f.name

                cmd = [sys.executable, "-m", "dpoint.cli.main", "basket"]
                cmd += ["--basket_path", basket_path.value.strip()]
                cmd += ["--config", config_path]

                log, status_label, progress = create_log_panel("篮子策略")
                ui.notify("开始运行篮子策略...", type="info")

                returncode = await stream_subprocess_output(cmd, log, status_label, progress)

                if returncode == 0:
                    ui.notify("篮子策略运行完成！", type="positive")
                else:
                    ui.notify(f"运行失败，退出码: {returncode}", type="negative")
            finally:
                is_running["value"] = False
                run_button.enable()

        run_button.on_click(on_run)
