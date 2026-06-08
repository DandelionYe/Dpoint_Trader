"""运行单股策略页面。

提供完整的单股策略配置表单，启动运行并实时显示日志。
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from nicegui import ui

from gui.components.layout import create_page_layout
from gui.components.config_forms import (
    feature_config_form,
    model_config_form,
    trade_config_form,
    search_config_form,
    split_config_form,
    collect_form_values,
)
from gui.components.log_panel import create_log_panel, stream_subprocess_output
from gui.state import app_state


def _safe_int(value, default: int) -> int:
    """安全地将 widget value 转为 int，处理 None 和空字符串。"""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value, default: float) -> float:
    """安全地将 widget value 转为 float。"""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@ui.page("/run/single")
def run_single_page():
    create_page_layout()

    with ui.column().classes("w-full p-4 gap-4"):
        ui.label("运行单股策略").classes("text-h4")
        ui.label("配置参数后点击「开始运行」，系统将调用 dpoint single 命令执行。").classes(
            "text-grey-6"
        )

        # ---- 基本参数 ----
        with ui.card().classes("w-full"):
            ui.label("基本参数").classes("text-h6")
            ui.separator()
            with ui.row().classes("gap-4 w-full"):
                data_path = ui.input(
                    label="数据文件路径",
                    placeholder="如: data/600519.xlsx",
                ).tooltip("支持 Excel(.xlsx) 或 CSV(.csv) 格式")

                output_dir = ui.input(
                    label="输出目录", value=app_state.output_dir
                )

            with ui.row().classes("gap-4 w-full"):
                device = ui.select(
                    ["auto", "cpu", "cuda"],
                    label="计算设备",
                    value="auto",
                )

        # ---- Tab 配置面板 ----
        with ui.tabs() as tabs:
            tab_feature = ui.tab("特征工程", icon="tune")
            tab_model = ui.tab("模型", icon="smart_toy")
            tab_search = ui.tab("搜索", icon="search")
            tab_trade = ui.tab("交易/分割", icon="show_chart")

        with ui.tab_panels(tabs, value=tab_feature).classes("w-full"):
            with ui.tab_panel(tab_feature):
                feature_widgets = feature_config_form()
            with ui.tab_panel(tab_model):
                model_widgets = model_config_form()
            with ui.tab_panel(tab_search):
                search_widgets = search_config_form()
            with ui.tab_panel(tab_trade):
                trade_widgets = trade_config_form()
                split_widgets = split_config_form()

        # ---- 运行按钮和状态 ----
        run_button = ui.button(
            "▶ 开始运行",
            color="blue",
            icon="play_arrow",
        ).classes("text-white")
        is_running = {"value": False}

        async def on_run():
            if is_running["value"]:
                ui.notify("已有实验正在运行，请等待完成", type="warning")
                return
            if not data_path.value or not data_path.value.strip():
                ui.notify("请填写数据文件路径", type="warning")
                return

            is_running["value"] = True
            run_button.disable()

            try:
                # 收集所有表单参数，写入临时 JSON 配置文件
                feat_vals = collect_form_values(feature_widgets)
                model_vals = collect_form_values(model_widgets)
                trade_vals = collect_form_values(trade_widgets)
                search_vals = collect_form_values(search_widgets)
                split_vals = collect_form_values(split_widgets)

                config_dict = {
                    "mode": "single",
                    "data_path": data_path.value.strip(),
                    "output_dir": output_dir.value or "output",
                    "seed": _safe_int(search_vals.get("seed"), 42),
                    "device": device.value,
                    "feature": feat_vals,
                    "model": model_vals,
                    "trade": trade_vals,
                    "search": {
                        "n_candidates": _safe_int(search_vals.get("n_candidates"), 100),
                        "n_rounds": _safe_int(search_vals.get("n_rounds"), 4),
                        "metric": search_vals.get("metric", "pnl"),
                        "seed": _safe_int(search_vals.get("seed"), 42),
                    },
                    "split": split_vals,
                }

                # 写入临时配置文件
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False, encoding="utf-8"
                ) as f:
                    json.dump(config_dict, f, ensure_ascii=False, indent=2)
                    config_path = f.name

                cmd = [sys.executable, "-m", "dpoint.cli.main", "single"]
                cmd += ["--data_path", data_path.value.strip()]
                cmd += ["--config", config_path]

                log, status_label, progress = create_log_panel("单股策略")
                ui.notify("开始运行单股策略...", type="info")

                returncode = await stream_subprocess_output(cmd, log, status_label, progress)

                if returncode == 0:
                    ui.notify("单股策略运行完成！", type="positive")
                else:
                    ui.notify(f"运行失败，退出码: {returncode}", type="negative")
            finally:
                is_running["value"] = False
                run_button.enable()

        run_button.on_click(on_run)
