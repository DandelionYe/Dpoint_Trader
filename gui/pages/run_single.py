"""运行单股策略页面。

提供完整的单股策略配置表单，启动运行并实时显示日志。
"""

from __future__ import annotations

import asyncio
import sys

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

        # ---- 运行按钮 ----
        async def on_run():
            if not data_path.value:
                ui.notify("请填写数据文件路径", type="warning")
                return

            # 构建 CLI 命令
            cmd = [sys.executable, "-m", "dpoint.cli.main", "single"]
            cmd += ["--data_path", data_path.value]
            cmd += ["--model", model_widgets["model_type"].value]
            cmd += ["--runs", str(int(search_widgets["n_candidates"].value))]
            cmd += ["--n_rounds", str(int(search_widgets["n_rounds"].value))]
            cmd += ["--metric", search_widgets["metric"].value]
            cmd += ["--seed", str(int(search_widgets["seed"].value))]
            cmd += ["--output", output_dir.value or "output"]
            cmd += ["--device", device.value]
            cmd += ["--n_folds", str(int(split_widgets["n_folds"].value))]
            cmd += ["--holdout_ratio", str(split_widgets["holdout_ratio"].value)]

            # 显示日志面板
            log, status_label, progress = create_log_panel("单股策略")
            ui.notify("开始运行单股策略...", type="info")

            returncode = await stream_subprocess_output(cmd, log, status_label, progress)

            if returncode == 0:
                ui.notify("单股策略运行完成！", type="positive")
            else:
                ui.notify(f"运行失败，退出码: {returncode}", type="negative")

        with ui.row().classes("gap-4 q-mt-md"):
            ui.button(
                "▶ 开始运行",
                on_click=on_run,
                color="blue",
                icon="play_arrow",
            ).classes("text-white")
