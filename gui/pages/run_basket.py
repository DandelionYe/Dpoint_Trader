"""运行篮子策略页面。

提供完整的篮子策略配置表单，启动运行并实时显示日志。
"""

from __future__ import annotations

from pathlib import Path

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
from gui.state import app_state
from gui.utils import safe_int, run_experiment_subprocess


def _scan_basket_dirs(data_dir: str | Path = "data") -> list[str]:
    """扫描 data/ 目录，返回包含 CSV 文件的子目录列表。"""
    data_path = Path(data_dir)
    if not data_path.is_dir():
        return []
    baskets = []
    for d in sorted(data_path.iterdir()):
        if d.is_dir() and list(d.glob("*.csv")):
            baskets.append(d.name)
    return baskets


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

            # 自动检测篮子目录
            basket_dirs = _scan_basket_dirs("data")
            basket_options = basket_dirs + ["(手动输入)"]

            with ui.row().classes("gap-4 w-full"):
                basket_select = ui.select(
                    basket_options,
                    label="篮子数据目录",
                    value=basket_dirs[0] if basket_dirs else "(手动输入)",
                ).tooltip("自动检测 data/ 下包含 CSV 文件的目录")

                basket_manual = ui.input(
                    label="自定义路径",
                    placeholder="如: data/basket/",
                )
                # 仅在选择"手动输入"时显示
                basket_manual.visible = (basket_select.value == "(手动输入)")
                basket_select.on_value_change(lambda e: setattr(basket_manual, "visible", e.value == "(手动输入)"))

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
                ).tooltip("none=不对齐, inner=取交集, outer=取并集, majority=取多数")

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

        # ---- 运行 ----
        run_button = ui.button("▶ 开始运行", color="green", icon="play_arrow").classes("text-white")
        is_running = {"value": False}

        async def on_run():
            feat_vals = collect_form_values(feature_widgets)
            model_vals = collect_form_values(model_widgets)
            trade_vals = collect_form_values(trade_widgets)
            search_vals = collect_form_values(search_widgets)
            split_vals = collect_form_values(split_widgets)
            portfolio_vals = collect_form_values(portfolio_widgets)

            # 确定篮子路径
            if basket_select.value == "(手动输入)":
                resolved_basket_path = (basket_manual.value or "").strip()
            else:
                resolved_basket_path = f"data/{basket_select.value}"

            config_dict = {
                "mode": "basket",
                "basket_path": resolved_basket_path,
                "output_dir": output_dir.value or "output",
                "seed": safe_int(search_vals.get("seed"), 42),
                "device": device.value,
                "feature": feat_vals,
                "model": model_vals,
                "trade": trade_vals,
                "search": {
                    "n_candidates": safe_int(search_vals.get("n_candidates"), 100),
                    "n_rounds": safe_int(search_vals.get("n_rounds"), 4),
                    "metric": search_vals.get("metric", "rank_ic"),
                    "seed": safe_int(search_vals.get("seed"), 42),
                },
                "split": split_vals,
                "portfolio": portfolio_vals,
            }

            await run_experiment_subprocess(
                subcommand="basket",
                primary_arg_name="--basket_path",
                primary_arg_value=resolved_basket_path,
                config_dict=config_dict,
                label="篮子策略",
                run_button=run_button,
                is_running=is_running,
            )

        run_button.on_click(on_run)
