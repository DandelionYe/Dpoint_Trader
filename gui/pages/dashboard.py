"""首页仪表盘。

显示统计信息、最近实验和快捷操作。
"""

from __future__ import annotations

from pathlib import Path

from nicegui import ui

from gui.components.layout import create_page_layout
from gui.state import app_state
from gui.utils import scan_experiments


@ui.page("/")
def dashboard_page():
    create_page_layout()

    with ui.column().classes("w-full p-4 gap-4"):
        ui.label("仪表盘").classes("text-h4")

        # 统计卡片
        output_dir = Path(app_state.output_dir)
        experiments = scan_experiments(output_dir)

        with ui.row().classes("gap-4"):
            with ui.card().classes("flex-1"):
                with ui.card_section():
                    ui.label("总实验数").classes("text-caption text-grey-6")
                    ui.label(str(len(experiments))).classes("text-h4 text-blue")

            with ui.card().classes("flex-1"):
                with ui.card_section():
                    ui.label("单股实验").classes("text-caption text-grey-6")
                    count = sum(1 for e in experiments if e["mode"] == "single")
                    ui.label(str(count)).classes("text-h4 text-green")

            with ui.card().classes("flex-1"):
                with ui.card_section():
                    ui.label("篮子实验").classes("text-caption text-grey-6")
                    count = sum(1 for e in experiments if e["mode"] == "basket")
                    ui.label(str(count)).classes("text-h4 text-orange")

            with ui.card().classes("flex-1"):
                with ui.card_section():
                    ui.label("最近运行").classes("text-caption text-grey-6")
                    last = experiments[0]["created"] if experiments else "无"
                    ui.label(last).classes("text-body1")

        # 快捷操作
        ui.label("快捷操作").classes("text-h6 q-mt-md")
        with ui.row().classes("gap-4"):
            ui.button(
                "📈 运行单股策略",
                on_click=lambda: ui.navigate.to("/run/single"),
                color="blue",
            )
            ui.button(
                "📊 运行篮子策略",
                on_click=lambda: ui.navigate.to("/run/basket"),
                color="green",
            )
            ui.button(
                "🔄 恢复搜索",
                on_click=lambda: ui.navigate.to("/resume"),
                color="orange",
            )
            ui.button(
                "📥 数据获取",
                on_click=lambda: ui.navigate.to("/fetch"),
                color="purple",
            )

        # 最近实验列表
        ui.label("最近实验").classes("text-h6 q-mt-md")
        if not experiments:
            ui.label("暂无实验记录。请先运行一个策略。").classes("text-grey-6")
        else:
            columns = [
                {"name": "name", "label": "实验名称", "field": "name", "align": "left"},
                {"name": "mode", "label": "模式", "field": "mode", "align": "center"},
                {"name": "model_type", "label": "模型", "field": "model_type", "align": "center"},
                {"name": "created", "label": "创建时间", "field": "created", "align": "center"},
            ]
            rows = experiments[:20]

            table = ui.table(columns=columns, rows=rows, row_key="name").classes("w-full")
            table.on(
                "rowClick",
                lambda e: ui.navigate.to(f"/results/{e.args[1]['name']}"),
            )
