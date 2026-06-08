"""实验浏览页面。

展示所有实验，支持搜索、筛选和跳转查看详情。
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from nicegui import ui

from gui.components.layout import create_page_layout
from gui.state import app_state
from gui.utils import scan_experiments

logger = logging.getLogger(__name__)


@ui.page("/experiments")
def experiments_page():
    create_page_layout()

    with ui.column().classes("w-full p-4 gap-4"):
        ui.label("实验浏览").classes("text-h4")

        output_dir = Path(app_state.output_dir)
        experiments = scan_experiments(output_dir)

        # 工具栏
        with ui.row().classes("gap-4 items-center w-full"):
            search_input = ui.input(
                label="搜索实验", placeholder="输入关键词筛选..."
            ).classes("flex-grow")
            mode_filter = ui.select(
                {"all": "全部", "single": "单股", "basket": "篮子"},
                label="模式筛选",
                value="all",
            )

            def refresh():
                ui.navigate.to("/experiments")

            ui.button("🔄 刷新", on_click=refresh, color="blue")

        # 表格
        if not experiments:
            ui.label("暂无实验记录。").classes("text-grey-6 q-mt-md")
            return

        columns = [
            {"name": "name", "label": "实验名称", "field": "name", "align": "left"},
            {"name": "mode", "label": "模式", "field": "mode", "align": "center"},
            {
                "name": "model_type",
                "label": "模型",
                "field": "model_type",
                "align": "center",
            },
            {"name": "seed", "label": "种子", "field": "seed", "align": "center"},
            {
                "name": "created",
                "label": "创建时间",
                "field": "created",
                "align": "center",
            },
            {
                "name": "has_report",
                "label": "报告",
                "field": "has_report",
                "align": "center",
            },
            {
                "name": "has_state",
                "label": "可恢复",
                "field": "has_state",
                "align": "center",
            },
        ]

        # 格式化行数据
        rows = []
        for exp in experiments:
            rows.append(
                {
                    "name": exp["name"],
                    "mode": exp["mode"],
                    "model_type": exp["model_type"],
                    "seed": str(exp["seed"]),
                    "created": exp["created"],
                    "has_report": "✓" if exp["has_report"] else "✗",
                    "has_state": "✓" if exp["has_state"] else "✗",
                    "path": exp["path"],
                }
            )

        table = ui.table(
            columns=columns,
            rows=rows,
            row_key="name",
            selection="multiple",
        ).classes("w-full")

        # 筛选逻辑
        def apply_filter():
            keyword = search_input.value.lower() if search_input.value else ""
            mode = mode_filter.value
            filtered = rows
            if keyword:
                filtered = [
                    r
                    for r in filtered
                    if keyword in r["name"].lower()
                    or keyword in r["model_type"].lower()
                ]
            if mode != "all":
                filtered = [r for r in filtered if r["mode"] == mode]
            table.rows = filtered
            # 清除选中状态，防止筛选后误删不可见的实验
            table.selected = []

        search_input.on("change", apply_filter)
        mode_filter.on("change", apply_filter)

        # 点击行查看详情
        table.on(
            "rowClick",
            lambda e: ui.navigate.to(f"/results/{e.args[1]['name']}"),
        )

        # 批量操作
        async def delete_selected():
            selected = table.selected
            if not selected:
                ui.notify("请先选择要删除的实验", type="warning")
                return

            names = [s["name"] for s in selected]
            # 确认对话框
            confirmed = await ui.dialog(
                f"确定要删除以下 {len(names)} 个实验吗？此操作不可撤销。\n" + "\n".join(names)
            ).add_button("取消", value=False).add_button("确定删除", value=True)

            if not confirmed:
                return

            success_count = 0
            fail_count = 0
            for s in selected:
                try:
                    shutil.rmtree(s["path"])
                    success_count += 1
                except Exception as e:
                    fail_count += 1
                    logger.error("删除实验 %s 失败: %s", s["name"], e)

            if fail_count > 0:
                ui.notify(
                    f"删除完成：成功 {success_count} 个，失败 {fail_count} 个",
                    type="warning",
                )
            else:
                ui.notify(f"已删除 {success_count} 个实验", type="positive")
            ui.navigate.to("/experiments")

        with ui.row().classes("gap-4 q-mt-sm"):
            ui.button(
                "🗑 删除选中",
                on_click=delete_selected,
                color="red",
            )
