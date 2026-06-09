"""共享布局组件：Header + Sidebar 导航。

每个页面函数调用 create_page_layout() 来创建统一的页面框架。
"""

from __future__ import annotations

from nicegui import ui, app


def create_page_layout() -> None:
    """创建标准页面布局（Header + 左侧导航栏）。

    在每个 @ui.page 函数开头调用此函数。
    返回后可在主内容区继续添加 UI 元素。
    """
    # Header
    with ui.header().classes("bg-blue-900"):
        ui.label("Dpoint Trader 量化研究平台").classes("text-h6 text-white")
        ui.space()

        # 暗色模式切换 — 使用 client storage 持久化偏好
        storage = app.storage.user
        is_dark = storage.get("dark_mode", False)
        dark = ui.dark_mode(value=is_dark)

        def toggle_dark():
            dark.toggle()
            storage["dark_mode"] = dark.value

        ui.button(on_click=toggle_dark, icon="dark_mode").props("flat color=white")

    # 左侧导航抽屉
    with ui.left_drawer(value=True).classes("bg-grey-2"):
        with ui.column().classes("w-full"):
            ui.label("导航").classes("text-h6 q-pa-md")
            ui.separator()

            nav_items = [
                ("🏠", "仪表盘", "/"),
                ("📈", "运行单股策略", "/run/single"),
                ("📊", "运行篮子策略", "/run/basket"),
                ("📥", "数据获取", "/fetch"),
                ("🔄", "恢复搜索", "/resume"),
                ("📋", "实验浏览", "/experiments"),
                ("🔍", "实验对比", "/compare"),
            ]

            for icon, label, path in nav_items:
                ui.link(f"{icon} {label}", path).classes(
                    "block q-pa-sm text-grey-8 no-underline"
                ).style("font-size: 14px")

            ui.separator()
            ui.label("设置").classes("text-caption q-pa-md text-grey-6")
