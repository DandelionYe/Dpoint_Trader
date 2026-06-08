"""实验浏览页面。

展示所有实验，支持搜索、筛选和跳转查看详情。
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from nicegui import ui

from gui.components.layout import create_page_layout
from gui.state import app_state


def _scan_experiments(output_dir: Path) -> list[dict]:
    """扫描所有实验并返回详情列表。"""
    experiments = []
    if not output_dir.exists():
        return experiments

    for exp_dir in sorted(output_dir.iterdir(), reverse=True):
        if not exp_dir.is_dir():
            continue

        info = {
            "name": exp_dir.name,
            "path": str(exp_dir),
            "mode": "unknown",
            "model_type": "unknown",
            "seed": "?",
            "created": datetime.fromtimestamp(exp_dir.stat().st_ctime).strftime(
                "%Y-%m-%d %H:%M"
            ),
            "has_report": (exp_dir / "report.xlsx").exists(),
            "has_state": (exp_dir / "search_state.json").exists(),
        }

        # 读取 manifest
        manifest_path = exp_dir / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                cfg = manifest.get("config", {})
                info["mode"] = cfg.get("mode", "unknown")
                info["model_type"] = cfg.get("model", {}).get("model_type", "unknown")
                info["seed"] = cfg.get("search", {}).get("seed", "?")
            except Exception:
                pass

        # 读取 config.json
        config_path = exp_dir / "config.json"
        if config_path.exists() and info["mode"] == "unknown":
            try:
                cfg = json.loads(config_path.read_text(encoding="utf-8"))
                info["mode"] = cfg.get("mode", "unknown")
                info["model_type"] = cfg.get("model", {}).get("model_type", "unknown")
            except Exception:
                pass

        experiments.append(info)

    return experiments


@ui.page("/experiments")
def experiments_page():
    create_page_layout()

    with ui.column().classes("w-full p-4 gap-4"):
        ui.label("实验浏览").classes("text-h4")

        output_dir = Path(app_state.output_dir)
        experiments = _scan_experiments(output_dir)

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
            for s in selected:
                try:
                    shutil.rmtree(s["path"])
                except Exception:
                    pass
            ui.notify(f"已删除 {len(names)} 个实验", type="positive")
            ui.navigate.to("/experiments")

        with ui.row().classes("gap-4 q-mt-sm"):
            ui.button(
                "🗑 删除选中",
                on_click=delete_selected,
                color="red",
            )
