"""恢复搜索页面。

从已有实验中恢复迭代搜索，可覆盖部分参数。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from nicegui import ui

from gui.components.layout import create_page_layout
from gui.components.log_panel import create_log_panel, stream_subprocess_output
from gui.state import app_state


def _list_experiments(output_dir: Path) -> list[dict]:
    """列出已有实验。"""
    experiments = []
    if not output_dir.exists():
        return experiments
    for exp_dir in sorted(output_dir.iterdir(), reverse=True):
        if not exp_dir.is_dir():
            continue
        state_path = exp_dir / "search_state.json"
        if state_path.exists():
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
            })
    return experiments


@ui.page("/resume")
def resume_page():
    create_page_layout()

    with ui.column().classes("w-full p-4 gap-4"):
        ui.label("恢复搜索").classes("text-h4")
        ui.label(
            "从已有实验恢复迭代搜索，可覆盖搜索轮数、种子等参数。"
        ).classes("text-grey-6")

        # 扫描实验
        output_dir = Path(app_state.output_dir)
        experiments = _list_experiments(output_dir)

        if not experiments:
            ui.label("未找到可恢复的实验。请先运行单股或篮子策略。").classes("text-grey-6")
            return

        # 实验选择
        with ui.card().classes("w-full"):
            ui.label("选择实验").classes("text-h6")
            ui.separator()

            exp_options = {e["path"]: f"{e['name']} ({e['mode']})" for e in experiments}
            selected_exp = ui.select(
                exp_options,
                label="实验目录",
                value=list(exp_options.keys())[0],
            ).classes("w-full")

        # 覆盖参数
        with ui.card().classes("w-full"):
            ui.label("覆盖参数（可选）").classes("text-h6")
            ui.separator()

            with ui.row().classes("gap-4 w-full"):
                runs = ui.number(
                    "本轮搜索候选数", value=100, min=10, max=5000, step=10
                )
                n_rounds = ui.number(
                    "本轮搜索轮数", value=4, min=1, max=20, step=1
                )

            with ui.row().classes("gap-4 w-full"):
                metric = ui.select(
                    ["pnl", "rank_ic"],
                    label="搜索目标（留空则沿用原配置）",
                    value=None,
                )
                seed = ui.number(
                    "新种子（留空则恢复原 RNG 状态）",
                    value=None,
                    min=0,
                    max=999999,
                )

            with ui.row().classes("gap-4 w-full"):
                output = ui.input(
                    label="输出目录", value=app_state.output_dir
                )
                device = ui.select(
                    ["auto", "cpu", "cuda"],
                    label="计算设备",
                    value="auto",
                )

        # 运行
        async def on_run():
            if not selected_exp.value:
                ui.notify("请选择一个实验", type="warning")
                return

            cmd = [sys.executable, "-m", "dpoint.cli.main", "resume"]
            cmd += [selected_exp.value]
            cmd += ["--runs", str(int(runs.value))]
            cmd += ["--n_rounds", str(int(n_rounds.value))]
            if metric.value:
                cmd += ["--metric", metric.value]
            if seed.value is not None:
                cmd += ["--seed", str(int(seed.value))]
            cmd += ["--output", output.value or "output"]
            cmd += ["--device", device.value]

            log, status_label, progress = create_log_panel("恢复搜索")
            ui.notify("开始恢复搜索...", type="info")

            returncode = await stream_subprocess_output(cmd, log, status_label, progress)

            if returncode == 0:
                ui.notify("恢复搜索完成！", type="positive")
            else:
                ui.notify(f"运行失败，退出码: {returncode}", type="negative")

        with ui.row().classes("gap-4 q-mt-md"):
            ui.button(
                "▶ 恢复搜索",
                on_click=on_run,
                color="orange",
                icon="play_arrow",
            ).classes("text-white")
