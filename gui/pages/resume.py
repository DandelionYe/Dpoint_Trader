"""恢复搜索页面。

从已有实验中恢复迭代搜索，可覆盖部分参数。
"""

from __future__ import annotations

import sys
from pathlib import Path

from nicegui import ui

from gui.components.layout import create_page_layout
from gui.components.log_panel import create_log_panel, stream_subprocess_output
from gui.state import app_state
from gui.utils import safe_int, scan_experiments


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
        experiments = scan_experiments(output_dir, require_state=True)

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

        # 运行按钮和状态
        run_button = ui.button(
            "▶ 恢复搜索",
            color="orange",
            icon="play_arrow",
        ).classes("text-white")
        is_running = {"value": False}

        async def on_run():
            if is_running["value"]:
                ui.notify("已有实验正在运行，请等待完成", type="warning")
                return
            if not selected_exp.value:
                ui.notify("请选择一个实验", type="warning")
                return

            is_running["value"] = True
            run_button.disable()

            try:
                cmd = [sys.executable, "-m", "dpoint.cli.main", "resume"]
                cmd += [selected_exp.value]
                runs_val = safe_int(runs.value, 100)
                cmd += ["--runs", str(runs_val)]
                n_rounds_val = safe_int(n_rounds.value, 4)
                cmd += ["--n_rounds", str(n_rounds_val)]
                if metric.value:
                    cmd += ["--metric", metric.value]
                seed_val = safe_int(seed.value, None)
                if seed_val is not None:
                    cmd += ["--seed", str(seed_val)]
                cmd += ["--output", output.value or "output"]
                cmd += ["--device", device.value]

                log, status_label, progress = create_log_panel("恢复搜索")
                ui.notify("开始恢复搜索...", type="info")

                returncode = await stream_subprocess_output(cmd, log, status_label, progress)

                if returncode == 0:
                    ui.notify("恢复搜索完成！", type="positive")
                else:
                    ui.notify(f"运行失败，退出码: {returncode}", type="negative")
            finally:
                is_running["value"] = False
                run_button.enable()

        run_button.on_click(on_run)
