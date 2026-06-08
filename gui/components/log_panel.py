"""实时日志面板组件。

提供运行监控面板，包含日志输出区域和状态指示器。
"""

from __future__ import annotations

import asyncio

from nicegui import ui, app


def create_log_panel(task_name: str = "实验") -> tuple:
    """创建运行监控面板。

    Returns:
        (log_element, status_label, progress_circular) 元组
    """
    with ui.card().classes("w-full"):
        with ui.row().classes("items-center gap-2"):
            ui.label(f"运行监控 - {task_name}").classes("text-h6")
            status_label = ui.label("等待中...").classes("text-caption text-grey-6")
            progress_circular = ui.circular_progress(
                value=0, show_value=False, size="24px", color="blue"
            ).classes("q-ml-auto")
            progress_circular.visible = False

        log = ui.log(max_lines=500).classes(
            "w-full font-mono text-sm bg-grey-1"
        ).style("height: 400px;")

    return log, status_label, progress_circular


async def stream_subprocess_output(
    cmd: list[str],
    log_element,
    status_label=None,
    progress_circular=None,
    cwd: str | None = None,
) -> int:
    """以子进程方式运行命令，并实时流式输出日志。

    Args:
        cmd: 命令行参数列表
        log_element: ui.log 组件
        status_label: 状态标签组件（可选）
        progress_circular: 进度指示器（可选）
        cwd: 工作目录（可选）

    Returns:
        进程退出码
    """
    if status_label:
        status_label.text = "运行中..."
        status_label.classes(replace="text-caption text-blue")
    if progress_circular:
        progress_circular.visible = True

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cwd,
        )

        async for raw_line in process.stdout:
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            log_element.push(line)

        returncode = await process.wait()

        if returncode == 0:
            if status_label:
                status_label.text = "运行完成 ✓"
                status_label.classes(replace="text-caption text-positive")
            log_element.push("--- 运行完成 ---")
        else:
            if status_label:
                status_label.text = f"运行失败 (退出码: {returncode})"
                status_label.classes(replace="text-caption text-negative")
            log_element.push(f"--- 运行失败，退出码: {returncode} ---")

        return returncode

    except (OSError, asyncio.CancelledError) as e:
        if status_label:
            status_label.text = f"错误: {e}"
            status_label.classes(replace="text-caption text-negative")
        log_element.push(f"错误: {e}")
        return -1

    finally:
        if progress_circular:
            progress_circular.visible = False
