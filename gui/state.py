"""全局状态管理模块。

管理 GUI 运行时的共享状态，包括正在运行的任务、实验缓存等。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RunningTask:
    """正在运行的实验任务。"""

    name: str
    mode: str  # "single" / "basket" / "resume"
    process: asyncio.subprocess.Process | None = None
    task: asyncio.Task | None = None
    log_lines: list[str] = field(default_factory=list)
    status: str = "pending"  # pending / running / success / failed


@dataclass
class AppState:
    """应用全局状态（单例）。"""

    output_dir: str = "output"
    running_tasks: dict[str, RunningTask] = field(default_factory=dict)
    experiments_cache: list[dict] = field(default_factory=list)
    experiments_cache_time: float = 0.0

    def get_output_path(self) -> Path:
        return Path(self.output_dir)


# 全局单例
app_state = AppState()
