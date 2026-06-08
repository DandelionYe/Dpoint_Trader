"""GUI 启动器。

提供 `dpoint-gui` 入口点，将 gui/ 目录添加到 sys.path 后启动 NiceGUI。
"""

from __future__ import annotations

import sys
from pathlib import Path


def main():
    # 将项目根目录加入 sys.path，以便 import gui 模块
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from gui.app import main as gui_main

    gui_main()
