"""GUI 启动器。

提供 `dpoint-gui` 入口点，将 gui/ 目录添加到 sys.path 后启动 NiceGUI。
"""

from __future__ import annotations

import sys
from pathlib import Path


def main():
    # 将项目根目录加入 sys.path，以便 import gui 模块
    project_root = Path(__file__).resolve().parent.parent.parent
    gui_dir = project_root / "gui"
    if not gui_dir.is_dir():
        print(
            "错误: 找不到 gui/ 目录。\n"
            "dpoint-gui 需要以开发模式安装（pip install -e '.[gui]'），\n"
            "且必须在项目根目录下运行。\n"
            f"当前查找路径: {gui_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from gui.app import main as gui_main

        gui_main()
    except ImportError as e:
        print(
            f"错误: 缺少依赖 {e.name}。请运行: pip install -e '.[gui]'",
            file=sys.stderr,
        )
        sys.exit(1)
