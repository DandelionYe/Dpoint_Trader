"""Dpoint Trader GUI 主入口。

启动 NiceGUI 服务器，注册所有页面路由。

启动方式:
    dpoint-gui                    # 浏览器模式
    python gui/app.py             # 浏览器模式
    python gui/app.py --native    # 桌面窗口模式
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中，以便 import dpoint 和 gui 模块
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from nicegui import ui

# 导入页面模块（触发 @ui.page 注册）
from gui.pages import dashboard  # noqa: F401
from gui.pages import run_single  # noqa: F401
from gui.pages import run_basket  # noqa: F401
from gui.pages import resume  # noqa: F401
from gui.pages import experiments  # noqa: F401
from gui.pages import results  # noqa: F401
from gui.pages import compare  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description="Dpoint Trader 量化研究平台 GUI")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址 (默认: 127.0.0.1，仅本机访问)")
    parser.add_argument("--port", type=int, default=8080, help="端口 (默认: 8080)")
    parser.add_argument("--native", action="store_true", help="桌面窗口模式")
    parser.add_argument("--dark", action="store_true", help="暗色模式")
    parser.add_argument(
        "--output",
        default="output",
        help="实验输出目录 (默认: output)",
    )
    args = parser.parse_args()

    # 设置全局输出目录
    from gui.state import app_state

    app_state.output_dir = args.output

    ui.run(
        title="Dpoint Trader 量化研究平台",
        host=args.host,
        port=args.port,
        native=args.native,
        dark=True if args.dark else None,
        reload=False,
        show=True,
        favicon="📈",
    )


if __name__ == "__main__":
    main()
