"""GUI 共享工具函数。

提供实验扫描、安全类型转换、路径校验、子进程运行等可复用工具。
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from nicegui import ui

from gui.components.log_panel import create_log_panel, stream_subprocess_output

logger = logging.getLogger(__name__)


# ── 安全类型转换 ──────────────────────────────────────────────


def safe_int(value, default: int | None = None) -> int | None:
    """安全地将 widget value 转为 int，处理 None 和空字符串。"""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value, default: float | None = None) -> float | None:
    """安全地将 widget value 转为 float。"""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ── 路径校验 ──────────────────────────────────────────────────


def validate_output_subpath(output_dir: str, name: str) -> Path | None:
    """校验 name 是 output_dir 下的合法子路径，防止路径穿越。

    Returns:
        解析后的合法路径，或 None（非法路径）。
    """
    base = Path(output_dir).resolve()
    target = (base / name).resolve()
    if not str(target).startswith(str(base)):
        return None
    return target


# ── 实验扫描 ──────────────────────────────────────────────────


def scan_experiments(
    output_dir: Path,
    *,
    require_report: bool = False,
    require_state: bool = False,
) -> list[dict]:
    """扫描 output 目录下的实验，返回统一的实验信息列表。

    Args:
        output_dir: 输出目录路径
        require_report: 仅返回有 report.xlsx 的实验
        require_state: 仅返回有 search_state.json 的实验

    Returns:
        实验信息字典列表，包含 name/path/mode/model_type/seed/created/
        has_report/has_state 等字段。
    """
    experiments: list[dict] = []
    if not output_dir.exists():
        return experiments

    for exp_dir in sorted(output_dir.iterdir(), reverse=True):
        if not exp_dir.is_dir():
            continue

        has_report = (exp_dir / "report.xlsx").exists()
        has_state = (exp_dir / "search_state.json").exists()

        if require_report and not has_report:
            continue
        if require_state and not has_state:
            continue

        info: dict = {
            "name": exp_dir.name,
            "path": str(exp_dir),
            "mode": "unknown",
            "model_type": "unknown",
            "seed": "?",
            "created": datetime.fromtimestamp(exp_dir.stat().st_mtime).strftime(
                "%Y-%m-%d %H:%M"
            ),
            "has_report": has_report,
            "has_state": has_state,
        }

        # 优先读 manifest，回退到 config.json
        manifest_path = exp_dir / "manifest.json"
        config_path = exp_dir / "config.json"

        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                cfg = manifest.get("config", {})
                info["mode"] = cfg.get("mode", "unknown")
                info["model_type"] = cfg.get("model", {}).get("model_type", "unknown")
                info["seed"] = cfg.get("search", {}).get("seed", "?")
            except Exception:
                pass

        if config_path.exists() and info["mode"] == "unknown":
            try:
                cfg = json.loads(config_path.read_text(encoding="utf-8"))
                info["mode"] = cfg.get("mode", "unknown")
                info["model_type"] = cfg.get("model", {}).get("model_type", "unknown")
            except Exception:
                pass

        experiments.append(info)

    return experiments


# ── 子进程运行 ────────────────────────────────────────────────


async def run_experiment_subprocess(
    subcommand: str,
    primary_arg_name: str,
    primary_arg_value: str,
    config_dict: dict,
    label: str,
    run_button,
    is_running: dict,
) -> None:
    """通用实验子进程运行器。

    将 config_dict 写入临时 JSON 文件，通过 --config 传递给 CLI 子进程，
    实时流式输出日志，运行完成后清理临时文件。

    Args:
        subcommand: CLI 子命令名（"single" / "basket"）
        primary_arg_name: 主参数名（"--data_path" / "--basket_path"）
        primary_arg_value: 主参数值
        config_dict: 完整的 RunConfig 字典
        label: 日志面板标题
        run_button: NiceGUI 按钮组件（运行期间 disable）
        is_running: 运行状态字典（{"value": bool}）
    """
    if is_running["value"]:
        ui.notify("已有实验正在运行，请等待完成", type="warning")
        return
    if not primary_arg_value or not primary_arg_value.strip():
        ui.notify(f"请填写{'数据文件路径' if 'data' in primary_arg_name else '数据目录'}", type="warning")
        return

    is_running["value"] = True
    run_button.disable()
    config_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
            config_path = f.name

        cmd = [sys.executable, "-m", "dpoint.cli.main", subcommand]
        cmd += [primary_arg_name, primary_arg_value.strip()]
        cmd += ["--config", config_path]

        log, status_label, progress = create_log_panel(label)
        ui.notify(f"开始运行{label}...", type="info")

        returncode = await stream_subprocess_output(cmd, log, status_label, progress)

        if returncode == 0:
            ui.notify(f"{label}运行完成！", type="positive")
        else:
            ui.notify(f"运行失败，退出码: {returncode}", type="negative")
    finally:
        is_running["value"] = False
        run_button.enable()
        if config_path:
            try:
                os.unlink(config_path)
            except OSError:
                pass
