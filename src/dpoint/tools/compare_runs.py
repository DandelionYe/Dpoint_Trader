# compare_runs.py
"""
多次运行结果对比工具。
来自两个项目的 compare_runs.py。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def load_run_manifest(exp_dir: str | Path) -> Optional[Dict[str, Any]]:
    """加载实验 manifest。"""
    manifest_path = Path(exp_dir) / "manifest.json"
    if not manifest_path.exists():
        return None
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_run_config(exp_dir: str | Path) -> Optional[Dict[str, Any]]:
    """加载实验配置。"""
    config_path = Path(exp_dir) / "config.json"
    if not config_path.exists():
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_run_metrics(exp_dir: str | Path) -> Optional[Dict[str, float]]:
    """从 Excel 报告中加载风险指标。"""
    report_path = Path(exp_dir) / "report.xlsx"
    if not report_path.exists():
        return None
    try:
        df = pd.read_excel(report_path, sheet_name="RiskMetrics")
        if not df.empty:
            return df.iloc[0].to_dict()
    except Exception:
        pass
    return None


def compare_runs(
    exp_dirs: List[str | Path],
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    对比多个实验的结果。

    Args:
        exp_dirs: 实验目录列表
        metrics: 要对比的指标列表（默认全部）

    Returns:
        DataFrame，每行一个实验，每列一个指标
    """
    if metrics is None:
        metrics = [
            "total_return",
            "annual_return",
            "sharpe",
            "sortino",
            "max_drawdown",
            "calmar",
            "win_rate",
            "n_trades",
            "rank_ic_mean",
            "rank_ic_ir",
            "topk_return_mean",
        ]

    rows = []
    for exp_dir in exp_dirs:
        exp_dir = Path(exp_dir)
        row = {"experiment": exp_dir.name}

        # 加载 manifest
        manifest = load_run_manifest(exp_dir)
        if manifest:
            row["seed"] = manifest.get("seed", "")
            row["git_commit"] = manifest.get("git_commit", "")
            row["created_at"] = manifest.get("created_at", "")

        # 加载配置
        config = load_run_config(exp_dir)
        if config:
            row["model_type"] = config.get("model", {}).get("model_type", "")
            row["metric"] = config.get("search", {}).get("metric", "")

        # 加载指标
        run_metrics = load_run_metrics(exp_dir)
        if run_metrics:
            for m in metrics:
                if m in run_metrics:
                    row[m] = run_metrics[m]

        rows.append(row)

    df = pd.DataFrame(rows)

    # 按 total_return 排序
    if "total_return" in df.columns:
        df = df.sort_values("total_return", ascending=False).reset_index(drop=True)

    return df


def find_experiments(output_dir: str | Path, prefix: str = "") -> List[Path]:
    """发现输出目录中的所有实验。"""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return []

    experiments = []
    for d in sorted(output_dir.iterdir()):
        if d.is_dir():
            if prefix and not d.name.startswith(prefix):
                continue
            if (d / "manifest.json").exists() or (d / "config.json").exists():
                experiments.append(d)

    return experiments


def print_comparison(df: pd.DataFrame) -> None:
    """打印对比表格。"""
    try:
        from tabulate import tabulate

        print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))
    except ImportError:
        print(df.to_string(index=False))
