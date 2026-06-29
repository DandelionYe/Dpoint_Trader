# utils.py
"""
通用工具：种子设置、哈希、manifest、git 信息。
合并自两个项目的 repro.py / utils.py / run_manifest.py。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ==============================================================
# 种子管理
# ==============================================================


def set_global_seed(seed: int = 42) -> None:
    """设置全局随机种子（Python / NumPy / PyTorch）。"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)


# ==============================================================
# 哈希工具
# ==============================================================


def compute_data_hash(df: pd.DataFrame) -> str:
    """计算 DataFrame 的 SHA-256 哈希。"""
    sort_cols = [c for c in ["date", "ticker"] if c in df.columns]
    sorted_df = df.sort_values(sort_cols).reset_index(drop=True) if sort_cols else df
    return hashlib.sha256(pd.util.hash_pandas_object(sorted_df).values.tobytes()).hexdigest()


def compute_config_hash(config: dict) -> str:
    """计算配置字典的稳定哈希。"""
    encoded = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


# ==============================================================
# Git 信息
# ==============================================================


def get_git_commit_hash() -> str:
    """获取当前 git commit hash。"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# ==============================================================
# 包版本
# ==============================================================


def get_package_versions(packages: list[str]) -> dict[str, str]:
    """获取指定包的版本号。"""
    versions = {}
    for pkg in packages:
        try:
            import importlib

            mod = importlib.import_module(pkg.replace("-", "_"))
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except (ImportError, OSError, Exception):
            versions[pkg] = "not_installed"
    return versions


# ==============================================================
# 实验目录管理
# ==============================================================


def create_experiment_dir(output_dir: str, prefix: str = "exp") -> Path:
    """创建实验输出目录，自动编号。"""
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    existing = [d for d in base.iterdir() if d.is_dir() and d.name.startswith(f"{prefix}_")]
    next_id = len(existing) + 1
    exp_dir = base / f"{prefix}_{next_id:03d}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def create_manifest(
    exp_dir: Path,
    *,
    config: dict,
    data_hash: str,
    seed: int,
    cli_args: Optional[dict] = None,
) -> dict:
    """创建并保存实验 manifest。"""
    manifest = {
        "created_at": datetime.now().isoformat(),
        "git_commit": get_git_commit_hash(),
        "data_hash": data_hash,
        "seed": seed,
        "packages": get_package_versions(["numpy", "pandas", "scikit-learn", "torch", "xgboost"]),
        "config": config,
        "cli_args": cli_args or {},
    }
    manifest_path = exp_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
    return manifest


def save_config(exp_dir: Path, config: dict) -> None:
    """保存运行配置到 JSON 文件。"""
    config_path = exp_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, default=str)
