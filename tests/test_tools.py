# test_tools.py
"""工具模块测试。"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_compare_runs(tmp_path):
    from dpoint.tools.compare_runs import compare_runs

    # 创建两个模拟实验
    for i in range(2):
        exp_dir = tmp_path / f"exp_{i+1:03d}"
        exp_dir.mkdir()
        manifest = {
            "seed": 42 + i,
            "git_commit": "abc123",
            "created_at": "2026-01-01",
        }
        with open(exp_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)

        config = {"model": {"model_type": "logreg"}, "search": {"metric": "pnl"}}
        with open(exp_dir / "config.json", "w") as f:
            json.dump(config, f)

        # 创建模拟 Excel 报告
        metrics_df = pd.DataFrame(
            [
                {
                    "total_return": 0.1 + i * 0.05,
                    "sharpe": 1.2 + i * 0.3,
                    "max_drawdown": -0.05 - i * 0.01,
                }
            ]
        )
        metrics_df.to_excel(exp_dir / "report.xlsx", sheet_name="RiskMetrics", index=False)

    df = compare_runs([tmp_path / "exp_001", tmp_path / "exp_002"])
    assert len(df) == 2
    assert "total_return" in df.columns
    assert "sharpe" in df.columns


def test_find_experiments(tmp_path):
    from dpoint.tools.compare_runs import find_experiments

    for i in range(3):
        exp_dir = tmp_path / f"single_{i+1:03d}"
        exp_dir.mkdir()
        with open(exp_dir / "config.json", "w") as f:
            json.dump({}, f)

    exps = find_experiments(tmp_path, prefix="single_")
    assert len(exps) == 3


def test_load_run_manifest(tmp_path):
    from dpoint.tools.compare_runs import load_run_manifest

    exp_dir = tmp_path / "exp_001"
    exp_dir.mkdir()
    manifest = {"seed": 42, "git_commit": "abc"}
    with open(exp_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)

    result = load_run_manifest(exp_dir)
    assert result["seed"] == 42
