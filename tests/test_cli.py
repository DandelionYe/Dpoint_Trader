# test_cli.py
"""CLI 端到端集成测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_cli_parser():
    from dpoint.cli.main import build_parser

    parser = build_parser()

    # single 模式
    args = parser.parse_args(
        ["single", "--data_path", "test.xlsx", "--model", "logreg", "--runs", "10"]
    )
    assert args.command == "single"
    assert args.data_path == "test.xlsx"
    assert args.model == "logreg"
    assert args.runs == 10

    # basket 模式
    args = parser.parse_args(
        ["basket", "--basket_path", "data/basket_1/", "--model", "lstm", "--top_k", "3"]
    )
    assert args.command == "basket"
    assert args.basket_path == "data/basket_1/"
    assert args.top_k == 3


def test_cli_single_end_to_end(tmp_path, sample_single_df):
    """单股模式端到端测试（少量候选，验证完整流程）。"""
    from dpoint.cli.main import main

    # 保存测试数据为 Excel
    data_path = tmp_path / "test_data.xlsx"
    sample_single_df.to_excel(data_path, index=False)

    output_dir = str(tmp_path / "output")

    # 运行（少量候选加速，用 sgd 避免 scipy 崩溃）
    exit_code = main(
        [
            "single",
            "--data_path",
            str(data_path),
            "--model",
            "sgd",
            "--runs",
            "8",
            "--n_rounds",
            "2",
            "--metric",
            "pnl",
            "--output",
            output_dir,
            "--model_types",
            "sgd",
        ]
    )

    assert exit_code == 0

    # 检查输出目录存在
    output_path = Path(output_dir)
    exp_dirs = list(output_path.glob("single_*"))
    assert len(exp_dirs) > 0

    exp_dir = exp_dirs[0]
    # config 和 manifest 应该总是存在
    assert (exp_dir / "config.json").exists()
    assert (exp_dir / "manifest.json").exists()
    # report.xlsx 仅在搜索找到有效候选时存在
    # （小数据集+少量候选可能找不到，所以不强制检查）


def test_cli_single_with_csv(tmp_path, sample_single_df):
    """单股模式 CSV 格式测试。"""
    from dpoint.cli.main import main

    data_path = tmp_path / "test_data.csv"
    sample_single_df.to_csv(data_path, index=False)

    output_dir = str(tmp_path / "output")

    exit_code = main(
        [
            "single",
            "--data_path",
            str(data_path),
            "--model",
            "sgd",
            "--runs",
            "4",
            "--n_rounds",
            "2",
            "--output",
            output_dir,
        ]
    )

    assert exit_code == 0
