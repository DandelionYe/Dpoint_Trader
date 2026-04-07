# main_cli.py
"""
A-Share Dpoint ML Trading System - Basket Mode CLI
===================================================

A 股量化交易研究框架 - 篮子模式（多股票组合）命令行接口

本工具专为多股票篮子（Basket）模式设计，支持：
    - 面板数据训练（Panel Data Training）
    - 组合回测（Portfolio Backtesting）
    - 横截面排名特征（Cross-Sectional Ranking Features）
    - 滚动再训练（Rolling Retraining）

使用示例：
    # 基本用法 - 使用 data/basket_1 目录
    python main_cli.py --basket basket_1 --runs 50 --seed 42

    # 指定 basket 完整路径
    python main_cli.py --basket /path/to/basket --portfolio_cash 2000000

    # 调整组合参数
    python main_cli.py --basket basket_1 --top_k 10 --rebalance_freq monthly

    # 滚动再训练模式
    python main_cli.py --basket basket_1 --rolling_mode expanding --retrain_frequency monthly

数据格式要求：
    basket 目录内每个 CSV 文件命名格式：{股票代码}_{上市日期 YYYYMMDD}.csv
    示例：300299_20120319.csv, 002555_20110302.csv

    CSV 列名要求（自动映射）：
        Date, Open (CNY, qfq), High (CNY, qfq), Low (CNY, qfq), Close (CNY, qfq), Volume (shares)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import subprocess
from datetime import datetime
from typing import Dict, Optional, List, Any, Tuple

# 配置根 logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =========================================================
# Conda 环境管理
# =========================================================

def is_in_conda_env(target_env: str) -> bool:
    """Check if currently running inside the specified conda environment."""
    return os.environ.get("CONDA_DEFAULT_ENV") == target_env


def warn_if_env_mismatch(target_env: str) -> None:
    """Print a warning if not running in the target conda environment."""
    if os.environ.get("CI", "").lower() == "true":
        return
    if os.environ.get("SKIP_CONDA") == "1":
        return
    if is_in_conda_env(target_env):
        return

    print(
        f"[WARNING] Current conda env is not '{target_env}'. "
        f"For reproducible runs, activate it manually first, or run with "
        f"'--use-conda-env {target_env}'."
    )


def relaunch_in_conda(target_env: str) -> bool:
    """Relaunch the script inside the specified conda environment."""
    if os.environ.get("_ASHARE_RELAUNCHED") == "1":
        return False
    if is_in_conda_env(target_env):
        return False

    import shutil
    if shutil.which("conda") is None:
        print(f"[ERROR] conda not found in PATH, cannot relaunch into '{target_env}'.")
        sys.exit(2)

    child_env = {**os.environ, "_ASHARE_RELAUNCHED": "1"}
    cmd = ["conda", "run", "--no-capture-output", "-n", target_env, "python"] + sys.argv

    print(f"[INFO] Relaunching inside conda env '{target_env}'...")
    subprocess.run(cmd, env=child_env, check=True)
    return True


def _handle_conda_env(args) -> None:
    """Handle conda environment switching based on CLI arguments."""
    if args.use_conda_env:
        if relaunch_in_conda(args.use_conda_env):
            sys.exit(0)
    else:
        warn_if_env_mismatch(args.target_conda_env)


# =========================================================
# 导入核心模块
# =========================================================

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import pandas as pd

from utils import (
    set_global_seed, get_git_commit_hash, get_package_versions,
    compute_data_hash, export_environment_lock,
    _get_next_experiment_id, create_experiment_dir, create_manifest, create_config_json,
    load_manifest, find_latest_experiment, replay_from_manifest,
)
from data_loader import recommend_n_folds, load_basket, build_panel_dataframe
from trainer import (
    # 面板训练入口（篮子模式专用）
    random_search_train_panel, train_final_model_panel,
)
from backtester import calculate_risk_metrics, format_metrics_summary
from reporter import save_run_outputs, find_latest_run
from portfolio_backtester import (
    PortfolioConfig,
    run_portfolio_backtest,
    format_portfolio_summary,
)
from constants import DATA_ROOT_DIR, BASKET_DIR_PREFIX


# =========================================================
# 辅助函数
# =========================================================

def _load_previous_best(output_dir: str) -> Optional[Dict[str, Any]]:
    """加载历史最优配置作为 incumbent."""
    from reporter import find_latest_run
    latest = find_latest_run(output_dir)
    if latest is None:
        return None
    _, cfg_path, _ = latest
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        return blob.get("best_config")
    except Exception:
        return None


def _resolve_basket_dir(basket_arg: str, data_dir: str) -> str:
    """
    将 --basket 参数解析为完整的 basket 目录路径。

    解析规则（优先级从高到低）：
        1. 绝对路径      → 直接使用
        2. 含路径分隔符  → 相对于 CWD 使用
        3. 纯目录名      → 拼接 data_dir（如 "basket_1" → "data/basket_1"）
    """
    if os.path.isabs(basket_arg):
        return basket_arg
    if os.sep in basket_arg or "/" in basket_arg:
        return basket_arg
    return os.path.join(data_dir, basket_arg)


def _resolve_n_jobs(n_jobs_arg: int) -> int:
    """
    根据 CUDA 可用性自动决定并行度。

    规则：
        --n_jobs -1（默认，自动）：
            - CUDA 可用 → 强制 1（joblib fork 与 CUDA 不兼容）
            - CUDA 不可用 → 4（保守默认，sklearn 模型可安全并行）
        --n_jobs N（N != -1，用户显式指定）：
            - 直接使用 N，但若 CUDA 可用且 N > 1，打印警告提示潜在风险
    """
    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except Exception:
        pass

    if n_jobs_arg == -1:
        if cuda_available:
            resolved = 1
            print(
                "[INFO] CUDA 可用，n_jobs 自动设置为 1，"
                "避免 joblib fork 进程与 CUDA 上下文冲突。"
            )
        else:
            resolved = 4
            print("[INFO] 未检测到 CUDA，n_jobs 自动设置为 4（CPU 并行）。")
    else:
        resolved = n_jobs_arg
        if cuda_available and n_jobs_arg > 1:
            print(
                f"[WARN] CUDA 可用但 --n_jobs={n_jobs_arg} > 1。"
                f"若搜索空间包含 DL 模型（MLP/LSTM/GRU/CNN/Transformer），"
                f"joblib fork 可能导致 CUDA 上下文冲突。"
                f"建议使用 --n_jobs 1 或让系统自动决定（--n_jobs -1）。"
            )

    return resolved


# =========================================================
# 篮子模式主流程
# =========================================================

def _run_basket_mode(args) -> None:
    """
    篮子模式完整流程（多股票组合训练 + 回测）。

    流程：
        1. 加载 basket（load_basket）
        2. 面板随机搜索训练（random_search_train_panel）
        3. 全量面板模型训练（train_final_model_panel）
        4. 组合回测（run_portfolio_backtest）
        5. 构建 log_notes / 保存输出
        6. 打印组合摘要
    """
    # ── 1. 解析 basket 路径 ────────────────────────────────
    basket_dir = _resolve_basket_dir(args.basket, args.data_dir)
    print(f"[INFO] 加载 basket 目录：{basket_dir}")

    if not os.path.isdir(basket_dir):
        print(f"[ERROR] basket 目录不存在：{basket_dir}")
        print(
            "请确认路径正确。示例："
            " --basket basket_1 (在 data/ 目录下查找)"
            " --basket data/basket_1 (相对路径)"
            " --basket /absolute/path/basket (绝对路径)"
        )
        sys.exit(1)

    stock_dict, basket_report = load_basket(basket_dir)
    basket_report.summary()

    if not stock_dict:
        print("[ERROR] basket 内没有有效股票数据，请检查 CSV 文件格式。")
        sys.exit(1)

    # ── 2. 随机种子 & 并行度 ───────────────────────────────
    seed_info = set_global_seed(args.seed)
    print(f"[INFO] Global seed set to {args.seed}: {seed_info}")

    n_jobs_effective = _resolve_n_jobs(args.n_jobs)
    print(f"[INFO] 实际使用 n_jobs={n_jobs_effective}")

    # ── 3. 构建面板 DataFrame（用于数据哈希 & 报告）─────────
    panel_df = build_panel_dataframe(stock_dict)
    print(
        f"[INFO] 面板数据：{len(stock_dict)} 只股票，{len(panel_df)} 行，"
        f"日期范围 {panel_df['date'].min().date()} ~ {panel_df['date'].max().date()}"
    )

    # ── 4. 面板随机搜索训练 ────────────────────────────────
    search_initial_params: Dict[str, Any] = {
        "initial_cash": float(args.portfolio_cash),
    }

    base_best_config: Optional[Dict[str, Any]] = None
    if args.mode == "continue":
        base_best_config = _load_previous_best(args.output_dir)
        if base_best_config is None:
            print("[WARN] Continue 模式但未找到历史配置，回退到 first 模式行为。")
        else:
            print("[INFO] 加载历史最优配置作为 incumbent。")

    # n_folds 自动推算
    if args.n_folds == -1:
        effective_samples = int(len(panel_df) * 0.88)
        n_folds_effective = recommend_n_folds(
            n_samples=effective_samples,
            target_trades_per_fold=4,
            assumed_trade_freq=1.0 / 15.0,
            min_rows=60,
            min_folds=2,
            max_folds=8,
        )
        print(
            f"[INFO] n_folds 自动推算 = {n_folds_effective} "
            f"(面板行数={len(panel_df)}, 收缩后={effective_samples})"
        )
    else:
        n_folds_effective = max(2, args.n_folds)
        print(f"[INFO] 使用用户指定 n_folds={n_folds_effective}")

    print(f"[INFO] 开始面板随机搜索训练（runs={args.runs}）...")
    train_res = random_search_train_panel(
        stock_dict=stock_dict,
        runs=int(args.runs),
        seed=int(args.seed),
        base_best_config=base_best_config,
        trade_params=search_initial_params,
        max_features=80,
        output_dir=str(args.output_dir),
        epsilon=0.01,
        exploit_ratio=0.7,
        top_k=10,
        n_folds=n_folds_effective,
        train_start_ratio=0.5,
        wf_min_dates=40,
        n_jobs=n_jobs_effective,
        n_rounds=4,
        use_holdout=bool(args.use_holdout),
        holdout_ratio=float(args.holdout_ratio),
        use_embargo=bool(args.use_embargo),
        embargo_dates=int(args.embargo_days),
    )

    best_config = train_res.best_config
    print(f"[INFO] 最优验证指标 (几何均值 ratio): {train_res.best_val_metric:.6f}")
    print(f"[INFO] 最优验证 equity proxy:      {train_res.best_val_final_equity_proxy:.2f}")

    # ── 5. 全量面板模型训练 → dpoint_matrix ───────────────
    print("[INFO] 在全量面板数据上训练最终模型...")
    dpoint_matrix, artifacts = train_final_model_panel(
        stock_dict, best_config, seed=int(args.seed)
    )
    print(f"[INFO] dpoint_matrix: {len(dpoint_matrix)} 只股票")
    for code, dp in dpoint_matrix.items():
        print(f"  {code}: {len(dp)} 个 dpoint，范围 [{dp.min():.3f}, {dp.max():.3f}]")

    # ── 6. 组合回测 ────────────────────────────────────────
    portfolio_cfg = PortfolioConfig(
        top_k=int(args.top_k),
        rebalance_freq=str(args.rebalance_freq),
        weighting_scheme=str(args.weighting),
        initial_cash=float(args.portfolio_cash),
        dpoint_buy_threshold=float(args.dpoint_threshold),
    )
    print(
        f"[INFO] 组合回测：top_k={portfolio_cfg.top_k}, "
        f"freq={portfolio_cfg.rebalance_freq}, "
        f"scheme={portfolio_cfg.weighting_scheme}, "
        f"cash={portfolio_cfg.initial_cash:,.0f}"
    )
    portfolio_result = run_portfolio_backtest(
        stock_dict=stock_dict,
        dpoint_matrix=dpoint_matrix,
        cfg=portfolio_cfg,
        compute_attribution=True,
    )

    final_equity = (
        float(portfolio_result.equity_curve["total_equity"].iloc[-1])
        if not portfolio_result.equity_curve.empty
        else float(args.portfolio_cash)
    )
    print(f"[INFO] 组合最终净值：{final_equity:,.2f}")
    print(f"[INFO] 总交易笔数：  {len(portfolio_result.trades)}")

    # ── 7. 构建 log_notes ─────────────────────────────────
    log_notes: List[str] = []

    # Basket 加载摘要
    log_notes.append("=== Basket Load Report ===")
    log_notes.append(f"Basket 目录：{basket_dir}")
    log_notes.append(f"CSV 文件总数：{basket_report.total_files}")
    log_notes.append(f"成功加载：{basket_report.loaded_ok} 只，失败：{basket_report.loaded_failed} 只")
    if basket_report.failed_codes:
        log_notes.append(f"失败股票：{basket_report.failed_codes}")
    if basket_report.date_range_min and basket_report.date_range_max:
        log_notes.append(
            f"日期范围：{basket_report.date_range_min.date()} ~ {basket_report.date_range_max.date()}"
        )
    log_notes.append(f"共同交易日数：{basket_report.common_date_count}")
    for note in basket_report.notes:
        log_notes.append(f"  [NOTE] {note}")

    # 训练摘要
    log_notes.append("")
    log_notes.append("=== Panel Training Summary ===")
    log_notes.append(f"训练模式：panel_pool（共享模型）")
    log_notes.append(f"股票数：{len(stock_dict)}")
    log_notes.append(f"总样本行数：{len(panel_df)}")
    log_notes.append(f"搜索轮次：{args.runs}")
    log_notes.append(f"n_folds: {n_folds_effective}")
    log_notes.append(f"随机种子：{args.seed}")
    log_notes.append(f"最优验证指标：{train_res.best_val_metric:.6f}")
    log_notes.append(f"最优验证 equity proxy: {train_res.best_val_final_equity_proxy:.2f}")
    log_notes.append(f"Global best updated: {train_res.global_best_updated}")
    log_notes.extend(train_res.training_notes)

    # Holdout 结果
    if train_res.holdout_metric is not None:
        log_notes.append("")
        log_notes.append("=== Holdout Test Result ===")
        log_notes.append(f"Search rows: {train_res.search_data_rows}")
        log_notes.append(f"Holdout rows: {train_res.holdout_data_rows}")
        log_notes.append(f"Holdout OOS metric: {train_res.holdout_metric:.6f}")
        log_notes.append(f"Holdout equity: {train_res.holdout_equity:.2f}")

    # dpoint_matrix 摘要
    log_notes.append("")
    log_notes.append("=== dpoint_matrix Summary ===")
    log_notes.append(f"训练模式：{artifacts.get('training_mode', 'panel_pool')}")
    log_notes.append(f"模型类型：{artifacts['model']['type']}")
    log_notes.append(f"特征数：{artifacts['feature_meta']['n_features']}")
    log_notes.append(
        "⚠️  WARNING: dpoint 为全样本训练后的样本内预测，早期数据存在前向偏差。"
        "真实样本外表现请查看 SearchLog 中的 walk-forward 验证指标。"
    )
    for code, dp in dpoint_matrix.items():
        log_notes.append(
            f"  {code}: {len(dp)} rows, [{dp.min():.3f}, {dp.max():.3f}], "
            f"{dp.index.min().date()} ~ {dp.index.max().date()}"
        )

    # 组合回测结果
    log_notes.append("")
    log_notes.append("=== Portfolio Backtest Result ===")
    log_notes.extend(portfolio_result.notes)

    # 组合风险指标
    m = portfolio_result.metrics
    if m:
        log_notes.append("")
        log_notes.append("=== Portfolio Risk Metrics ===")
        log_notes.append(f"Total Return    : {m.get('total_return_pct', 0):+.2f}%")
        log_notes.append(f"Annual Return   : {m.get('annual_return_pct', 0):+.2f}%")
        log_notes.append(f"Annual Vol      : {m.get('annual_vol_pct', 0):.2f}%")
        log_notes.append(f"Sharpe          : {m.get('sharpe', 0):.3f}")
        log_notes.append(f"Max Drawdown    : {m.get('max_drawdown_pct', 0):.2f}%")
        log_notes.append(f"Calmar          : {m.get('calmar', 0):.3f}")

    # 归因分析
    if not portfolio_result.attribution.empty:
        log_notes.append("")
        log_notes.append("=== Attribution Analysis ===")
        for _, row in portfolio_result.attribution.iterrows():
            log_notes.append(
                f"  {row['stock_code']}: pnl={row['total_pnl']:,.0f}, "
                f"win_rate={row['win_rate']:.0%}, "
                f"contribution={row['contribution_pct']:+.1f}%"
            )

    log_notes.append("")
    log_notes.append("=== Backtest Notes (IN-SAMPLE) ===")
    log_notes.append(
        "⚠️  WARNING: The portfolio equity curve uses in-sample dpoint predictions. "
        "It WILL overstate real trading performance. "
        "See SearchLog for out-of-sample walk-forward metrics."
    )

    # ── 8. 创建实验目录 & 保存输出 ─────────────────────────
    timestamp = datetime.now().isoformat(timespec="seconds")
    git_commit_hash = get_git_commit_hash()
    package_versions = get_package_versions()
    data_hash = compute_data_hash(panel_df)

    if args.experiment_dir:
        experiment_dir = args.experiment_dir
        os.makedirs(experiment_dir, exist_ok=True)
        experiment_id = _get_next_experiment_id(args.output_dir)
    else:
        experiment_id = _get_next_experiment_id(args.output_dir)
        experiment_dir = create_experiment_dir(args.output_dir, experiment_id)

    cli_args = {
        "mode": args.mode,
        "basket": args.basket,
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "runs": args.runs,
        "seed": args.seed,
        "portfolio_cash": args.portfolio_cash,
        "top_k": args.top_k,
        "rebalance_freq": args.rebalance_freq,
        "weighting": args.weighting,
        "dpoint_threshold": args.dpoint_threshold,
        "n_folds": args.n_folds,
        "use_holdout": args.use_holdout,
        "holdout_ratio": args.holdout_ratio,
        "use_embargo": args.use_embargo,
        "embargo_days": args.embargo_days,
    }

    data_info = {
        "basket_dir": basket_dir,
        "data_hash": data_hash,
        "n_stocks": len(stock_dict),
        "n_rows": len(panel_df),
        "stock_codes": sorted(stock_dict.keys()),
    }

    metrics_summary: Dict[str, Any] = {
        "best_val_metric": train_res.best_val_metric,
        "best_val_final_equity_proxy": train_res.best_val_final_equity_proxy,
        "final_equity": final_equity,
        "n_trades": len(portfolio_result.trades),
    }
    if train_res.holdout_metric is not None:
        metrics_summary["holdout_metric"] = train_res.holdout_metric
        metrics_summary["holdout_equity"] = train_res.holdout_equity

    manifest = create_manifest(
        experiment_dir=experiment_dir,
        run_id=experiment_id,
        timestamp=timestamp,
        git_commit_hash=git_commit_hash,
        package_versions=package_versions,
        seed=args.seed,
        data_info=data_info,
        cli_args=cli_args,
        best_config=best_config,
        metrics=metrics_summary,
    )

    create_config_json(experiment_dir, manifest)

    # 保存 Excel 和 HTML 报告
    # 注意：artifacts["feature_meta"] 可能没有 params 键（面板模式简化了）
    feature_meta_for_save = {
        "feature_names": artifacts["feature_meta"]["feature_names"],
        "n_features": artifacts["feature_meta"]["n_features"],
    }
    if "params" in artifacts["feature_meta"]:
        feature_meta_for_save["params"] = artifacts["feature_meta"]["params"]

    excel_path, config_path, run_id = save_run_outputs(
        output_dir=experiment_dir,
        log_notes=log_notes,
        trades=portfolio_result.trades,
        equity_curve=portfolio_result.equity_curve,
        config=best_config,
        feature_meta=feature_meta_for_save,
        search_log=None,  # 面板模式暂不提供 search_log
        portfolio_result=portfolio_result,  # P-basket: 传入组合回测结果
    )

    print(f"\n[INFO] Run completed and saved:")
    print(f"  Excel: {excel_path}")
    print(f"  Config: {config_path}")
    print(f"  Manifest: {experiment_dir}/manifest.json")

    # 打印组合摘要
    print("\n" + format_portfolio_summary(portfolio_result))


# =========================================================
# CLI 参数定义
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="A-share basket ML Dpoint trader (Basket Mode Only)."
    )

    # 基本参数
    parser.add_argument(
        "--mode", type=str, default="first",
        choices=["first", "continue"],
        help="first = start new search; continue = load previous best_config as incumbent."
    )
    parser.add_argument(
        "--runs", type=int, default=50,
        help="Number of random search iterations (default: 50)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)."
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output",
        help="Output directory for experiment results (default: ./output)."
    )
    parser.add_argument(
        "--experiment_dir", type=str, default="",
        help="Optional: specify experiment directory name directly."
    )
    parser.add_argument(
        "--n_jobs", type=int, default=-1,
        help="Parallel jobs. -1 (default) = auto; >0 = explicit count."
    )
    parser.add_argument(
        "--n_folds", type=int, default=-1,
        help="Walk-forward folds. -1 (default) = auto; >0 = explicit count."
    )

    # Basket 模式参数
    parser.add_argument(
        "--basket",
        type=str,
        required=True,
        help=(
            "Basket 子目录名（如 'basket_1'）或完整路径。"
            "目录内每个 {code}_{YYYYMMDD}.csv 文件被视为一只股票。"
            "此参数为必需参数。"
        ),
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DATA_ROOT_DIR,
        help=f"Basket 根目录（默认 '{DATA_ROOT_DIR}'）。"
    )
    parser.add_argument(
        "--portfolio_cash",
        type=float,
        default=1_000_000.0,
        help="组合回测初始资金（元），默认 1_000_000。"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="每期最大持仓股票数，默认 5。"
    )
    parser.add_argument(
        "--rebalance_freq",
        type=str,
        default="weekly",
        choices=["daily", "weekly", "monthly"],
        help="调仓频率（daily | weekly | monthly），默认 weekly。"
    )
    parser.add_argument(
        "--weighting",
        type=str,
        default="equal",
        choices=["equal", "signal"],
        help="持仓权重方案（equal | signal），默认 equal。"
    )
    parser.add_argument(
        "--dpoint_threshold",
        type=float,
        default=0.5,
        help="选股 dpoint 最低门槛，低于此值不纳入候选，默认 0.5。"
    )

    # Holdout 参数
    parser.add_argument(
        "--use_holdout", type=int, default=1,
        help="Use final holdout test. 1 (default) = enable, 0 = disable."
    )
    parser.add_argument(
        "--holdout_ratio", type=float, default=0.15,
        help="Holdout ratio (default 0.15 = 15%%)."
    )

    # Embargo 参数
    parser.add_argument(
        "--use_embargo", type=int, default=0,
        help="Use embargo gap to prevent temporal leakage. 1 = enable, 0 = disable (default)."
    )
    parser.add_argument(
        "--embargo_days", type=int, default=5,
        help="Embargo days (default 5). Only used when --use_embargo=1."
    )

    # Conda 环境参数
    parser.add_argument(
        "--use-conda-env", type=str, default="",
        help="Relaunch into specified conda environment."
    )
    parser.add_argument(
        "--target-conda-env", type=str, default="ashare_dpoint",
        help="Expected conda environment name for warning messages."
    )

    # 滚动再训练参数
    parser.add_argument(
        "--rolling_mode", type=str, default="",
        help="Rolling retrain mode. Empty (default) = normal training; Supports: expanding, rolling."
    )
    parser.add_argument(
        "--rolling_window_length", type=int, default=None,
        help="Rolling window length (days), only used when rolling_mode=rolling."
    )
    parser.add_argument(
        "--retrain_frequency", type=str, default="monthly",
        help="Retrain frequency. Default monthly, supports: daily, weekly, monthly, quarterly."
    )
    parser.add_argument(
        "--retrain_eval_days", type=int, default=30,
        help="Days to evaluate after retrain."
    )
    parser.add_argument(
        "--snapshot_max_keep", type=int, default=5,
        help="Max number of model snapshots to keep."
    )

    # Replay 参数
    parser.add_argument(
        "--replay", type=str, default="",
        help="Replay from historical experiment. 'latest' for most recent, or specify directory."
    )
    parser.add_argument(
        "--export_lock", type=str, default="",
        help="Export environment lock file to specified path."
    )

    args = parser.parse_args()

    # 处理 conda 环境切换
    _handle_conda_env(args)

    # 执行篮子模式
    _run_basket_mode(args)


if __name__ == "__main__":
    main()
