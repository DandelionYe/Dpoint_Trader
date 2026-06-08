# main.py
"""
统一 CLI 入口：dpoint single / dpoint basket 子命令。
Phase 5：完整端到端流程。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from dpoint.core.config import (
    FeatureConfig, ModelConfig, PortfolioConfig, RunConfig,
    SearchConfig, SplitConfig, TradeConfig,
)
from dpoint.core.utils import (
    compute_data_hash, create_experiment_dir, create_manifest,
    save_config, set_global_seed,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dpoint",
        description="Dpoint_Trader — A股深度学习量化交易研究框架",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细日志")

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # === dpoint single ===
    single = subparsers.add_parser("single", help="单股策略模式")
    single.add_argument("--data_path", required=True, help="Excel/CSV 数据文件路径")
    single.add_argument("--model", default="logreg",
                        choices=["logreg", "sgd", "xgb", "mlp", "lstm", "gru", "cnn", "transformer"],
                        help="模型类型")
    single.add_argument("--runs", type=int, default=100, help="搜索候选数")
    single.add_argument("--n_rounds", type=int, default=4, help="搜索轮数")
    single.add_argument("--metric", default="pnl", choices=["pnl", "rank_ic"], help="搜索目标函数")
    single.add_argument("--seed", type=int, default=42, help="随机种子")
    single.add_argument("--output", default="output", help="输出目录")
    single.add_argument("--device", default="auto", help="设备 (auto/cpu/cuda)")
    single.add_argument("--n_folds", type=int, default=4, help="Walk-Forward 折数")
    single.add_argument("--holdout_ratio", type=float, default=0.15, help="Holdout 比例")
    single.add_argument("--model_types", nargs="+", default=None,
                        help="搜索的模型类型列表（默认全部）")
    single.add_argument("--config", default=None,
                        help="JSON 配置文件路径（提供后忽略其他参数，直接使用文件中的完整配置）")

    # === dpoint basket ===
    basket = subparsers.add_parser("basket", help="篮子/组合策略模式")
    basket.add_argument("--basket_path", required=True, help="篮子数据目录路径")
    basket.add_argument("--model", default="logreg",
                        choices=["logreg", "sgd", "xgb", "mlp", "lstm", "gru", "cnn", "transformer"],
                        help="模型类型")
    basket.add_argument("--runs", type=int, default=100, help="搜索候选数")
    basket.add_argument("--n_rounds", type=int, default=4, help="搜索轮数")
    basket.add_argument("--metric", default="rank_ic", choices=["pnl", "rank_ic"], help="搜索目标函数")
    basket.add_argument("--seed", type=int, default=42, help="随机种子")
    basket.add_argument("--output", default="output", help="输出目录")
    basket.add_argument("--device", default="auto", help="设备")
    basket.add_argument("--top_k", type=int, default=5, help="组合 Top-K 选股数")
    basket.add_argument("--rebalance", default="daily", choices=["daily", "weekly", "monthly"], help="调仓频率")
    basket.add_argument("--weighting", default="equal", choices=["equal", "score", "vol_inv"], help="权重方式")
    basket.add_argument("--n_folds", type=int, default=4, help="Walk-Forward 折数")
    basket.add_argument("--holdout_ratio", type=float, default=0.15, help="Holdout 比例")
    basket.add_argument("--calendar_align", default="none",
                        choices=["none", "inner", "outer", "majority"], help="日历对齐")
    basket.add_argument("--model_types", nargs="+", default=None, help="搜索的模型类型列表")
    basket.add_argument("--config", default=None, help="JSON 配置文件路径")

    # === dpoint resume ===
    resume = subparsers.add_parser("resume", help="从上次搜索结果继续迭代搜索")
    resume.add_argument("experiment_dir", help="上次实验目录路径（如 output/single_002）")
    resume.add_argument("--runs", type=int, default=100, help="本轮搜索候选数")
    resume.add_argument("--n_rounds", type=int, default=4, help="本轮搜索轮数")
    resume.add_argument("--metric", default=None, choices=["pnl", "rank_ic"],
                        help="搜索目标函数（默认沿用上次配置）")
    resume.add_argument("--seed", type=int, default=None,
                        help="新随机种子（默认沿用上次 RNG 状态）")
    resume.add_argument("--output", default="output", help="输出目录")
    resume.add_argument("--device", default="auto", help="设备 (auto/cpu/cuda)")
    resume.add_argument("--model_types", nargs="+", default=None, help="搜索的模型类型列表")
    resume.add_argument("--config", default=None, help="JSON 配置文件路径")

    # === dpoint fetch ===
    fetch = subparsers.add_parser("fetch", help="自动获取价格数据（需要 XtMiniQMT 运行）")
    fetch_sub = fetch.add_subparsers(dest="fetch_mode", help="获取模式")

    # dpoint fetch single
    fetch_single = fetch_sub.add_parser("single", help="获取单只股票历史数据")
    fetch_single.add_argument("--code", required=True, help="股票代码，如 000001.SZ")
    fetch_single.add_argument("--start", default="", help="起始日期 YYYYMMDD（默认6年前）")
    fetch_single.add_argument("--end", default="", help="结束日期 YYYYMMDD（默认今天）")
    fetch_single.add_argument("--output", default="", help="输出文件路径")
    fetch_single.add_argument("--format", default="xlsx", choices=["xlsx", "csv"], help="输出格式")

    # dpoint fetch basket
    fetch_basket = fetch_sub.add_parser("basket", help="获取行业篮子数据")
    fetch_basket.add_argument("--industry", required=True, help="行业代码，如 C27")
    fetch_basket.add_argument("--start", default="", help="起始日期 YYYYMMDD（默认6年前）")
    fetch_basket.add_argument("--end", default="", help="结束日期 YYYYMMDD（默认今天）")
    fetch_basket.add_argument("--output", default="", help="输出目录路径")
    fetch_basket.add_argument("--format", default="csv", choices=["xlsx", "csv"], help="输出格式")
    fetch_basket.add_argument("--db", default="", help="行业分类 SQLite 路径")

    return parser


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run_single(args) -> int:
    """单股模式完整流程。"""
    logger = logging.getLogger("dpoint.single")

    # 1. 构建配置
    if args.config:
        logger.info("Loading config from: %s", args.config)
        config = RunConfig.from_dict(json.loads(Path(args.config).read_text(encoding="utf-8")))
        config.mode = "single"
    else:
        config = RunConfig(
            mode="single",
            data_path=args.data_path,
            output_dir=args.output,
            seed=args.seed,
            device=args.device,
            model=ModelConfig(model_type=args.model),
            search=SearchConfig(n_candidates=args.runs, n_rounds=args.n_rounds, metric=args.metric, seed=args.seed),
            split=SplitConfig(n_folds=args.n_folds, holdout_ratio=args.holdout_ratio),
        )

    set_global_seed(config.seed)

    # 2. 加载数据
    logger.info("Loading data: %s", args.data_path)
    data_path = Path(args.data_path)
    if data_path.suffix in (".xlsx", ".xls"):
        from dpoint.data.excel_loader import load_stock_excel
        df, report = load_stock_excel(data_path)
    elif data_path.suffix == ".csv":
        from dpoint.data.csv_loader import load_single_csv
        df, report = load_single_csv(data_path)
    else:
        logger.error("Unsupported file format: %s", data_path.suffix)
        return 1

    logger.info("Loaded %d rows, ticker=%s", report.rows_after_clean, report.ticker)

    # 确保有 date 列（不是索引）
    if df.index.name == "date":
        df = df.reset_index()

    # 3. 创建实验目录
    exp_dir = create_experiment_dir(config.output_dir, prefix="single")
    logger.info("Experiment directory: %s", exp_dir)

    # 4. 随机搜索
    logger.info("Starting search: %d candidates, %d rounds, metric=%s",
                config.search.n_candidates, config.search.n_rounds, config.search.metric)

    from dpoint.search.engine import create_evaluate_fn_single, random_search, save_search_state

    evaluate_fn = create_evaluate_fn_single(df, config)
    model_types = args.model_types or [args.model]

    rng = np.random.Generator(np.random.PCG64(config.search.seed))

    t0 = time.time()
    state, rng = random_search(config, evaluate_fn, model_types=model_types, rng=rng)
    elapsed = time.time() - t0

    # 保存搜索状态（供迭代训练使用）
    save_search_state(state, rng, exp_dir / "search_state.json")

    logger.info("Search completed in %.1fs: %d evaluated, best_score=%.4f",
                elapsed, state.n_evaluated, state.best_score)

    # 5. 用最优配置训练最终模型并回测
    if state.best_config:
        logger.info("Training final model with best config...")
        from dpoint.core.config import FeatureConfig
        from dpoint.features.pipeline import build_features_and_labels
        from dpoint.models.registry import make_model
        from dpoint.models.trainer import predict_pytorch_model, predict_sklearn_model, train_pytorch_model, train_sklearn_model
        from dpoint.backtester.single_stock import backtest_from_dpoint, compute_fold_metrics
        from dpoint.core.tasks import resolve_label_spec

        best = state.best_config
        feat_cfg = best.get("feature", {})
        feature_config = FeatureConfig(
            use_momentum=feat_cfg.get("use_momentum", True),
            use_volatility=feat_cfg.get("use_volatility", True),
            use_volume=feat_cfg.get("use_volume", True),
            use_candle=feat_cfg.get("use_candle", True),
            use_turnover=feat_cfg.get("use_turnover", True),
            use_ta_indicators=feat_cfg.get("use_ta_indicators", True),
        )

        df_feat, y, meta = build_features_and_labels(df.copy(), feature_config, mode="single")
        feature_names = meta.feature_names
        model_cfg = best.get("model", {})
        trade_cfg = best.get("trade", {})
        model_type = model_cfg.get("model_type", "logreg")

        X_all = df_feat[feature_names].values
        y_all = y.values
        X_all = __import__("numpy").nan_to_num(X_all, nan=0.0)

        label_spec = resolve_label_spec()
        model, kind = make_model(model_type, X_all.shape[1], model_cfg, label_spec)

        if kind == "sklearn":
            model = train_sklearn_model(model, X_all, y_all)
            proba = predict_sklearn_model(model, X_all)
        else:
            train_pytorch_model(model, X_all, y_all, epochs=model_cfg.get("epochs", 50),
                                batch_size=model_cfg.get("batch_size", 256),
                                learning_rate=model_cfg.get("learning_rate", 1e-3),
                                device=config.device)
            proba = predict_pytorch_model(model, X_all, device=config.device)

        dpoint = __import__("pandas").Series(proba, index=df_feat["date"].values)
        bt_result = backtest_from_dpoint(
            df_feat, dpoint,
            buy_threshold=trade_cfg.get("buy_threshold", 0.55),
            sell_threshold=trade_cfg.get("sell_threshold", 0.45),
            confirm_days=trade_cfg.get("confirm_days", 1),
            max_hold_days=trade_cfg.get("max_hold_days", 20),
        )

        logger.info("Final model: total_return=%.4f, sharpe=%.4f, max_dd=%.4f",
                     bt_result.risk_metrics.get("total_return", 0),
                     bt_result.risk_metrics.get("sharpe", 0),
                     bt_result.risk_metrics.get("max_drawdown", 0))

        # 6. 保存报告
        from dpoint.reports.excel_reporter import save_excel_report

        search_log = []
        for r in state.all_results:
            log_entry = {"score": r.score, "elapsed": r.elapsed_sec}
            log_entry.update(r.config.get("model", {}))
            log_entry.update(r.config.get("trade", {}))
            search_log.append(log_entry)

        report_path = save_excel_report(
            exp_dir / "report.xlsx",
            equity_curve=bt_result.equity_curve,
            trades=bt_result.trades,
            risk_metrics=bt_result.risk_metrics,
            config=config.to_dict(),
            search_log=search_log,
            notes=bt_result.notes,
        )
        logger.info("Report saved: %s", report_path)

        # 尝试保存 HTML 报告
        try:
            from dpoint.reports.html_reporter import save_html_report
            save_html_report(
                exp_dir / "report.html",
                equity_curve=bt_result.equity_curve,
                risk_metrics=bt_result.risk_metrics,
                trades=bt_result.trades,
                title=f"Dpoint Single-Stock Report: {report.ticker}",
            )
        except ImportError:
            pass

        # 保存 manifest
        save_config(exp_dir, config.to_dict())
        create_manifest(
            exp_dir,
            config=config.to_dict(),
            data_hash=compute_data_hash(df),
            seed=config.seed,
        )

        logger.info("All outputs saved to: %s", exp_dir)
    else:
        logger.warning("No valid candidates found during search")

    # 总是保存配置和 manifest
    save_config(exp_dir, config.to_dict())
    create_manifest(exp_dir, config=config.to_dict(), data_hash=compute_data_hash(df), seed=config.seed)
    logger.info("Outputs saved to: %s", exp_dir)

    return 0


def run_basket(args) -> int:
    """篮子模式完整流程。"""
    logger = logging.getLogger("dpoint.basket")

    # 1. 构建配置
    if args.config:
        logger.info("Loading config from: %s", args.config)
        config = RunConfig.from_dict(json.loads(Path(args.config).read_text(encoding="utf-8")))
        config.mode = "basket"
    else:
        config = RunConfig(
            mode="basket",
            basket_path=args.basket_path,
            output_dir=args.output,
            seed=args.seed,
            device=args.device,
            model=ModelConfig(model_type=args.model),
            search=SearchConfig(n_candidates=args.runs, n_rounds=args.n_rounds, metric=args.metric, seed=args.seed),
            split=SplitConfig(n_folds=args.n_folds, holdout_ratio=args.holdout_ratio),
            portfolio=PortfolioConfig(top_k=args.top_k, weighting=args.weighting, rebalance_freq=args.rebalance),
        )

    set_global_seed(config.seed)

    # 2. 加载篮子数据
    logger.info("Loading basket: %s", args.basket_path)
    from dpoint.data.basket_loader import load_basket_folder

    panel_df, basket_report = load_basket_folder(
        args.basket_path, calendar_align=args.calendar_align,
    )
    logger.info("Loaded basket: %d rows, %d tickers, %d dates",
                len(panel_df), basket_report.n_files_loaded, panel_df["date"].nunique())

    # 3. 创建实验目录
    exp_dir = create_experiment_dir(config.output_dir, prefix="basket")
    logger.info("Experiment directory: %s", exp_dir)

    # 4. 随机搜索
    from dpoint.search.engine import create_evaluate_fn_basket, random_search, save_search_state

    evaluate_fn = create_evaluate_fn_basket(panel_df, config)
    model_types = args.model_types or [args.model]

    rng = np.random.Generator(np.random.PCG64(config.search.seed))

    t0 = time.time()
    state, rng = random_search(config, evaluate_fn, model_types=model_types, rng=rng)
    elapsed = time.time() - t0

    # 保存搜索状态（供迭代训练使用）
    save_search_state(state, rng, exp_dir / "search_state.json")

    logger.info("Search completed in %.1fs: %d evaluated, best_score=%.4f",
                elapsed, state.n_evaluated, state.best_score)

    # 5. 用最优配置训练最终模型并计算因子指标
    if state.best_config:
        logger.info("Training final model with best config...")
        from dpoint.core.config import FeatureConfig
        from dpoint.features.pipeline import build_features_and_labels
        from dpoint.models.registry import make_model
        from dpoint.models.trainer import predict_pytorch_model, predict_sklearn_model, train_pytorch_model, train_sklearn_model
        from dpoint.reports.metrics import compute_ranking_metrics
        from dpoint.core.tasks import resolve_label_spec

        best = state.best_config
        feat_cfg = best.get("feature", {})
        feature_config = FeatureConfig(
            use_momentum=feat_cfg.get("use_momentum", True),
            use_volatility=feat_cfg.get("use_volatility", True),
            use_volume=feat_cfg.get("use_volume", True),
            use_candle=feat_cfg.get("use_candle", True),
            use_turnover=feat_cfg.get("use_turnover", True),
            use_ta_indicators=feat_cfg.get("use_ta_indicators", True),
        )

        df_feat, y, meta = build_features_and_labels(panel_df.copy(), feature_config, mode="basket")
        feature_names = meta.feature_names
        model_cfg = best.get("model", {})
        model_type = model_cfg.get("model_type", "logreg")

        X_all = df_feat[feature_names].values
        y_all = y.values
        X_all = __import__("numpy").nan_to_num(X_all, nan=0.0)

        label_spec = resolve_label_spec()
        model, kind = make_model(model_type, X_all.shape[1], model_cfg, label_spec)

        if kind == "sklearn":
            model = train_sklearn_model(model, X_all, y_all)
            proba = predict_sklearn_model(model, X_all)
        else:
            train_pytorch_model(model, X_all, y_all, epochs=model_cfg.get("epochs", 50),
                                batch_size=model_cfg.get("batch_size", 256),
                                learning_rate=model_cfg.get("learning_rate", 1e-3),
                                device=config.device)
            proba = predict_pytorch_model(model, X_all, device=config.device)

        # 计算因子指标
        df_feat["pred_score"] = proba
        df_feat["label"] = y.values
        ranking_metrics = compute_ranking_metrics(df_feat, score_col="pred_score", label_col="label")

        logger.info("Rank IC: mean=%.4f, std=%.4f, IR=%.4f",
                     ranking_metrics.rank_ic_mean or 0, ranking_metrics.rank_ic_std or 0, ranking_metrics.rank_ic_ir or 0)
        logger.info("Top-K return: mean=%.6f, annual=%.4f",
                     ranking_metrics.topk_return_mean or 0, ranking_metrics.topk_return_annual or 0)

        # 6. 保存报告
        from dpoint.reports.excel_reporter import save_excel_report

        search_log = []
        for r in state.all_results:
            log_entry = {"score": r.score, "elapsed": r.elapsed_sec}
            log_entry.update(r.config.get("model", {}))
            search_log.append(log_entry)

        rm_dict = {
            "ic_mean": ranking_metrics.ic_mean,
            "rank_ic_mean": ranking_metrics.rank_ic_mean,
            "rank_ic_ir": ranking_metrics.rank_ic_ir,
            "topk_return_mean": ranking_metrics.topk_return_mean,
        }
        if ranking_metrics.layered_returns:
            rm_dict.update(ranking_metrics.layered_returns)

        report_path = save_excel_report(
            exp_dir / "report.xlsx",
            ranking_metrics=rm_dict,
            config=config.to_dict(),
            search_log=search_log,
        )
        logger.info("Report saved: %s", report_path)

        # 保存 manifest
        save_config(exp_dir, config.to_dict())
        create_manifest(
            exp_dir,
            config=config.to_dict(),
            data_hash=compute_data_hash(panel_df),
            seed=config.seed,
        )

        logger.info("All outputs saved to: %s", exp_dir)
    else:
        logger.warning("No valid candidates found during search")

    return 0


def run_resume(args) -> int:
    """从上次搜索结果继续迭代搜索。"""
    logger = logging.getLogger("dpoint.resume")

    exp_dir = Path(args.experiment_dir)
    if not exp_dir.is_dir():
        logger.error("Experiment directory not found: %s", exp_dir)
        return 1

    # 1. 加载搜索状态
    state_path = exp_dir / "search_state.json"
    if not state_path.exists():
        logger.error("search_state.json not found in %s. Was this run created with the old version?", exp_dir)
        return 1

    from dpoint.search.engine import load_search_state
    state, rng_state = load_search_state(state_path)

    # 2. 加载原始配置
    config_path = exp_dir / "config.json"
    if not config_path.exists():
        logger.error("config.json not found in %s", exp_dir)
        return 1

    config = RunConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8")))

    # 3. CLI 参数覆盖
    config.search.n_candidates = args.runs
    config.search.n_rounds = args.n_rounds
    config.output_dir = args.output
    config.device = args.device
    if args.metric:
        config.search.metric = args.metric

    # 恢复或重建 RNG
    if args.seed is not None:
        # 用户指定新种子，忽略原 RNG 状态
        rng = np.random.Generator(np.random.PCG64(args.seed))
        logger.info("Using new seed: %d (ignoring previous RNG state)", args.seed)
    else:
        # 恢复原 RNG 状态（精确续跑）
        rng = np.random.Generator(np.random.PCG64())
        rng.bit_generator.state = rng_state
        logger.info("Restored previous RNG state")

    # 4. 加载数据
    data_path_str = config.data_path if config.mode == "single" else config.basket_path
    if not data_path_str:
        logger.error("No data_path/basket_path found in config.json")
        return 1

    data_path = Path(data_path_str)
    if not data_path.is_absolute():
        # 相对路径：相对于原实验目录的父目录（即项目根目录）
        data_path = exp_dir.parent / data_path
        if not data_path.exists():
            # 再试当前工作目录
            data_path = Path(data_path_str)

    if config.mode == "single":
        return _run_resume_single(config, state, rng, data_path, exp_dir, args, logger)
    else:
        return _run_resume_basket(config, state, rng, data_path, exp_dir, args, logger)


def _run_resume_single(config, state, rng, data_path, parent_dir, args, logger) -> int:
    """单股模式的 resume 流程。"""
    import numpy as _np
    import pandas as _pd

    logger.info("Resuming single-stock search from: %s", parent_dir)

    # 加载数据
    logger.info("Loading data: %s", data_path)
    if data_path.suffix in (".xlsx", ".xls"):
        from dpoint.data.excel_loader import load_stock_excel
        df, report = load_stock_excel(data_path)
    elif data_path.suffix == ".csv":
        from dpoint.data.csv_loader import load_single_csv
        df, report = load_single_csv(data_path)
    else:
        logger.error("Unsupported file format: %s", data_path.suffix)
        return 1

    logger.info("Loaded %d rows, ticker=%s", report.rows_after_clean, report.ticker)

    if df.index.name == "date":
        df = df.reset_index()

    # 创建新实验目录
    from dpoint.search.engine import create_evaluate_fn_single, random_search, save_search_state
    from dpoint.core.utils import create_experiment_dir

    exp_dir = create_experiment_dir(config.output_dir, prefix="single")
    logger.info("New experiment directory: %s (parent: %s)", exp_dir, parent_dir)

    # 搜索
    evaluate_fn = create_evaluate_fn_single(df, config)
    model_types = args.model_types or [config.model.model_type]

    t0 = time.time()
    state, rng = random_search(
        config, evaluate_fn,
        model_types=model_types,
        initial_state=state,
        rng=rng,
    )
    elapsed = time.time() - t0

    save_search_state(state, rng, exp_dir / "search_state.json")

    logger.info("Search completed in %.1fs: %d evaluated, best_score=%.4f",
                elapsed, state.n_evaluated, state.best_score)

    # 训练最终模型并回测
    if state.best_config:
        logger.info("Training final model with best config...")
        from dpoint.core.config import FeatureConfig
        from dpoint.features.pipeline import build_features_and_labels
        from dpoint.models.registry import make_model
        from dpoint.models.trainer import predict_pytorch_model, predict_sklearn_model, train_pytorch_model, train_sklearn_model
        from dpoint.backtester.single_stock import backtest_from_dpoint
        from dpoint.core.tasks import resolve_label_spec

        best = state.best_config
        feat_cfg = best.get("feature", {})
        feature_config = FeatureConfig(
            use_momentum=feat_cfg.get("use_momentum", True),
            use_volatility=feat_cfg.get("use_volatility", True),
            use_volume=feat_cfg.get("use_volume", True),
            use_candle=feat_cfg.get("use_candle", True),
            use_turnover=feat_cfg.get("use_turnover", True),
            use_ta_indicators=feat_cfg.get("use_ta_indicators", True),
        )

        df_feat, y, meta = build_features_and_labels(df.copy(), feature_config, mode="single")
        feature_names = meta.feature_names
        model_cfg = best.get("model", {})
        trade_cfg = best.get("trade", {})
        model_type = model_cfg.get("model_type", "logreg")

        X_all = df_feat[feature_names].values
        y_all = y.values
        X_all = _np.nan_to_num(X_all, nan=0.0)

        label_spec = resolve_label_spec()
        model, kind = make_model(model_type, X_all.shape[1], model_cfg, label_spec)

        if kind == "sklearn":
            model = train_sklearn_model(model, X_all, y_all)
            proba = predict_sklearn_model(model, X_all)
        else:
            train_pytorch_model(model, X_all, y_all, epochs=model_cfg.get("epochs", 50),
                                batch_size=model_cfg.get("batch_size", 256),
                                learning_rate=model_cfg.get("learning_rate", 1e-3),
                                device=config.device)
            proba = predict_pytorch_model(model, X_all, device=config.device)

        dpoint = _pd.Series(proba, index=df_feat["date"].values)
        bt_result = backtest_from_dpoint(
            df_feat, dpoint,
            buy_threshold=trade_cfg.get("buy_threshold", 0.52),
            sell_threshold=trade_cfg.get("sell_threshold", 0.48),
            confirm_days=trade_cfg.get("confirm_days", 1),
            max_hold_days=trade_cfg.get("max_hold_days", 20),
        )

        logger.info("Final model: total_return=%.4f, sharpe=%.4f, max_dd=%.4f",
                     bt_result.risk_metrics.get("total_return", 0),
                     bt_result.risk_metrics.get("sharpe", 0),
                     bt_result.risk_metrics.get("max_drawdown", 0))

        # 保存报告
        from dpoint.reports.excel_reporter import save_excel_report

        search_log = []
        for r in state.all_results:
            log_entry = {"score": r.score, "elapsed": r.elapsed_sec}
            log_entry.update(r.config.get("model", {}))
            log_entry.update(r.config.get("trade", {}))
            search_log.append(log_entry)

        report_path = save_excel_report(
            exp_dir / "report.xlsx",
            equity_curve=bt_result.equity_curve,
            trades=bt_result.trades,
            risk_metrics=bt_result.risk_metrics,
            config=config.to_dict(),
            search_log=search_log,
            notes=bt_result.notes + [f"Resumed from: {parent_dir}"],
        )
        logger.info("Report saved: %s", report_path)

        try:
            from dpoint.reports.html_reporter import save_html_report
            save_html_report(
                exp_dir / "report.html",
                equity_curve=bt_result.equity_curve,
                risk_metrics=bt_result.risk_metrics,
                trades=bt_result.trades,
                title=f"Dpoint Resume Report: {report.ticker}",
            )
        except ImportError:
            pass

        save_config(exp_dir, config.to_dict())
        create_manifest(
            exp_dir,
            config=config.to_dict(),
            data_hash=compute_data_hash(df),
            seed=config.search.seed,
        )

        logger.info("All outputs saved to: %s", exp_dir)
    else:
        logger.warning("No valid candidates found during search")

    return 0


def _run_resume_basket(config, state, rng, data_path, parent_dir, args, logger) -> int:
    """篮子模式的 resume 流程。"""
    import numpy as _np

    logger.info("Resuming basket search from: %s", parent_dir)

    # 加载数据
    logger.info("Loading basket: %s", data_path)
    from dpoint.data.basket_loader import load_basket_folder

    panel_df, basket_report = load_basket_folder(
        str(data_path), calendar_align="none",
    )
    logger.info("Loaded basket: %d rows, %d tickers, %d dates",
                len(panel_df), basket_report.n_files_loaded, panel_df["date"].nunique())

    # 创建新实验目录
    from dpoint.search.engine import create_evaluate_fn_basket, random_search, save_search_state
    from dpoint.core.utils import create_experiment_dir

    exp_dir = create_experiment_dir(config.output_dir, prefix="basket")
    logger.info("New experiment directory: %s (parent: %s)", exp_dir, parent_dir)

    # 搜索
    evaluate_fn = create_evaluate_fn_basket(panel_df, config)
    model_types = args.model_types or [config.model.model_type]

    t0 = time.time()
    state, rng = random_search(
        config, evaluate_fn,
        model_types=model_types,
        initial_state=state,
        rng=rng,
    )
    elapsed = time.time() - t0

    save_search_state(state, rng, exp_dir / "search_state.json")

    logger.info("Search completed in %.1fs: %d evaluated, best_score=%.4f",
                elapsed, state.n_evaluated, state.best_score)

    # 训练最终模型并计算因子指标
    if state.best_config:
        logger.info("Training final model with best config...")
        from dpoint.core.config import FeatureConfig
        from dpoint.features.pipeline import build_features_and_labels
        from dpoint.models.registry import make_model
        from dpoint.models.trainer import predict_pytorch_model, predict_sklearn_model, train_pytorch_model, train_sklearn_model
        from dpoint.reports.metrics import compute_ranking_metrics
        from dpoint.core.tasks import resolve_label_spec

        best = state.best_config
        feat_cfg = best.get("feature", {})
        feature_config = FeatureConfig(
            use_momentum=feat_cfg.get("use_momentum", True),
            use_volatility=feat_cfg.get("use_volatility", True),
            use_volume=feat_cfg.get("use_volume", True),
            use_candle=feat_cfg.get("use_candle", True),
            use_turnover=feat_cfg.get("use_turnover", True),
            use_ta_indicators=feat_cfg.get("use_ta_indicators", True),
        )

        df_feat, y, meta = build_features_and_labels(panel_df.copy(), feature_config, mode="basket")
        feature_names = meta.feature_names
        model_cfg = best.get("model", {})
        model_type = model_cfg.get("model_type", "logreg")

        X_all = df_feat[feature_names].values
        y_all = y.values
        X_all = _np.nan_to_num(X_all, nan=0.0)

        label_spec = resolve_label_spec()
        model, kind = make_model(model_type, X_all.shape[1], model_cfg, label_spec)

        if kind == "sklearn":
            model = train_sklearn_model(model, X_all, y_all)
            proba = predict_sklearn_model(model, X_all)
        else:
            train_pytorch_model(model, X_all, y_all, epochs=model_cfg.get("epochs", 50),
                                batch_size=model_cfg.get("batch_size", 256),
                                learning_rate=model_cfg.get("learning_rate", 1e-3),
                                device=config.device)
            proba = predict_pytorch_model(model, X_all, device=config.device)

        df_feat["pred_score"] = proba
        df_feat["label"] = y.values
        ranking_metrics = compute_ranking_metrics(df_feat, score_col="pred_score", label_col="label")

        logger.info("Rank IC: mean=%.4f, std=%.4f, IR=%.4f",
                     ranking_metrics.rank_ic_mean or 0, ranking_metrics.rank_ic_std or 0, ranking_metrics.rank_ic_ir or 0)
        logger.info("Top-K return: mean=%.6f, annual=%.4f",
                     ranking_metrics.topk_return_mean or 0, ranking_metrics.topk_return_annual or 0)

        # 保存报告
        from dpoint.reports.excel_reporter import save_excel_report

        search_log = []
        for r in state.all_results:
            log_entry = {"score": r.score, "elapsed": r.elapsed_sec}
            log_entry.update(r.config.get("model", {}))
            search_log.append(log_entry)

        rm_dict = {
            "ic_mean": ranking_metrics.ic_mean,
            "rank_ic_mean": ranking_metrics.rank_ic_mean,
            "rank_ic_ir": ranking_metrics.rank_ic_ir,
            "topk_return_mean": ranking_metrics.topk_return_mean,
        }
        if ranking_metrics.layered_returns:
            rm_dict.update(ranking_metrics.layered_returns)

        report_path = save_excel_report(
            exp_dir / "report.xlsx",
            ranking_metrics=rm_dict,
            config=config.to_dict(),
            search_log=search_log,
        )
        logger.info("Report saved: %s", report_path)

        save_config(exp_dir, config.to_dict())
        create_manifest(
            exp_dir,
            config=config.to_dict(),
            data_hash=compute_data_hash(panel_df),
            seed=config.search.seed,
        )

        logger.info("All outputs saved to: %s", exp_dir)
    else:
        logger.warning("No valid candidates found during search")

    return 0


def _default_date_range(args) -> tuple[str, str]:
    """计算默认日期范围（6年前至今）。"""
    from datetime import datetime, timedelta

    now = datetime.now()
    start = args.start or (now - timedelta(days=365 * 6)).strftime("%Y%m%d")
    end = args.end or now.strftime("%Y%m%d")
    return start, end


def _make_qmt_client():
    """创建 QMT 客户端，失败时返回 None。"""
    from dpoint.data.fetch.qmt_client import QMTClient

    try:
        return QMTClient()
    except ImportError as e:
        logging.getLogger("dpoint.fetch").error(str(e))
        return None


def run_fetch_single(args) -> int:
    """获取单只股票历史数据。"""
    logger = logging.getLogger("dpoint.fetch.single")

    from dpoint.data.fetch.formatter import generate_csv_filename, qmt_to_dpoint_single

    start, end = _default_date_range(args)

    # 确定输出路径
    if args.output:
        output_path = Path(args.output)
    else:
        ext = "xlsx" if args.format == "xlsx" else "csv"
        stem = generate_csv_filename(args.code, start).replace(".csv", "")
        output_path = Path("data") / f"{stem}_{end}.{ext}"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 获取数据
    logger.info("Fetching %s from QMT...", args.code)
    client = _make_qmt_client()
    if client is None:
        return 1

    raw_df = client.fetch_daily_history(args.code, start_date=start, end_date=end)
    if raw_df.empty:
        logger.error("未获取到 %s 的数据", args.code)
        return 1

    # 转换格式
    df = qmt_to_dpoint_single(raw_df)
    logger.info("Converted to Dpoint_Trader format: %d rows", len(df))

    # 保存
    if output_path.suffix in (".xlsx", ".xls"):
        df.to_excel(output_path, index=False, engine="openpyxl")
    else:
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

    logger.info("Saved to: %s", output_path)
    logger.info("可直接用于: dpoint single --data_path %s", output_path)
    return 0


def run_fetch_basket(args) -> int:
    """获取行业篮子数据。"""
    logger = logging.getLogger("dpoint.fetch.basket")

    from dpoint.data.fetch.formatter import (
        generate_csv_filename,
        qmt_to_dpoint_csv,
        qmt_to_dpoint_single,
    )
    from dpoint.data.fetch.industry import DEFAULT_DB_PATH, IndustryDB

    # 确定数据库路径
    db_path = args.db if args.db else DEFAULT_DB_PATH

    # 查询行业成员
    try:
        with IndustryDB(db_path) as db:
            members = db.get_industry_members(args.industry)
            if not members:
                logger.error("行业代码 '%s' 未找到任何股票", args.industry)
                logger.info("可用行业示例:")
                for info in db.list_industries()[:10]:
                    logger.info("  %s %s (%d只)", info.code, info.name, info.count)
                return 1
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    logger.info("行业 %s 共 %d 只股票", args.industry, len(members))

    # 确定输出目录
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path("data") / f"basket_{args.industry}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 批量获取
    client = _make_qmt_client()
    if client is None:
        return 1

    start, end = _default_date_range(args)
    data = client.fetch_batch(members, start_date=start, end_date=end)

    # 保存
    saved = 0
    for code, raw_df in data.items():
        if args.format == "xlsx":
            df = qmt_to_dpoint_single(raw_df)
            stem = generate_csv_filename(code, start).replace(".csv", "")
            filepath = output_dir / f"{stem}.xlsx"
            df.to_excel(filepath, index=False, engine="openpyxl")
        else:
            df = qmt_to_dpoint_csv(raw_df)
            filename = generate_csv_filename(code, start)
            filepath = output_dir / filename
            df.to_csv(filepath, index=False, encoding="utf-8-sig")
        saved += 1

    logger.info("Saved %d stocks to: %s", saved, output_dir)
    logger.info("可直接用于: dpoint basket --basket_path %s", output_dir)
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    setup_logging(args.verbose)

    if args.command == "single":
        return run_single(args)
    elif args.command == "basket":
        return run_basket(args)
    elif args.command == "resume":
        return run_resume(args)
    elif args.command == "fetch":
        if not args.fetch_mode:
            # 重新解析 fetch --help 来打印帮助（捕获 SystemExit）
            try:
                parser.parse_args(["fetch", "--help"])
            except SystemExit:
                pass
            return 1
        if args.fetch_mode == "single":
            return run_fetch_single(args)
        elif args.fetch_mode == "basket":
            return run_fetch_basket(args)
        else:
            logger.error("未知的 fetch 模式: %s", args.fetch_mode)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
