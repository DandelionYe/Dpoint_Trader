# engine.py
"""
统一搜索引擎 + 完整评估流程。
Phase 3：集成特征→模型→回测→指标的完整 evaluate_fn。
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from dpoint.core.config import RunConfig
from dpoint.search.metrics import MetricFn, get_metric_fn
from dpoint.search.space import (
    ALL_MODELS, mutate_model_config, sample_feature_config,
    sample_model_config, sample_trade_config,
)

logger = logging.getLogger(__name__)


@dataclass
class CandidateResult:
    """单个候选配置的评估结果。"""
    config: Dict[str, Any]
    score: float
    fold_results: List[Dict[str, Any]]
    elapsed_sec: float = 0.0
    status: str = "ok"


@dataclass
class SearchState:
    """搜索状态。"""
    best_score: float = -np.inf
    best_config: Dict[str, Any] = field(default_factory=dict)
    top_k_pool: List[CandidateResult] = field(default_factory=list)
    all_results: List[CandidateResult] = field(default_factory=list)
    n_evaluated: int = 0
    n_skipped: int = 0
    n_errors: int = 0


def update_top_k(pool: List[CandidateResult], candidate: CandidateResult, k: int = 10) -> List[CandidateResult]:
    """更新 Top-K 候选池。"""
    pool.append(candidate)
    pool.sort(key=lambda c: c.score, reverse=True)
    return pool[:k]


# ==============================================================
# 搜索状态序列化（用于迭代训练 / resume）
# ==============================================================

def _json_default(obj: Any) -> Any:
    """JSON 序列化辅助：处理 numpy 类型和无穷值。"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if np.isinf(val) or np.isnan(val):
            return str(val)  # "inf" / "-inf" / "nan" 以字符串保存
        return val
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _json_loads_hook(d: dict) -> dict:
    """JSON 反序列化辅助：恢复 "inf" / "-inf" 字符串为 float。"""
    for k, v in d.items():
        if isinstance(v, str):
            if v == "inf":
                d[k] = float("inf")
            elif v == "-inf":
                d[k] = float("-inf")
            elif v == "nan":
                d[k] = float("nan")
    return d


def save_search_state(
    state: SearchState,
    rng: np.random.Generator,
    path: Path,
) -> None:
    """将搜索状态和 RNG 状态保存到 JSON 文件。"""
    data = {
        "best_score": state.best_score,
        "best_config": state.best_config,
        "top_k_pool": [asdict(c) for c in state.top_k_pool],
        "all_results": [asdict(c) for c in state.all_results],
        "n_evaluated": state.n_evaluated,
        "n_skipped": state.n_skipped,
        "n_errors": state.n_errors,
        "rng_state": rng.bit_generator.state,
    }
    path.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")
    logger.info("Search state saved: %s (%d evaluated, top-K size %d)", path, state.n_evaluated, len(state.top_k_pool))


def load_search_state(path: Path) -> Tuple[SearchState, dict]:
    """
    从 JSON 文件加载搜索状态。
    返回 (SearchState, rng_state_dict)。
    """
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw, object_hook=_json_loads_hook)

    state = SearchState(
        best_score=data["best_score"],
        best_config=data["best_config"],
        top_k_pool=[CandidateResult(**c) for c in data["top_k_pool"]],
        all_results=[CandidateResult(**c) for c in data["all_results"]],
        n_evaluated=data["n_evaluated"],
        n_skipped=data["n_skipped"],
        n_errors=data["n_errors"],
    )
    rng_state = data["rng_state"]

    logger.info(
        "Search state loaded: %d evaluated, best_score=%.4f, top-K size %d",
        state.n_evaluated, state.best_score, len(state.top_k_pool),
    )
    return state, rng_state


def create_evaluate_fn_single(
    df: pd.DataFrame,
    config: RunConfig,
) -> Callable[[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    创建单股模式的评估函数。
    完整流程：特征构建 → Walk-Forward 切分 → 训练 → 预测 → 回测 → 指标。
    """
    from dpoint.core.config import FeatureConfig
    from dpoint.features.pipeline import build_features_and_labels
    from dpoint.models.registry import make_model
    from dpoint.models.trainer import (
        predict_pytorch_model, predict_sklearn_model,
        train_pytorch_model, train_sklearn_model,
    )
    from dpoint.splits.splitters import walkforward_splits
    from dpoint.backtester.single_stock import backtest_from_dpoint, compute_fold_metrics

    def evaluate_fn(candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
        model_cfg = candidate.get("model", {})
        trade_cfg = candidate.get("trade", {})
        feat_cfg = candidate.get("feature", {})

        # 构建特征配置
        feature_config = FeatureConfig(
            use_momentum=feat_cfg.get("use_momentum", True),
            use_volatility=feat_cfg.get("use_volatility", True),
            use_volume=feat_cfg.get("use_volume", True),
            use_candle=feat_cfg.get("use_candle", True),
            use_turnover=feat_cfg.get("use_turnover", True),
            use_ta_indicators=feat_cfg.get("use_ta_indicators", True),
            vol_metric=feat_cfg.get("vol_metric", "std"),
            liq_transform=feat_cfg.get("liq_transform", "ratio"),
        )

        # 构建特征
        df_feat, y, meta = build_features_and_labels(
            df.copy(), feature_config, mode="single",
        )

        if meta.n_samples < config.split.min_rows:
            return []

        feature_names = meta.feature_names
        model_type = model_cfg.get("model_type", "logreg")

        # Walk-Forward 切分
        splits = walkforward_splits(
            df_feat, n_folds=config.split.n_folds,
            train_start_ratio=config.split.train_start_ratio,
            min_rows=config.split.min_rows,
        )

        if not splits:
            return []

        fold_results = []
        from dpoint.core.tasks import resolve_label_spec
        label_spec = resolve_label_spec()

        for split in splits:
            train_mask = df_feat["date"].isin(split.train_dates)
            val_mask = df_feat["date"].isin(split.val_dates)

            X_train = df_feat.loc[train_mask, feature_names].values
            y_train = y[train_mask].values
            X_val = df_feat.loc[val_mask, feature_names].values
            y_val = y[val_mask].values

            if len(X_train) < 10 or len(X_val) < 5:
                continue

            # 处理 NaN
            X_train = np.nan_to_num(X_train, nan=0.0)
            X_val = np.nan_to_num(X_val, nan=0.0)

            # 创建并训练模型
            try:
                model, kind = make_model(model_type, X_train.shape[1], model_cfg, label_spec)

                if kind == "sklearn":
                    model = train_sklearn_model(model, X_train, y_train)
                    proba = predict_sklearn_model(model, X_val)
                else:
                    train_pytorch_model(
                        model, X_train, y_train, X_val, y_val,
                        epochs=model_cfg.get("epochs", 50),
                        batch_size=model_cfg.get("batch_size", 256),
                        learning_rate=model_cfg.get("learning_rate", 1e-3),
                        patience=model_cfg.get("patience", 10),
                        device=config.device,
                    )
                    proba = predict_pytorch_model(model, X_val, device=config.device)
            except Exception as e:
                logger.debug("Model training failed: %s", e)
                continue

            # 构建 dpoint 序列
            val_df = df_feat[val_mask].copy()
            dpoint = pd.Series(proba, index=val_df["date"].values)

            # 回测
            bt_result = backtest_from_dpoint(
                val_df, dpoint,
                buy_threshold=trade_cfg.get("buy_threshold", 0.55),
                sell_threshold=trade_cfg.get("sell_threshold", 0.45),
                confirm_days=trade_cfg.get("confirm_days", 1),
                max_hold_days=trade_cfg.get("max_hold_days", 20),
                take_profit=trade_cfg.get("take_profit", 0.0),
                stop_loss=trade_cfg.get("stop_loss", 0.0),
                initial_cash=100_000.0,
            )

            fold_metrics = compute_fold_metrics(bt_result)
            fold_metrics["fold_id"] = split.spec.fold_id
            fold_results.append(fold_metrics)

        return fold_results

    return evaluate_fn


def create_evaluate_fn_basket(
    panel_df: pd.DataFrame,
    config: RunConfig,
) -> Callable[[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    创建篮子模式的评估函数。
    完整流程：面板特征 → Walk-Forward → 训练 → 预测 → Rank IC 计算。
    """
    from dpoint.core.config import FeatureConfig
    from dpoint.features.pipeline import build_features_and_labels
    from dpoint.models.registry import make_model
    from dpoint.models.trainer import (
        predict_pytorch_model, predict_sklearn_model,
        train_pytorch_model, train_sklearn_model,
    )
    from dpoint.splits.splitters import walkforward_splits

    def evaluate_fn(candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
        model_cfg = candidate.get("model", {})
        feat_cfg = candidate.get("feature", {})

        feature_config = FeatureConfig(
            use_momentum=feat_cfg.get("use_momentum", True),
            use_volatility=feat_cfg.get("use_volatility", True),
            use_volume=feat_cfg.get("use_volume", True),
            use_candle=feat_cfg.get("use_candle", True),
            use_turnover=feat_cfg.get("use_turnover", True),
            use_ta_indicators=feat_cfg.get("use_ta_indicators", True),
            vol_metric=feat_cfg.get("vol_metric", "std"),
            liq_transform=feat_cfg.get("liq_transform", "ratio"),
        )

        df_feat, y, meta = build_features_and_labels(
            panel_df.copy(), feature_config, mode="basket",
        )

        if meta.n_samples < config.split.min_rows:
            return []

        feature_names = meta.feature_names
        model_type = model_cfg.get("model_type", "logreg")

        splits = walkforward_splits(
            df_feat, n_folds=config.split.n_folds,
            train_start_ratio=config.split.train_start_ratio,
            min_rows=config.split.min_rows,
        )

        if not splits:
            return []

        fold_results = []
        from dpoint.core.tasks import resolve_label_spec
        label_spec = resolve_label_spec()

        for split in splits:
            train_mask = df_feat["date"].isin(split.train_dates)
            val_mask = df_feat["date"].isin(split.val_dates)

            X_train = df_feat.loc[train_mask, feature_names].values
            y_train = y[train_mask].values
            X_val = df_feat.loc[val_mask, feature_names].values

            if len(X_train) < 10 or len(X_val) < 5:
                continue

            X_train = np.nan_to_num(X_train, nan=0.0)
            X_val = np.nan_to_num(X_val, nan=0.0)

            try:
                model, kind = make_model(model_type, X_train.shape[1], model_cfg, label_spec)

                if kind == "sklearn":
                    model = train_sklearn_model(model, X_train, y_train)
                    proba = predict_sklearn_model(model, X_val)
                else:
                    y_val = y[val_mask].values
                    train_pytorch_model(
                        model, X_train, y_train, X_val, y_val,
                        epochs=model_cfg.get("epochs", 50),
                        batch_size=model_cfg.get("batch_size", 256),
                        learning_rate=model_cfg.get("learning_rate", 1e-3),
                        patience=model_cfg.get("patience", 10),
                        device=config.device,
                    )
                    proba = predict_pytorch_model(model, X_val, device=config.device)
            except Exception as e:
                logger.debug("Model training failed: %s", e)
                continue

            # 计算 Rank IC
            val_df = df_feat[val_mask].copy()
            val_df = val_df.copy()
            val_df["pred_score"] = proba
            val_df["label"] = y[val_mask].values

            rank_ics = []
            for dt, group in val_df.groupby("date"):
                if len(group) < 5:
                    continue
                true_rank = group["close_qfq"].rank(pct=True) if "close_qfq" in group.columns else None
                pred_rank = group["pred_score"].rank(pct=True)
                # 用次日收益率的 rank 作为真实 rank
                if "label" in group.columns:
                    true_rank = group["label"].rank(pct=True)
                    ic = true_rank.corr(pred_rank)
                    if not np.isnan(ic):
                        rank_ics.append(ic)

            avg_rank_ic = float(np.mean(rank_ics)) if rank_ics else 0.0

            fold_results.append({
                "fold_id": split.spec.fold_id,
                "rank_ic": avg_rank_ic,
                "n_samples": len(val_df),
            })

        return fold_results

    return evaluate_fn


def random_search(
    config: RunConfig,
    evaluate_fn: Callable[[Dict[str, Any]], List[Dict[str, Any]]],
    *,
    model_types: Optional[List[str]] = None,
    metric_name: str = "",
    progress_fn: Optional[Callable[[int, int, SearchState], None]] = None,
    initial_state: Optional[SearchState] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[SearchState, np.random.Generator]:
    """
    随机搜索引擎。

    Args:
        config: 运行配置
        evaluate_fn: 评估函数
        model_types: 搜索的模型类型列表
        metric_name: 目标函数名称
        progress_fn: 进度回调
        initial_state: 恢复模式 — 传入上一轮的搜索状态
        rng: 外部传入的 RNG（恢复模式下传入已恢复状态的 RNG）

    Returns:
        (SearchState, rng) — 返回 RNG 以便调用方保存最终状态
    """
    if rng is None:
        rng = np.random.Generator(np.random.PCG64(config.search.seed))
    metric_fn = get_metric_fn(metric_name or config.search.metric)
    model_types = model_types or ALL_MODELS

    # 恢复模式：从 initial_state 继续
    if initial_state is not None:
        state = initial_state
        logger.info(
            "Resuming search: %d previously evaluated, best_score=%.4f, top-K pool size %d",
            state.n_evaluated, state.best_score, len(state.top_k_pool),
        )
    else:
        state = SearchState()

    n_candidates = config.search.n_candidates
    n_rounds = config.search.n_rounds
    candidates_per_round = max(1, n_candidates // n_rounds)

    logger.info(
        "Starting search: %d candidates, %d rounds, metric=%s",
        n_candidates, n_rounds, config.search.metric,
    )

    for round_idx in range(n_rounds):
        logger.info("=== Round %d/%d ===", round_idx + 1, n_rounds)

        for i in range(candidates_per_round):
            global_idx = round_idx * candidates_per_round + i

            # 采样候选配置
            if state.top_k_pool and rng.random() < 0.3:
                base = state.top_k_pool[rng.integers(len(state.top_k_pool))]
                model_cfg = mutate_model_config(base.config.get("model", {}), rng)
            else:
                model_type = rng.choice(model_types)
                model_cfg = sample_model_config(model_type, rng)

            trade_cfg = sample_trade_config(rng)
            feat_cfg = sample_feature_config(rng)
            candidate = {"model": model_cfg, "trade": trade_cfg, "feature": feat_cfg}

            # 评估
            t0 = time.time()
            try:
                fold_results = evaluate_fn(candidate)
                elapsed = time.time() - t0

                if not fold_results:
                    state.n_skipped += 1
                    continue

                score = metric_fn(fold_results)
                result = CandidateResult(
                    config=candidate, score=score, fold_results=fold_results, elapsed_sec=elapsed,
                )
                state.all_results.append(result)
                state.n_evaluated += 1

                if score > state.best_score:
                    state.best_score = score
                    state.best_config = candidate

                state.top_k_pool = update_top_k(state.top_k_pool, result, k=10)

                if progress_fn:
                    progress_fn(global_idx + 1, n_candidates, state)

            except Exception as e:
                logger.warning("Candidate %d failed: %s", global_idx, e)
                state.n_errors += 1

    logger.info(
        "Search complete: %d evaluated, %d skipped, %d errors, best_score=%.4f",
        state.n_evaluated, state.n_skipped, state.n_errors, state.best_score,
    )

    return state, rng
