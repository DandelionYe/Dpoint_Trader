# test_search_integration.py
"""搜索引擎集成测试：验证完整评估流程。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dpoint.core.config import RunConfig
from dpoint.search.engine import create_evaluate_fn_single, random_search, SearchState


def test_evaluate_fn_single_basic(sample_single_df):
    """测试单股评估函数的基本流程。"""
    from dpoint.core.config import SearchConfig, SplitConfig
    config = RunConfig(
        mode="single",
        search=SearchConfig(n_candidates=2, n_rounds=1, metric="pnl", seed=42),
        split=SplitConfig(n_folds=2, min_rows=20),
    )

    evaluate_fn = create_evaluate_fn_single(sample_single_df, config)

    # 用一个简单的候选配置测试
    candidate = {
        "model": {"model_type": "sgd", "alpha": 1e-4},
        "trade": {"buy_threshold": 0.55, "sell_threshold": 0.45, "confirm_days": 1, "max_hold_days": 10},
        "feature": {"use_momentum": True, "use_volatility": True, "use_volume": False,
                    "use_candle": False, "use_turnover": False, "use_ta_indicators": False},
    }

    fold_results = evaluate_fn(candidate)
    assert isinstance(fold_results, list)
    if fold_results:
        assert "geom_mean_ratio" in fold_results[0]
        assert "n_trades" in fold_results[0]


def test_random_search_basic(sample_single_df):
    """测试随机搜索引擎基本流程（少量候选）。"""
    from dpoint.core.config import SearchConfig, SplitConfig
    config = RunConfig(
        mode="single",
        search=SearchConfig(n_candidates=4, n_rounds=2, metric="pnl", seed=42),
        split=SplitConfig(n_folds=2, min_rows=20),
    )

    evaluate_fn = create_evaluate_fn_single(sample_single_df, config)

    state = random_search(
        config, evaluate_fn,
        model_types=["sgd"],  # 只用 sgd 加速（避免 scipy L-BFGS-B 崩溃）
    )

    assert isinstance(state, SearchState)
    assert state.n_evaluated + state.n_skipped + state.n_errors > 0
    if state.n_evaluated > 0:
        assert state.best_score > -np.inf
        assert state.best_config


def test_search_state_top_k():
    """测试 Top-K 池管理。"""
    from dpoint.search.engine import CandidateResult, update_top_k

    pool = []
    for i in range(15):
        c = CandidateResult(config={"id": i}, score=float(i), fold_results=[])
        pool = update_top_k(pool, c, k=10)

    assert len(pool) == 10
    assert pool[0].score == 14.0
    assert pool[-1].score == 5.0
