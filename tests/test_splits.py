# test_splits.py
"""样本划分测试。"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dpoint.splits.splitters import (
    walkforward_splits, walkforward_splits_with_embargo,
    final_holdout_split, recommend_n_folds,
)


def test_walkforward_basic(sample_single_df):
    results = walkforward_splits(sample_single_df, n_folds=3, train_start_ratio=0.5)
    assert len(results) > 0
    for r in results:
        assert len(r.train_dates) > 0
        assert len(r.val_dates) > 0
        # 训练集时间严格早于验证集
        assert max(r.train_dates) < min(r.val_dates)


def test_walkforward_no_overlap(sample_single_df):
    results = walkforward_splits(sample_single_df, n_folds=4)
    for i in range(len(results) - 1):
        # 验证集不重叠
        assert max(results[i].val_dates) <= min(results[i + 1].val_dates)


def test_embargo_gap(sample_single_df):
    results = walkforward_splits_with_embargo(sample_single_df, n_folds=3, embargo_days=5)
    for r in results:
        assert r.spec.embargo_days == 5
        # 训练集末尾和验证集开头之间有间隔
        gap = (min(r.val_dates) - max(r.train_dates)).days
        assert gap >= 0


def test_holdout_split(sample_single_df):
    search_dates, holdout_dates = final_holdout_split(sample_single_df, holdout_ratio=0.15)
    assert len(search_dates) > 0
    assert len(holdout_dates) > 0
    # holdout 在 search 之后
    assert max(search_dates) < min(holdout_dates)


def test_recommend_n_folds():
    assert recommend_n_folds(500) >= 2
    assert recommend_n_folds(500) <= 8
    assert recommend_n_folds(100) >= 2


def test_panel_walkforward(sample_panel_df):
    results = walkforward_splits(sample_panel_df, n_folds=3)
    assert len(results) > 0
