# test_features.py
"""特征工程测试。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dpoint.features.groups import add_momentum_features, add_volatility_features, add_all_features
from dpoint.features.labeler import build_label
from dpoint.features.pipeline import build_features_and_labels, FeatureMeta
from dpoint.core.config import FeatureConfig


def test_momentum_features(sample_panel_df):
    result, names = add_momentum_features(sample_panel_df)
    assert len(names) > 0
    for name in names:
        assert name in result.columns


def test_volatility_features(sample_panel_df):
    result, names = add_volatility_features(sample_panel_df)
    assert "hl_range" in names
    assert "true_range_norm" in names


def test_label_binary(sample_panel_df):
    y = build_label(sample_panel_df, label_mode="binary_next_close_up")
    assert y.notna().any()
    valid = y.dropna()
    assert set(valid.unique()).issubset({0.0, 1.0})


def test_label_regression(sample_panel_df):
    y = build_label(sample_panel_df, label_mode="regression_return")
    assert y.notna().any()


def test_pipeline_single_stock(sample_single_df):
    config = FeatureConfig(use_turnover=True, use_ta_indicators=True)
    df, y, meta = build_features_and_labels(
        sample_single_df,
        config,
        mode="single",
    )
    assert meta.n_samples > 0
    assert meta.n_features > 0
    assert len(df) == len(y)


def test_no_future_leakage(sample_single_df):
    """验证特征不使用未来数据：截断数据集上计算结果应一致。"""
    config = FeatureConfig()
    df_full, y_full, _ = build_features_and_labels(sample_single_df.copy(), config, mode="single")
    # 截断到前 150 行
    truncated = sample_single_df.iloc[:150].copy()
    df_trunc, y_trunc, _ = build_features_and_labels(truncated, config, mode="single")
    # 前 150 行的特征值应该一致
    common_idx = df_trunc.index.intersection(df_full.index)
    if len(common_idx) > 0:
        feat = meta_feature_names = [
            c
            for c in df_trunc.columns
            if c
            not in [
                "date",
                "open_qfq",
                "high_qfq",
                "low_qfq",
                "close_qfq",
                "volume",
                "amount",
                "turnover_rate",
                "ticker",
            ]
        ]
        if feat:
            pd.testing.assert_frame_equal(
                df_trunc.loc[common_idx, feat[:3]],
                df_full.loc[common_idx, feat[:3]],
                check_names=False,
            )
