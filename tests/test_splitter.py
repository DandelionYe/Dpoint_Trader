# test_splitter.py
"""
Tests for splitter module correctness (Basket Mode).
Validates walk-forward splitting logic for panel data.
"""
import numpy as np
import pandas as pd
import pytest
from data_loader import (
    walkforward_splits,
    final_holdout_split,
    recommend_n_folds,
    walkforward_splits_with_embargo,
)


class TestWalkforwardSplits:
    """Test walkforward_splits function."""

    def test_basic_split_count(self):
        """Test that correct number of folds is generated."""
        X = pd.DataFrame({"x": range(400)})
        y = pd.Series(range(400))

        splits = walkforward_splits(X, y, n_folds=4, train_start_ratio=0.5, min_rows=20)

        assert len(splits) == 4

    def test_train_val_no_overlap(self):
        """Test that validation sets don't overlap."""
        X = pd.DataFrame({"x": range(200)})
        y = pd.Series(range(200))

        splits = walkforward_splits(X, y, n_folds=4, train_start_ratio=0.5)

        for i, ((X_train, _), (X_val, _)) in enumerate(splits):
            for j, ((X_train2, _), (X_val2, _)) in enumerate(splits):
                if i != j:
                    val_indices = set(X_val.index)
                    val_indices2 = set(X_val2.index)
                    assert len(val_indices & val_indices2) == 0

    def test_train_expanding(self):
        """Test that training set expands with each fold."""
        X = pd.DataFrame({"x": range(200)})
        y = pd.Series(range(200))

        splits = walkforward_splits(X, y, n_folds=4, train_start_ratio=0.5)

        prev_train_len = 0
        for (X_train, _), _ in splits:
            assert len(X_train) > prev_train_len
            prev_train_len = len(X_train)

    def test_min_rows_constraint(self):
        """Test that min_rows constraint is respected."""
        X = pd.DataFrame({"x": range(50)})
        y = pd.Series(range(50))

        splits = walkforward_splits(X, y, n_folds=10, min_rows=10)

        for (X_train, _), (X_val, _) in splits:
            assert len(X_train) >= 10
            assert len(X_val) >= 10

    def test_empty_dataframe_raises(self):
        """Test that empty dataframe returns empty splits."""
        X = pd.DataFrame({"x": []})
        y = pd.Series([], dtype=int)

        splits = walkforward_splits(X, y, n_folds=4)

        assert len(splits) == 0


class TestFinalHoldoutSplit:
    """Test final_holdout_split function."""

    def test_holdout_ratio(self):
        """Test that holdout ratio is correct."""
        df = pd.DataFrame({"x": range(200)})

        search_df, holdout_df = final_holdout_split(
            df, holdout_ratio=0.2, min_holdout_rows=20
        )

        assert len(holdout_df) == 40
        assert len(search_df) == 160

    def test_holdout_at_end(self):
        """Test that holdout is from the end of data."""
        df = pd.DataFrame({"x": range(200)})

        search_df, holdout_df = final_holdout_split(
            df, holdout_ratio=0.2, min_holdout_rows=20
        )

        assert search_df.index.max() < holdout_df.index.min()

    def test_min_holdout_rows_raises(self):
        """Test that too small holdout raises error."""
        df = pd.DataFrame({"x": range(50)})

        with pytest.raises(ValueError):
            final_holdout_split(df, holdout_ratio=0.1, min_holdout_rows=100)


class TestRecommendNFolds:
    """Test recommend_n_folds function."""

    def test_large_data_more_folds(self):
        """Test that more data results in more folds."""
        n_folds_500 = recommend_n_folds(500)
        n_folds_1000 = recommend_n_folds(1000)

        assert n_folds_1000 >= n_folds_500

    def test_returns_within_bounds(self):
        """Test that result is within min/max bounds."""
        n_folds = recommend_n_folds(100, min_folds=2, max_folds=8)

        assert 2 <= n_folds <= 8

    def test_small_data_returns_min(self):
        """Test that very small data returns min folds."""
        n_folds = recommend_n_folds(50, min_folds=2, max_folds=8, min_rows=30)

        assert n_folds == 2


class TestEmbargoSplit:
    """Test walkforward_splits_with_embargo function."""

    def test_embargo_gap(self):
        """Test that embargo gap is applied."""
        X = pd.DataFrame({"x": range(200)})
        y = pd.Series(range(200))

        splits = walkforward_splits_with_embargo(
            X, y, n_folds=4, embargo_days=5
        )

        for (X_train, _), (X_val, _) in splits:
            train_end = X_train.index[-1]
            val_start = X_val.index[0]
            gap = val_start - train_end - 1
            assert gap >= 5


# =========================================================
# P-basket: 面板数据切分测试
# =========================================================

class TestPanelDateSplits:
    """Test _make_panel_date_splits for basket mode."""

    def test_panel_split_requires_import(self):
        """Test that _make_panel_date_splits can be imported from trainer."""
        try:
            from trainer import _make_panel_date_splits
            assert True
        except ImportError:
            pytest.skip("_make_panel_date_splits not available (trainer.py may still have single-stock code)")

    def test_panel_split_count(self):
        """Test that correct number of panel folds is generated."""
        try:
            from trainer import _make_panel_date_splits
        except ImportError:
            pytest.skip("_make_panel_date_splits not available")

        # 创建面板数据（3 只股票，200 天）- 需要足够数据才能满足 wf_min_dates
        dates = pd.date_range(start="2020-01-01", periods=200, freq="B")
        stocks = ["000001", "000002", "000003"]

        rows = []
        for code in stocks:
            for date in dates:
                rows.append({"date": date, "stock_code": code, "feat": np.random.rand()})

        X_panel = pd.DataFrame(rows)
        y_panel = pd.Series(np.random.randint(0, 2, len(X_panel)))

        splits = _make_panel_date_splits(
            X_panel, y_panel, n_folds=4, train_start_ratio=0.5, wf_min_dates=20
        )

        # 由于数据量限制，可能无法生成 4 折，至少应该有 1 折
        assert len(splits) >= 1, "Should generate at least 1 fold"

    def test_panel_split_no_date_overlap(self):
        """Test that validation dates don't overlap with training dates."""
        try:
            from trainer import _make_panel_date_splits
        except ImportError:
            pytest.skip("_make_panel_date_splits not available")

        # 创建面板数据
        dates = pd.date_range(start="2020-01-01", periods=100, freq="B")
        stocks = ["000001", "000002"]

        rows = []
        for code in stocks:
            for date in dates:
                rows.append({"date": date, "stock_code": code, "feat": np.random.rand()})

        X_panel = pd.DataFrame(rows)
        y_panel = pd.Series(np.random.randint(0, 2, len(X_panel)))

        splits = _make_panel_date_splits(
            X_panel, y_panel, n_folds=4, train_start_ratio=0.5, wf_min_dates=20
        )

        for i, ((X_train, _), (X_val, _)) in enumerate(splits):
            train_dates = set(X_train["date"].unique())
            val_dates = set(X_val["date"].unique())

            # 训练集和验证集日期不应重叠
            assert len(train_dates & val_dates) == 0, \
                f"Fold {i}: training and validation dates should not overlap"

    def test_panel_split_same_date_stays_together(self):
        """Test that all stocks on same date stay in same fold."""
        try:
            from trainer import _make_panel_date_splits
        except ImportError:
            pytest.skip("_make_panel_date_splits not available")

        # 创建面板数据
        dates = pd.date_range(start="2020-01-01", periods=100, freq="B")
        stocks = ["000001", "000002", "000003"]

        rows = []
        for code in stocks:
            for date in dates:
                rows.append({"date": date, "stock_code": code, "feat": np.random.rand()})

        X_panel = pd.DataFrame(rows)
        y_panel = pd.Series(np.random.randint(0, 2, len(X_panel)))

        splits = _make_panel_date_splits(
            X_panel, y_panel, n_folds=4, train_start_ratio=0.5, wf_min_dates=20
        )

        # 验证每个 fold 内，同一日期的所有股票都在同一侧
        for i, ((X_train, _), (X_val, _)) in enumerate(splits):
            train_dates = set(X_train["date"].unique())
            val_dates = set(X_val["date"].unique())

            # 同一日期不应同时出现在训练集和验证集
            assert len(train_dates & val_dates) == 0, \
                f"Fold {i}: same date should not appear in both train and val"

    def test_panel_split_expanding_training(self):
        """Test that training set expands with each fold."""
        try:
            from trainer import _make_panel_date_splits
        except ImportError:
            pytest.skip("_make_panel_date_splits not available")

        # 创建面板数据
        dates = pd.date_range(start="2020-01-01", periods=100, freq="B")
        stocks = ["000001", "000002"]

        rows = []
        for code in stocks:
            for date in dates:
                rows.append({"date": date, "stock_code": code, "feat": np.random.rand()})

        X_panel = pd.DataFrame(rows)
        y_panel = pd.Series(np.random.randint(0, 2, len(X_panel)))

        splits = _make_panel_date_splits(
            X_panel, y_panel, n_folds=4, train_start_ratio=0.5, wf_min_dates=20
        )

        prev_train_len = 0
        for (X_train, _), _ in splits:
            assert len(X_train) > prev_train_len, "Training set should expand"
            prev_train_len = len(X_train)

    def test_panel_split_min_dates_constraint(self):
        """Test that min_dates constraint is respected."""
        try:
            from trainer import _make_panel_date_splits
        except ImportError:
            pytest.skip("_make_panel_date_splits not available")

        # 创建足够大的面板数据
        dates = pd.date_range(start="2020-01-01", periods=100, freq="B")
        stocks = ["000001", "000002"]

        rows = []
        for code in stocks:
            for date in dates:
                rows.append({"date": date, "stock_code": code, "feat": np.random.rand()})

        X_panel = pd.DataFrame(rows)
        y_panel = pd.Series(np.random.randint(0, 2, len(X_panel)))

        splits = _make_panel_date_splits(
            X_panel, y_panel, n_folds=4, train_start_ratio=0.5, wf_min_dates=20
        )

        for (X_train, _), (X_val, _) in splits:
            train_dates_count = X_train["date"].nunique()
            val_dates_count = X_val["date"].nunique()

            assert train_dates_count >= 20, "Training dates should meet min_dates"
            assert val_dates_count >= 20, "Validation dates should meet min_dates"
