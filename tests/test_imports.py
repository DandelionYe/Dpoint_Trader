# test_imports.py
"""基本导入测试：验证所有模块可以正常导入。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_core_imports():
    from dpoint.core.config import FeatureConfig, ModelConfig, RunConfig
    from dpoint.core.constants import COL_CLOSE, COL_DATE, COL_OPEN
    from dpoint.core.contract import DataContract, RunContract
    from dpoint.core.tasks import LabelSpec, resolve_label_spec
    from dpoint.core.utils import set_global_seed

    assert COL_CLOSE == "close_qfq"
    assert COL_DATE == "date"


def test_data_imports():
    from dpoint.data.basket_loader import discover_csv_files, load_basket_folder
    from dpoint.data.cleaner import DataReport, clean_ohlcv
    from dpoint.data.csv_loader import load_single_csv, standardize_columns
    from dpoint.data.excel_loader import load_stock_excel
    from dpoint.data.panel_builder import align_calendar, build_panel


def test_features_imports():
    from dpoint.features.cross_sectional import add_cross_sectional_features
    from dpoint.features.groups import add_all_features, add_momentum_features
    from dpoint.features.labeler import build_label
    from dpoint.features.pipeline import FeatureMeta, build_features_and_labels
    from dpoint.features.sequence_builder import PanelSequenceStore


def test_models_imports():
    from dpoint.models.registry import ALL_MODELS, DL_MODELS, ML_MODELS, make_model
    from dpoint.models.sklearn_models import create_sklearn_model
    from dpoint.models.torch_models import DL_MODEL_REGISTRY, TORCH_AVAILABLE
    from dpoint.models.trainer import predict_sklearn_model, train_sklearn_model


def test_splits_imports():
    from dpoint.splits.splitters import (
        SplitSpec,
        final_holdout_split,
        recommend_n_folds,
        walkforward_splits,
        walkforward_splits_with_embargo,
    )


def test_search_imports():
    from dpoint.search.engine import SearchState, random_search
    from dpoint.search.metrics import (
        METRIC_REGISTRY,
        get_metric_fn,
        pnl_metric,
        rank_ic_metric,
    )
    from dpoint.search.space import ALL_MODELS, sample_model_config, sample_trade_config


def test_cli_imports():
    from dpoint.cli.main import build_parser, main


def test_config_roundtrip():
    from dpoint.core.config import RunConfig

    cfg = RunConfig(mode="single", data_path="test.xlsx")
    d = cfg.to_dict()
    cfg2 = RunConfig.from_dict(d)
    assert cfg2.mode == "single"
    assert cfg2.data_path == "test.xlsx"
    assert cfg2.model.model_type == "lstm"
