# test_imports.py
"""基本导入测试：验证所有模块可以正常导入。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_core_imports():
    from dpoint.core.constants import COL_CLOSE, COL_DATE, COL_OPEN
    from dpoint.core.config import FeatureConfig, ModelConfig, RunConfig
    from dpoint.core.tasks import LabelSpec, resolve_label_spec
    from dpoint.core.contract import DataContract, RunContract
    from dpoint.core.utils import set_global_seed
    assert COL_CLOSE == "close_qfq"
    assert COL_DATE == "date"


def test_data_imports():
    from dpoint.data.cleaner import clean_ohlcv, DataReport
    from dpoint.data.excel_loader import load_stock_excel
    from dpoint.data.csv_loader import load_single_csv, standardize_columns
    from dpoint.data.panel_builder import build_panel, align_calendar
    from dpoint.data.basket_loader import load_basket_folder, discover_csv_files


def test_features_imports():
    from dpoint.features.groups import add_all_features, add_momentum_features
    from dpoint.features.cross_sectional import add_cross_sectional_features
    from dpoint.features.labeler import build_label
    from dpoint.features.sequence_builder import PanelSequenceStore
    from dpoint.features.pipeline import build_features_and_labels, FeatureMeta


def test_models_imports():
    from dpoint.models.registry import make_model, ALL_MODELS, ML_MODELS, DL_MODELS
    from dpoint.models.sklearn_models import create_sklearn_model
    from dpoint.models.torch_models import TORCH_AVAILABLE, DL_MODEL_REGISTRY
    from dpoint.models.trainer import train_sklearn_model, predict_sklearn_model


def test_splits_imports():
    from dpoint.splits.splitters import walkforward_splits, walkforward_splits_with_embargo
    from dpoint.splits.splitters import final_holdout_split, recommend_n_folds, SplitSpec


def test_search_imports():
    from dpoint.search.metrics import pnl_metric, rank_ic_metric, get_metric_fn, METRIC_REGISTRY
    from dpoint.search.space import sample_model_config, sample_trade_config, ALL_MODELS
    from dpoint.search.engine import random_search, SearchState


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
