# contract.py
"""
实验契约系统：确保 Continue 模式下的数据/特征/训练契约兼容性。
来自 DpointTrader_deeplearning_Ver1.0/experiment_contract.py。
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class DataContract:
    data_hash: str
    date_min: str
    date_max: str
    n_rows: int
    n_tickers: int


@dataclass
class FeatureContract:
    feature_names: list[str]
    feature_schema_hash: str
    include_cross_section: bool
    seq_len: Optional[int]
    feature_config_hash: str


@dataclass
class TrainingContract:
    task_type: str
    label_mode: str
    label_horizon_days: int
    model_type: str
    target_version: str


@dataclass
class RunContract:
    data: DataContract
    feature: FeatureContract
    training: TrainingContract


class ContinueCompatibilityError(ValueError):
    pass


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def compute_data_hash(df: pd.DataFrame) -> str:
    """计算 DataFrame 的稳定哈希值。"""
    sort_cols = [c for c in ["date", "ticker"] if c in df.columns]
    sorted_df = df.sort_values(sort_cols).reset_index(drop=True) if sort_cols else df
    return hashlib.sha256(
        pd.util.hash_pandas_object(sorted_df).values.tobytes()
    ).hexdigest()


def compute_feature_schema_hash(
    feature_names: list[str], extra: Optional[Dict[str, Any]] = None,
) -> str:
    payload = {"feature_names": list(feature_names), "extra": extra or {}}
    return _stable_hash(payload)


def compute_feature_config_hash(feature_cfg: Dict[str, Any]) -> str:
    return _stable_hash(feature_cfg)


def contract_to_dict(contract: RunContract) -> Dict[str, Any]:
    return asdict(contract)


def build_run_contract(
    panel_df: pd.DataFrame,
    *,
    feature_meta: Any,
    model_config: Dict[str, Any],
    label_mode: str = "binary_next_close_up",
    task_type: str = "",
    horizon_days: int = 1,
) -> RunContract:
    """从运行时数据构建契约。"""
    sort_cols = [c for c in ["date", "ticker"] if c in panel_df.columns]
    sorted_df = panel_df.sort_values(sort_cols).reset_index(drop=True) if sort_cols else panel_df

    date_min, date_max = "", ""
    if "date" in sorted_df.columns and not sorted_df.empty:
        date_min = str(pd.to_datetime(sorted_df["date"]).min())
        date_max = str(pd.to_datetime(sorted_df["date"]).max())

    feature_names = list(getattr(feature_meta, "feature_names", []))
    feature_params = dict(getattr(feature_meta, "params", {}))
    include_cross_section = bool(feature_params.get("include_cross_section", True))
    model_params = dict(model_config.get("model_params", {}))
    seq_len = model_params.get("seq_len")
    task_type = task_type or ("binary_classification" if "binary" in label_mode else "regression")

    return RunContract(
        data=DataContract(
            data_hash=compute_data_hash(sorted_df),
            date_min=date_min,
            date_max=date_max,
            n_rows=int(len(sorted_df)),
            n_tickers=int(sorted_df["ticker"].nunique()) if "ticker" in sorted_df.columns else 0,
        ),
        feature=FeatureContract(
            feature_names=feature_names,
            feature_schema_hash=compute_feature_schema_hash(
                feature_names, extra={"seq_len": seq_len, "include_cross_section": include_cross_section},
            ),
            include_cross_section=include_cross_section,
            seq_len=int(seq_len) if seq_len is not None else None,
            feature_config_hash=compute_feature_config_hash(feature_params),
        ),
        training=TrainingContract(
            task_type=task_type,
            label_mode=label_mode,
            label_horizon_days=max(1, horizon_days),
            model_type=str(model_config.get("model_type", "")),
            target_version="1",
        ),
    )
