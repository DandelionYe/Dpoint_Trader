# config.py
"""
统一配置 dataclass 体系。
合并自两个项目的参数设计，用子命令区分 single/basket 模式。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class FeatureConfig:
    """特征工程配置。"""

    windows: list[int] = field(default_factory=lambda: [5, 10, 20])
    use_momentum: bool = True
    use_volatility: bool = True
    use_volume: bool = True
    use_candle: bool = True
    use_turnover: bool = True
    use_ta_indicators: bool = True
    include_cross_section: bool = True  # 篮子模式独占
    vol_metric: str = "std"  # std / mad
    liq_transform: str = "ratio"  # ratio / zscore

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelConfig:
    """模型配置。"""

    model_type: str = "lstm"  # logreg / sgd / xgb / mlp / lstm / gru / cnn / transformer
    hidden_dim: int = 64
    num_layers: int = 2
    dropout_rate: float = 0.3
    learning_rate: float = 1e-3
    batch_size: int = 256
    epochs: int = 100
    patience: int = 10
    seq_len: int = 10
    bidirectional: bool = False
    # XGBoost 专用
    n_estimators: int = 200
    max_depth: int = 3
    # sklearn 专用
    C: float = 1.0
    penalty: str = "l2"
    alpha: float = 1e-4

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TradeConfig:
    """交易参数配置。"""

    buy_threshold: float = 0.52
    sell_threshold: float = 0.48
    confirm_days: int = 1
    max_hold_days: int = 20
    take_profit: float = 0.0
    stop_loss: float = 0.0
    commission_rate_buy: float = 0.0003
    commission_rate_sell: float = 0.0013
    slippage_bps: float = 20.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SearchConfig:
    """搜索配置。"""

    n_candidates: int = 100
    n_rounds: int = 4
    explore_ratio: float = 0.3
    metric: str = "pnl"  # pnl / rank_ic
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SplitConfig:
    """样本划分配置。"""

    mode: str = "wf"  # wf / wf_embargo / nested_wf
    n_folds: int = 4
    train_start_ratio: float = 0.5
    min_rows: int = 80
    embargo_days: int = 5
    holdout_ratio: float = 0.15
    holdout_gap_days: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PortfolioConfig:
    """组合构建配置（篮子模式独占）。"""

    top_k: int = 5
    weighting: str = "equal"  # equal / score / vol_inv
    max_weight: float = 0.20
    rebalance_freq: str = "daily"  # daily / weekly / monthly
    cash_buffer: float = 0.05

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunConfig:
    """统一运行配置。"""

    mode: str = "single"  # single / basket
    data_path: str = ""  # 单股模式：Excel 文件路径
    basket_path: str = ""  # 篮子模式：basket 目录路径
    output_dir: str = "output"
    seed: int = 42
    n_jobs: int = -1
    device: str = "auto"  # auto / cpu / cuda
    use_amp: bool = True

    feature: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trade: TradeConfig = field(default_factory=TradeConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunConfig":
        feature = FeatureConfig(**d.pop("feature", {}))
        model = ModelConfig(**d.pop("model", {}))
        trade = TradeConfig(**d.pop("trade", {}))
        search = SearchConfig(**d.pop("search", {}))
        split = SplitConfig(**d.pop("split", {}))
        portfolio = PortfolioConfig(**d.pop("portfolio", {}))
        return cls(
            feature=feature,
            model=model,
            trade=trade,
            search=search,
            split=split,
            portfolio=portfolio,
            **d,
        )
