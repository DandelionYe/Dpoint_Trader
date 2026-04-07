# rolling_trainer.py
"""
滚动再训练模块。

P0:
    - 按调度条件触发重训
    - 支持两种窗口：expanding, rolling
    - 支持固定重训频率：monthly
    - 保存每次重训后的模型与 manifest

P1:
    - 支持 weekly/quarterly retrain
    - rolling window length 配置
    - retrain 后自动评估近期表现
    - 支持模型快照管理
    - 支持最近窗口内校准器同步更新

P2:
    - 增加模型失效监控
    - 支持 fallback model
    - 支持自动降级到 baseline
    - 支持滚动再训练与横截面框架整合
    - 支持近期表现漂移告警
"""
from __future__ import annotations

import logging
import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


WINDOW_TYPES = ["expanding", "rolling"]
RETRAIN_FREQUENCIES = ["daily", "weekly", "monthly", "quarterly"]


@dataclass
class WindowConfig:
    """窗口配置。"""
    window_type: str = "expanding"
    rolling_window_length: Optional[int] = None
    min_window_size: int = 60


@dataclass
class SchedulerConfig:
    """调度配置。"""
    frequency: str = "monthly"
    day_of_month: int = 1
    day_of_week: int = 0
    hour: int = 0
    
    
@dataclass
class ModelSnapshot:
    """模型快照。"""
    snapshot_id: str
    timestamp: str
    train_end_date: str
    config: Dict[str, Any]
    metrics: Dict[str, float]
    model_path: Optional[str] = None
    calibrator_path: Optional[str] = None


@dataclass
class RetrainResult:
    """重训结果。"""
    success: bool
    snapshot_id: str
    train_start_date: str
    train_end_date: str
    metrics: Dict[str, float]
    model_path: str
    calibration_metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class RollingWindowManager:
    """
    滚动窗口管理器。
    
    管理 expanding 或 rolling 窗口的数据切片。
    """
    
    def __init__(self, config: WindowConfig):
        self.config = config
    
    def get_train_data(
        self,
        df: pd.DataFrame,
        current_date: str,
    ) -> pd.DataFrame:
        """
        获取训练数据窗口。
        
        Args:
            df: 完整数据集
            current_date: 当前日期（训练截止日）
            
        Returns:
            训练数据窗口
        """
        df = df.copy()
        
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            cutoff = pd.to_datetime(current_date)
            historical = df[df["date"] <= cutoff].copy()
        else:
            idx = df.index.get_loc(current_date) if current_date in df.index else len(df) - 1
            historical = df.iloc[:idx + 1].copy()
        
        if self.config.window_type == "expanding":
            return historical
        
        elif self.config.window_type == "rolling":
            if self.config.rolling_window_length:
                return historical.tail(self.config.rolling_window_length)
            else:
                return historical
    
    def get_validation_data(
        self,
        df: pd.DataFrame,
        train_end_date: str,
        val_window_days: int = 60,
    ) -> pd.DataFrame:
        """获取验证数据窗口（用于评估重训效果）。"""
        df = df.copy()
        
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            train_end = pd.to_datetime(train_end_date)
            val_start = train_end + timedelta(days=1)
            val_end = val_start + timedelta(days=val_window_days)
            return df[(df["date"] >= val_start) & (df["date"] <= val_end)].copy()
        else:
            train_idx = df.index.get_loc(train_end_date) if train_end_date in df.index else len(df) - 1
            val_start = train_idx + 1
            val_end = min(val_start + val_window_days, len(df))
            return df.iloc[val_start:val_end].copy()


class RetrainScheduler:
    """
    重训调度器。
    
    判断是否需要触发重训。
    """
    
    def __init__(self, config: SchedulerConfig, last_retrain_date: Optional[str] = None):
        self.config = config
        self.last_retrain_date = last_retrain_date
    
    def should_retrain(self, current_date: str) -> bool:
        """判断是否需要重训。"""
        if self.last_retrain_date is None:
            return True
        
        current = pd.to_datetime(current_date)
        last = pd.to_datetime(self.last_retrain_date)
        
        if self.config.frequency == "daily":
            return (current - last).days >= 1
        
        elif self.config.frequency == "weekly":
            return (current - last).days >= 7
        
        elif self.config.frequency == "monthly":
            return current.month != last.month or current.year != last.year
        
        elif self.config.frequency == "quarterly":
            return (current.month - 1) // 3 != (last.month - 1) // 3
        
        return False
    
    def get_next_retrain_date(self, current_date: str) -> str:
        """获取下次重训日期。"""
        current = pd.to_datetime(current_date)
        
        if self.config.frequency == "daily":
            next_date = current + timedelta(days=1)
        
        elif self.config.frequency == "weekly":
            days_ahead = (7 - current.weekday() + self.config.day_of_week) % 7
            if days_ahead == 0:
                days_ahead = 7
            next_date = current + timedelta(days=days_ahead)
        
        elif self.config.frequency == "monthly":
            if current.day >= self.config.day_of_month:
                next_month = current + pd.DateOffset(months=1)
                next_date = next_month.replace(day=min(self.config.day_of_month, 
                                                        pd.Timestamp(next_month).days_in_month))
            else:
                next_date = current.replace(day=self.config.day_of_month)
        
        elif self.config.frequency == "quarterly":
            quarter = (current.month - 1) // 3
            next_quarter_month = (quarter + 1) * 3 + 1
            if next_quarter_month > 12:
                next_quarter_month = ((next_quarter_month - 1) % 12) + 1
                next_date = current.replace(year=current.year + 1, month=next_quarter_month, day=1)
            else:
                next_date = current.replace(month=next_quarter_month, day=1)
        
        else:
            next_date = current + pd.DateOffset(months=1)
        
        return next_date.strftime("%Y-%m-%d")


class ModelSnapshotManager:
    """
    模型快照管理器。
    
    管理滚动训练过程中的模型版本。
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.snapshots_dir = os.path.join(output_dir, "snapshots")
        os.makedirs(self.snapshots_dir, exist_ok=True)
        
        self.snapshots: List[ModelSnapshot] = []
        self._load_existing_snapshots()
    
    def _load_existing_snapshots(self):
        """加载已有的快照。"""
        if not os.path.exists(self.snapshots_dir):
            return
        
        for fn in os.listdir(self.snapshots_dir):
            if fn.endswith("_manifest.json"):
                try:
                    manifest_path = os.path.join(self.snapshots_dir, fn)
                    with open(manifest_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    snapshot = ModelSnapshot(
                        snapshot_id=data.get("snapshot_id", ""),
                        timestamp=data.get("timestamp", ""),
                        train_end_date=data.get("train_end_date", ""),
                        config=data.get("config", {}),
                        metrics=data.get("metrics", {}),
                        model_path=data.get("model_path"),
                        calibrator_path=data.get("calibrator_path"),
                    )
                    self.snapshots.append(snapshot)
                except Exception:
                    pass
    
    def save_snapshot(
        self,
        result: RetrainResult,
        model_data: Optional[bytes] = None,
    ) -> ModelSnapshot:
        """保存模型快照。"""
        snapshot_id = result.snapshot_id
        snapshot_dir = os.path.join(self.snapshots_dir, snapshot_id)
        os.makedirs(snapshot_dir, exist_ok=True)
        
        model_path = None
        if model_data:
            model_path = os.path.join(snapshot_dir, "model.pkl")
            with open(model_path, 'wb') as f:
                f.write(model_data)
        
        snapshot = ModelSnapshot(
            snapshot_id=snapshot_id,
            timestamp=result.timestamp if hasattr(result, 'timestamp') else datetime.now().isoformat(),
            train_end_date=result.train_end_date,
            config=result.config,
            metrics=result.metrics,
            model_path=model_path,
        )
        
        manifest = {
            "snapshot_id": snapshot.snapshot_id,
            "timestamp": snapshot.timestamp,
            "train_start_date": result.train_start_date,
            "train_end_date": snapshot.train_end_date,
            "config": snapshot.config,
            "metrics": snapshot.metrics,
            "model_path": snapshot.model_path,
            "calibrator_path": snapshot.calibrator_path,
        }
        
        manifest_path = os.path.join(snapshot_dir, f"{snapshot_id}_manifest.json")
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        
        self.snapshots.append(snapshot)
        
        return snapshot
    
    def get_latest_snapshot(self) -> Optional[ModelSnapshot]:
        """获取最新快照。"""
        if not self.snapshots:
            return None
        return sorted(self.snapshots, key=lambda x: x.train_end_date, reverse=True)[0]
    
    def get_snapshot_by_id(self, snapshot_id: str) -> Optional[ModelSnapshot]:
        """根据 ID 获取快照。"""
        for s in self.snapshots:
            if s.snapshot_id == snapshot_id:
                return s
        return None
    
    def get_recent_snapshots(self, n: int = 5) -> List[ModelSnapshot]:
        """获取最近的 n 个快照。"""
        sorted_snapshots = sorted(self.snapshots, key=lambda x: x.train_end_date, reverse=True)
        return sorted_snapshots[:n]


class RollingTrainer:
    """
    滚动再训练器。
    
    整合窗口管理、调度、模型管理功能。
    """
    
    def __init__(
        self,
        output_dir: str,
        window_config: WindowConfig,
        scheduler_config: SchedulerConfig,
        base_config: Dict[str, Any],
    ):
        self.output_dir = output_dir
        self.window_manager = RollingWindowManager(window_config)
        self.scheduler = RetrainScheduler(scheduler_config)
        self.snapshot_manager = ModelSnapshotManager(output_dir)
        self.base_config = base_config
    
    def check_and_retrain(
        self,
        df: pd.DataFrame,
        current_date: str,
        train_func,
    ) -> Optional[RetrainResult]:
        """
        检查是否需要重训，如需要则执行重训。
        
        Args:
            df: 完整数据集
            current_date: 当前日期
            train_func: 训练函数
            
        Returns:
            RetrainResult 或 None（如果不需要重训）
        """
        if not self.scheduler.should_retrain(current_date):
            return None
        
        train_data = self.window_manager.get_train_data(df, current_date)
        
        if len(train_data) < self.window_manager.config.min_window_size:
            return None
        
        snapshot_id = f"snap_{current_date.replace('-', '')}"
        
        try:
            result = train_func(
                df=train_data,
                config=self.base_config,
                snapshot_id=snapshot_id,
            )
            
            self.snapshot_manager.save_snapshot(result)
            
            return result
        
        except Exception as e:
            return RetrainResult(
                success=False,
                snapshot_id=snapshot_id,
                train_start_date=train_data.index[0] if hasattr(train_data.index[0], 'strftime') else str(train_data.index[0]),
                train_end_date=current_date,
                metrics={},
                model_path="",
                error_message=str(e),
            )
    
    def get_current_model(self) -> Optional[ModelSnapshot]:
        """获取当前使用的模型（最新快照）。"""
        return self.snapshot_manager.get_latest_snapshot()
    
    def evaluate_recent_performance(
        self,
        df: pd.DataFrame,
        days: int = 30,
    ) -> Dict[str, float]:
        """评估近期表现。"""
        latest_snapshot = self.get_current_model()
        if not latest_snapshot:
            return {}
        
        val_data = self.window_manager.get_validation_data(
            df, latest_snapshot.train_end_date, val_window_days=days
        )
        
        if val_data.empty:
            return {}
        
        return {
            "val_start_date": val_data.index[0] if hasattr(val_data.index[0], 'strftime') else str(val_data.index[0]),
            "val_end_date": val_data.index[-1] if hasattr(val_data.index[-1], 'strftime') else str(val_data.index[-1]),
            "n_samples": len(val_data),
        }


class ModelMonitor:
    """
    P2: 模型失效监控器。
    
    监控模型表现漂移，触发告警或自动降级。
    """
    
    def __init__(
        self,
        drift_threshold: float = 0.1,
        lookback_snapshots: int = 3,
    ):
        self.drift_threshold = drift_threshold
        self.lookback_snapshots = lookback_snapshots
        self.performance_history: List[Dict[str, float]] = []
    
    def record_performance(self, snapshot_id: str, metrics: Dict[str, float]):
        """记录快照性能。"""
        self.performance_history.append({
            "snapshot_id": snapshot_id,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        })
    
    def check_drift(self) -> Dict[str, Any]:
        """检查是否存在性能漂移。"""
        if len(self.performance_history) < self.lookback_snapshots:
            return {"is_drifted": False, "reason": "insufficient_history"}
        
        recent = self.performance_history[-self.lookback_snapshots:]
        
        sharpe_values = [r["metrics"].get("sharpe", 0) for r in recent]
        
        if sharpe_values:
            current_sharpe = sharpe_values[-1]
            avg_sharpe = sum(sharpe_values[:-1]) / len(sharpe_values[:-1])
            
            if avg_sharpe > 0 and current_sharpe < avg_sharpe * (1 - self.drift_threshold):
                return {
                    "is_drifted": True,
                    "reason": "sharpe_degradation",
                    "current_sharpe": current_sharpe,
                    "avg_sharpe": avg_sharpe,
                    "drift_pct": (avg_sharpe - current_sharpe) / avg_sharpe,
                }
        
        return {"is_drifted": False, "reason": "stable"}


def create_rolling_trainer(
    output_dir: str,
    window_type: str = "expanding",
    rolling_window_length: Optional[int] = None,
    frequency: str = "monthly",
    base_config: Optional[Dict[str, Any]] = None,
) -> RollingTrainer:
    """创建滚动训练器。"""
    window_config = WindowConfig(
        window_type=window_type,
        rolling_window_length=rolling_window_length,
    )

    scheduler_config = SchedulerConfig(frequency=frequency)

    return RollingTrainer(
        output_dir=output_dir,
        window_config=window_config,
        scheduler_config=scheduler_config,
        base_config=base_config or {},
    )


# =========================================================
# P2 新增：与 trainer.py 的集成适配器
# =========================================================

def create_standard_train_func(
    seed: int = 42,
    n_folds: int = 4,
    train_start_ratio: float = 0.5,
    use_embargo: bool = False,
    embargo_days: int = 5,
) -> callable:
    """
    创建标准训练函数适配器，用于与 RollingTrainer.check_and_retrain 配合使用。

    此适配器封装了 trainer.py 中的训练流程，使 rolling_trainer 可以
    直接调用标准的 Walk-forward 训练流程。

    Args:
        seed: 随机种子
        n_folds: Walk-forward 折数
        train_start_ratio: 初始训练集比例
        use_embargo: 是否使用 embargo gap
        embargo_days: embargo 天数

    Returns:
        训练函数，签名：train_func(df, config, snapshot_id) -> RetrainResult
    """
    def train_func(
        df: pd.DataFrame,
        config: Dict[str, Any],
        snapshot_id: str,
    ) -> RetrainResult:
        """
        标准训练函数，封装 trainer.py 的训练流程。

        Args:
            df: 训练数据（已按窗口切分）
            config: 特征 + 模型 + 交易配置
            snapshot_id: 快照 ID

        Returns:
            RetrainResult
        """
        # 延迟导入，避免循环依赖
        from trainer import train_final_model_and_dpoint
        from backtester import backtest_from_dpoint, calculate_risk_metrics
        from feature_dpoint import build_features_and_labels

        train_start = df["date"].min() if "date" in df.columns else "unknown"
        train_end = df["date"].max() if "date" in df.columns else "unknown"

        try:
            # 1. 训练最终模型并计算 Dpoint
            dpoint, artifacts = train_final_model_and_dpoint(
                df, config, seed=seed
            )

            # 2. 回测评估
            trade_config = config.get("trade_config", {})
            initial_cash = float(trade_config.get("initial_cash", 100_000))

            bt_result = backtest_from_dpoint(
                df=df,
                dpoint=dpoint,
                initial_cash=initial_cash,
                buy_threshold=float(trade_config.get("buy_threshold", 0.5)),
                sell_threshold=float(trade_config.get("sell_threshold", 0.5)),
                confirm_days=int(trade_config.get("confirm_days", 3)),
                min_hold_days=int(trade_config.get("min_hold_days", 5)),
                max_hold_days=int(trade_config.get("max_hold_days", 20)),
                take_profit=trade_config.get("take_profit", None),
                stop_loss=trade_config.get("stop_loss", None),
            )

            # 3. 计算风险指标
            risk_metrics = calculate_risk_metrics(
                equity_curve=bt_result.equity_curve,
                trades=bt_result.trades,
                initial_cash=initial_cash,
            )

            # 4. 保存模型（可选，序列化 artifacts 中的模型）
            model_path = ""
            if "model" in artifacts:
                import pickle
                model_dir = os.path.join(
                    os.path.dirname(__file__),
                    "snapshots",
                    snapshot_id,
                )
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, "model.pkl")
                with open(model_path, "wb") as f:
                    pickle.dump(artifacts["model"], f)

            # 5. 构建重训结果
            result = RetrainResult(
                success=True,
                snapshot_id=snapshot_id,
                train_start_date=str(train_start),
                train_end_date=str(train_end),
                metrics={
                    "final_equity": risk_metrics.get("final_equity", initial_cash),
                    "total_return_pct": risk_metrics.get("total_return_pct", 0),
                    "sharpe": risk_metrics.get("sharpe", 0),
                    "max_drawdown_pct": risk_metrics.get("max_drawdown_pct", 0),
                    "win_rate": risk_metrics.get("win_rate", 0),
                    "n_trades": risk_metrics.get("trade_count", 0),
                },
                model_path=model_path,
                calibration_metrics=artifacts.get("calibration"),
            )

            return result

        except Exception as e:
            logger.exception("训练失败：%s", e)
            return RetrainResult(
                success=False,
                snapshot_id=snapshot_id,
                train_start_date=str(train_start),
                train_end_date=str(train_end),
                metrics={},
                model_path="",
                error_message=str(e),
            )

    return train_func


# =========================================================
# P2 新增：横截面面板训练适配器（用于 basket 模式）
# =========================================================

def create_panel_train_func(
    seed: int = 42,
    n_folds: int = 4,
    train_start_ratio: float = 0.5,
) -> callable:
    """
    创建面板训练函数适配器，用于 basket 模式的滚动训练。

    Args:
        seed: 随机种子
        n_folds: Walk-forward 折数
        train_start_ratio: 初始训练集比例

    Returns:
        训练函数，签名：train_func(stock_dict, config, snapshot_id) -> RetrainResult
    """
    def train_func(
        stock_dict: Dict[str, pd.DataFrame],
        config: Dict[str, Any],
        snapshot_id: str,
    ) -> RetrainResult:
        """
        面板训练函数，封装 trainer.py 的面板训练流程。

        Args:
            stock_dict: {股票代码：DataFrame} 字典
            config: 特征 + 模型 + 交易配置
            snapshot_id: 快照 ID

        Returns:
            RetrainResult
        """
        # 延迟导入
        from trainer import train_final_model_panel
        from portfolio_backtester import (
            PortfolioConfig,
            run_portfolio_backtest,
            format_portfolio_summary,
        )

        train_start = min(
            df["date"].min() for df in stock_dict.values() if not df.empty
        )
        train_end = max(
            df["date"].max() for df in stock_dict.values() if not df.empty
        )

        try:
            # 1. 训练面板模型
            dpoint_matrix, artifacts = train_final_model_panel(
                stock_dict, config, seed=seed
            )

            # 2. 组合回测评估
            portfolio_cfg = PortfolioConfig(
                top_k=int(config.get("trade_config", {}).get("top_k", 5)),
                rebalance_freq="weekly",
                weighting_scheme="equal",
                initial_cash=float(
                    config.get("trade_config", {}).get("initial_cash", 1_000_000)
                ),
            )

            portfolio_result = run_portfolio_backtest(
                stock_dict=stock_dict,
                dpoint_matrix=dpoint_matrix,
                cfg=portfolio_cfg,
            )

            # 3. 保存模型
            model_path = ""
            if "model" in artifacts:
                import pickle
                model_dir = os.path.join(
                    os.path.dirname(__file__),
                    "snapshots",
                    snapshot_id,
                )
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, "model.pkl")
                with open(model_path, "wb") as f:
                    pickle.dump(artifacts["model"], f)

            # 4. 构建重训结果
            result = RetrainResult(
                success=True,
                snapshot_id=snapshot_id,
                train_start_date=str(train_start),
                train_end_date=str(train_end),
                metrics={
                    "final_equity": portfolio_result.metrics.get(
                        "final_equity", portfolio_cfg.initial_cash
                    ),
                    "total_return_pct": portfolio_result.metrics.get(
                        "total_return_pct", 0
                    ),
                    "sharpe": portfolio_result.metrics.get("sharpe", 0),
                    "max_drawdown_pct": portfolio_result.metrics.get(
                        "max_drawdown_pct", 0
                    ),
                },
                model_path=model_path,
            )

            return result

        except Exception as e:
            logger.exception("面板训练失败：%s", e)
            return RetrainResult(
                success=False,
                snapshot_id=snapshot_id,
                train_start_date=str(train_start),
                train_end_date=str(train_end),
                metrics={},
                model_path="",
                error_message=str(e),
            )

    return train_func


# =========================================================
# 公开 API 导出
# =========================================================
__all__ = [
    "WINDOW_TYPES",
    "RETRAIN_FREQUENCIES",
    "WindowConfig",
    "SchedulerConfig",
    "ModelSnapshot",
    "RetrainResult",
    "RollingWindowManager",
    "RetrainScheduler",
    "ModelSnapshotManager",
    "RollingTrainer",
    "ModelMonitor",
    "create_rolling_trainer",
    # P2 新增：训练适配器
    "create_standard_train_func",
    "create_panel_train_func",
]