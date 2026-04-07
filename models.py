# models.py
"""
模型构建与训练模块（合并自 dl_model_builder.py + model_builder.py）。

支持：
- PyTorch 深度学习模型：MLP, LSTM, GRU, CNN1D, Transformer
- sklearn 模型：LogisticRegression, SGDClassifier, XGBClassifier

主要功能：
- 模型构建（make_model）
- 模型训练（train_pytorch_model）
- 模型预测（predict_dpoint, predict_pytorch_model）
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TORCH_AVAILABLE = False
TORCH_IMPORT_ERROR: Optional[Exception] = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.amp import GradScaler, autocast
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except Exception as exc:  # pragma: no cover - exercised indirectly via import tests
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]
    GradScaler = None  # type: ignore[assignment]
    autocast = None  # type: ignore[assignment]
    DataLoader = Any  # type: ignore[assignment]
    TensorDataset = Any  # type: ignore[assignment]
    TORCH_IMPORT_ERROR = exc

__all__ = [
    # 深度学习模型类
    "MLP", "LSTM", "GRU", "CNN1D", "Transformer",
    # 设备检测
    "_get_device",
    # GPU 显存检测与优化
    "get_gpu_memory_info", "auto_select_batch_size",
    # PyTorch 训练与预测
    "train_pytorch_model", "predict_pytorch_model", "create_sequence_dataset",
    # sklearn 模型构建
    "make_model", "predict_dpoint",
]


class _CpuFallbackDevice:
    """Torch 不可用时的最小设备占位对象。"""

    type = "cpu"

    def __str__(self) -> str:
        return self.type


def _require_torch(feature: str) -> None:
    """在真正使用 PyTorch 路径时再抛出清晰错误。"""
    if TORCH_AVAILABLE:
        return

    detail = ""
    if TORCH_IMPORT_ERROR is not None:
        detail = f" Original import error: {TORCH_IMPORT_ERROR!r}"
    raise RuntimeError(
        f"{feature} requires PyTorch, but PyTorch is not installed or failed to import.{detail}"
    )


# =========================================================
# 设备检测
# =========================================================
def _get_device() -> torch.device:
    """自动检测并返回可用的计算设备（CUDA 或 CPU）。"""
    if not TORCH_AVAILABLE:
        return _CpuFallbackDevice()  # type: ignore[return-value]
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# GPU 显存检测与自适应 Batch Size 选择
# =========================================================

def get_gpu_memory_info(device: Optional[torch.device] = None) -> Dict[str, float]:
    """
    获取 GPU 显存信息。

    Args:
        device: PyTorch 设备，None 时自动检测

    Returns:
        包含以下键的字典:
            - total: 总显存 (MB)
            - allocated: 已分配显存 (MB)
            - reserved: 已预留显存 (MB)
            - free: 可用显存 (MB)
            - available: 可用显存 (MB)（与 free 相同）
    """
    if not TORCH_AVAILABLE:
        return {"total": 0, "allocated": 0, "reserved": 0, "free": 0, "available": 0}

    if device is None:
        device = _get_device()

    if device.type != "cuda":
        return {"total": 0, "allocated": 0, "reserved": 0, "free": 0, "available": 0}

    try:
        # 触发一次 CUDA 操作以确保初始化
        torch.cuda.current_device()

        total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        free = total - allocated - reserved

        return {
            "total": round(total, 2),
            "allocated": round(allocated, 2),
            "reserved": round(reserved, 2),
            "free": round(free, 2),
            "available": round(free, 2),
        }
    except Exception:
        # CUDA 未初始化或其他错误
        return {"total": 0, "allocated": 0, "reserved": 0, "free": 0, "available": 0}


def auto_select_batch_size(
    device: Optional[torch.device] = None,
    model_type: str = "mlp",
    input_dim: int = 64,
    seq_len: int = 20,
    memory_fraction: float = 0.7,
    mode: str = "auto",
    manual_batch_size: Optional[int] = None,
) -> int:
    """
    根据 GPU 显存自动选择最优 batch_size。

    **显存估算模型：**

    对于不同模型类型，使用经验公式估算每个样本的显存占用：

    - MLP:    base_mem = input_dim × 4 bytes (float32) × 2 (activation + grad)
    - LSTM:   base_mem = input_dim × seq_len × 8 bytes
    - GRU:    base_mem = input_dim × seq_len × 6 bytes
    - CNN:    base_mem = input_dim × seq_len × 12 bytes
    - Transformer: base_mem = input_dim × seq_len × 16 bytes

    然后考虑：
    - 模型参数占用（hidden_dim^2 × 4 bytes）
    - 优化器状态（Adam: 参数 × 8 bytes）
    - 梯度（参数 × 4 bytes）
    - 预留缓冲（20%）

    Args:
        device: PyTorch 设备
        model_type: 模型类型 (mlp | lstm | gru | cnn | transformer)
        input_dim: 输入特征维度
        seq_len: 序列长度（LSTM/GRU/CNN/Transformer 需要）
        memory_fraction: 使用的显存比例（0.5-0.9，默认 0.7 保守）
        mode: 选择模式
            - "auto": 自动根据显存选择
            - "conservative": 保守模式（使用 50% 显存）
            - "standard": 标准模式（使用 70% 显存）
            - "aggressive": 激进模式（使用 85% 显存）
            - "manual": 使用 manual_batch_size
        manual_batch_size: 手动指定的 batch_size（mode="manual" 时使用）

    Returns:
        推荐的 batch_size

    Examples:
        >>> auto_select_batch_size(mode="auto", model_type="cnn", input_dim=28, seq_len=20)
        256  # 假设有 8GB 显存
    """
    # CPU 环境或无法检测时返回默认值
    if not TORCH_AVAILABLE:
        return manual_batch_size if manual_batch_size else 64

    if device is None:
        device = _get_device()

    if device.type != "cpu":
        try:
            torch.cuda.current_device()
        except Exception:
            # CUDA 设备存在但未初始化，返回默认值
            return manual_batch_size if manual_batch_size else 64

    # 手动模式
    if mode == "manual" and manual_batch_size is not None:
        return manual_batch_size

    # 获取显存信息
    mem_info = get_gpu_memory_info(device)
    total_mem = mem_info.get("total", 0)

    if total_mem <= 0:
        # 无法获取显存信息，返回默认值
        return manual_batch_size if manual_batch_size else 64

    # 根据模式调整显存使用比例
    if mode == "conservative":
        memory_fraction = 0.5
    elif mode == "standard":
        memory_fraction = 0.7
    elif mode == "aggressive":
        memory_fraction = 0.85

    # 估算每个样本的显存占用（MB）
    # 基础内存：输入数据 + 激活值 + 梯度
    bytes_per_float = 4  # float32

    if model_type == "mlp":
        # MLP: input → hidden → output
        # 每个样本：input_dim (输入) + hidden_dim (激活) + 1 (输出)
        base_mem_per_sample = (input_dim + 64 + 1) * bytes_per_float * 3  # ×3 for activation+grad+optimizer
    elif model_type == "lstm":
        # LSTM: 4 gates × hidden_dim × (input_dim + hidden_dim)
        hidden_dim = 64  # 假设值
        base_mem_per_sample = seq_len * (4 * hidden_dim * (input_dim + hidden_dim)) * bytes_per_float * 0.1  # 简化估算
    elif model_type == "gru":
        hidden_dim = 64
        base_mem_per_sample = seq_len * (3 * hidden_dim * (input_dim + hidden_dim)) * bytes_per_float * 0.1
    elif model_type == "cnn":
        # CNN: multiple conv filters
        num_filters = 64
        kernel_sizes = [2, 3, 5]
        base_mem_per_sample = (input_dim * seq_len * num_filters * len(kernel_sizes)) * bytes_per_float * 0.1
    elif model_type == "transformer":
        d_model = 64
        base_mem_per_sample = (input_dim * seq_len * d_model) * bytes_per_float * 0.2  # Transformer 更耗显存
    else:
        base_mem_per_sample = input_dim * bytes_per_float * 2

    # 可用显存（MB）
    available_mem = total_mem * memory_fraction

    # 计算理论最大 batch_size
    if base_mem_per_sample <= 0:
        return manual_batch_size if manual_batch_size else 64

    theoretical_batch = int(available_mem / (base_mem_per_sample / (1024 ** 2)))

    # 限制范围
    batch_size = max(16, min(theoretical_batch, 1024))

    # 对齐到 16 的倍数（优化 CUDA 核心利用率）
    batch_size = (batch_size // 16) * 16

    # 如果用户指定了 manual_batch_size，作为上限
    if manual_batch_size is not None:
        batch_size = min(batch_size, manual_batch_size)

    return batch_size


# =========================================================
# 序列数据构建工具
# =========================================================
def create_sequence_dataset(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    seq_len: int = 20,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
    shuffle: bool = True,
    num_workers: int = 2,
    prefetch_factor: int = 2,
) -> DataLoader:
    """
    将 DataFrame 转换为 PyTorch DataLoader，支持序列数据。

    Args:
        X: 特征 DataFrame (n_samples, n_features)
        y: 标签 Series（可选）
        seq_len: 序列长度（用于 LSTM/GRU/Transformer/CNN）
        batch_size: 批量大小
        device: 已弃用，保留以向后兼容
        shuffle: 是否打乱数据
        num_workers: 数据加载工作进程数，默认 2（0 表示单进程）
        prefetch_factor: 每个 worker 预取的 batch 数，默认 2

    Returns:
        DataLoader
    """
    _require_torch("create_sequence_dataset")
    _ = device

    X_tensor = torch.tensor(X.values, dtype=torch.float32)

    if y is not None:
        y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)
    else:
        y_tensor = None

    if seq_len > 1:
        X_seq, y_seq = _make_sequences(X_tensor, y_tensor, seq_len)
        if y_seq is not None:
            dataset = TensorDataset(X_seq, y_seq)
        else:
            dataset = TensorDataset(X_seq)
    else:
        if y_tensor is not None:
            dataset = TensorDataset(X_tensor, y_tensor)
        else:
            dataset = TensorDataset(X_tensor)

    # 数据加载优化：启用多进程预取，使 GPU 获得更平稳的数据供应
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )


def _make_sequences(
    X: torch.Tensor,
    y: Optional[torch.Tensor],
    seq_len: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    将时间序列数据转换为滑动窗口格式（向量化实现）。

    Args:
        X: 特征张量 (N, n_features)
        y: 标签张量 (N, 1)，可为 None
        seq_len: 序列长度

    Returns:
        X_seq: (N-seq_len+1, seq_len, n_features)
        y_seq: (N-seq_len+1, 1) 或 None
    """
    n_samples = X.shape[0]
    n_sequences = n_samples - seq_len + 1

    if n_sequences <= 0:
        raise ValueError(
            f"seq_len ({seq_len}) 超过样本数 ({n_samples})，无法构建序列。"
        )

    X_seq = X.unfold(0, seq_len, 1).permute(0, 2, 1).contiguous()
    y_seq = y[seq_len - 1:n_samples] if y is not None else None

    return X_seq, y_seq


def _logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    """将 raw logits 转为 [0, 1] 概率。"""
    return torch.sigmoid(logits)


# =========================================================
# PyTorch 模型定义
# =========================================================
if TORCH_AVAILABLE:
    class MLP(nn.Module):
        """简单的多层感知机，用于二分类任务（输出 raw logits）。"""

        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1, dropout_rate: float = 0.5):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout_rate)
            self.fc2 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x


    class LSTM(nn.Module):
        """LSTM（长短期记忆网络）用于时序二分类（输出 raw logits）。"""

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 64,
            num_layers: int = 2,
            output_dim: int = 1,
            dropout_rate: float = 0.3,
            bidirectional: bool = False,
            batch_first: bool = True,
        ):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.bidirectional = bidirectional
            self.batch_first = batch_first

            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=batch_first,
                dropout=dropout_rate if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )

            self.dropout = nn.Dropout(dropout_rate)
            lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
            self.fc = nn.Linear(lstm_output_dim, output_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            _, (h_n, _) = self.lstm(x)
            if self.bidirectional:
                last_output = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
            else:
                last_output = h_n[-1, :, :]
            x = self.dropout(last_output)
            x = self.fc(x)
            return x


    class GRU(nn.Module):
        """GRU（门控循环单元）用于时序二分类（输出 raw logits）。"""

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 64,
            num_layers: int = 2,
            output_dim: int = 1,
            dropout_rate: float = 0.3,
            bidirectional: bool = False,
            batch_first: bool = True,
        ):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.bidirectional = bidirectional

            self.gru = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=batch_first,
                dropout=dropout_rate if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )

            self.dropout = nn.Dropout(dropout_rate)
            gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
            self.fc = nn.Linear(gru_output_dim, output_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            _, h_n = self.gru(x)
            if self.bidirectional:
                last_output = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
            else:
                last_output = h_n[-1, :, :]
            x = self.dropout(last_output)
            x = self.fc(x)
            return x


    class CNN1D(nn.Module):
        """一维卷积神经网络（CNN）用于时序二分类（输出 raw logits）。"""

        def __init__(
            self,
            input_dim: int,
            seq_len: int,
            num_filters: int = 64,
            kernel_sizes: Optional[list] = None,
            output_dim: int = 1,
            dropout_rate: float = 0.3,
        ):
            super().__init__()

            if kernel_sizes is None:
                kernel_sizes = [2, 3, 5]

            self.convs = nn.ModuleList([
                nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=k, padding=k // 2)
                for k in kernel_sizes
            ])

            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout_rate)

            total_features = num_filters * len(kernel_sizes)
            self.fc = nn.Linear(total_features, output_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.transpose(1, 2)
            conv_outputs = []
            for conv in self.convs:
                conv_out = self.relu(conv(x))
                pooled = torch.max(conv_out, dim=2)[0]
                conv_outputs.append(pooled)
            x = torch.cat(conv_outputs, dim=1)
            x = self.dropout(x)
            x = self.fc(x)
            return x


    class Transformer(nn.Module):
        """Transformer Encoder 用于时序二分类（输出 raw logits）。"""

        def __init__(
            self,
            input_dim: int,
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 2,
            dim_feedforward: int = 128,
            dropout_rate: float = 0.1,
            output_dim: int = 1,
            max_seq_len: int = 100,
        ):
            super().__init__()

            self.d_model = d_model
            self.input_embedding = nn.Linear(input_dim, d_model)
            self.pos_encoder = PositionalEncoding(d_model, dropout_rate, max_seq_len)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout_rate,
                activation="gelu",
                batch_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            self.dropout = nn.Dropout(dropout_rate)
            self.fc = nn.Linear(d_model, output_dim)
            self._init_weights()

        def _init_weights(self):
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.input_embedding(x)
            x = self.pos_encoder(x)
            x = self.transformer_encoder(x)
            x = x.mean(dim=1)
            x = self.dropout(x)
            x = self.fc(x)
            return x


    class PositionalEncoding(nn.Module):
        """Transformer 位置编码。"""

        def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)

            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(1, max_len, d_model)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.pe[:, :x.size(1), :]
            return self.dropout(x)
else:
    class _TorchUnavailableModel:
        def __init__(self, *args: Any, **kwargs: Any):
            _require_torch("PyTorch model construction")


    class MLP(_TorchUnavailableModel):
        pass


    class LSTM(_TorchUnavailableModel):
        pass


    class GRU(_TorchUnavailableModel):
        pass


    class CNN1D(_TorchUnavailableModel):
        pass


    class Transformer(_TorchUnavailableModel):
        pass


    class PositionalEncoding(_TorchUnavailableModel):
        pass


# =========================================================
# PyTorch 训练函数
# =========================================================
# ✅ 新签名（X_val / y_val 可选，不传则退化为原行为）
def train_pytorch_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: Dict[str, Any],
    device: torch.device,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
) -> nn.Module:
    """
    训练 PyTorch 模型（支持 MLP/LSTM/GRU/CNN/Transformer）。

    **性能优化特性：**
        - 混合精度训练（AMP）：自动启用，减少显存占用，提高训练速度
        - 自适应 batch size：根据 GPU 显存自动调整（可选）
        - 梯度累积：模拟更大 batch size（可选）
        - 推理优化：预测时使用混合精度

    Args:
        X_train: 训练特征 DataFrame
        y_train: 训练标签 Series
        config: 模型配置字典
        device: 计算设备
        X_val: 可选验证集特征。传入时早停基于验证 loss，并保存验证 loss 最低时的
               权重作为最终模型；为 None 时退化为监控训练 loss（向后兼容）。
        y_val: 可选验证集标签，与 X_val 配套使用。

    Returns:
        训练好的模型
    """
    _require_torch("train_pytorch_model")
    model_type = str(config.get("model_type", "mlp")).lower()
    input_dim = X_train.shape[1]

    hidden_dim = int(config.get("hidden_dim", 64))
    d_model = int(config.get("d_model", 64))
    learning_rate = float(config.get("learning_rate", 0.001))
    epochs = int(config.get("epochs", 20))
    
    # Batch Size 配置（支持自适应）
    batch_size_mode = str(config.get("batch_size_mode", "manual"))  # auto | manual
    batch_size_manual = int(config.get("batch_size", 64))
    memory_fraction = float(config.get("gpu_memory_fraction", 0.7))  # 显存使用比例
    
    # 根据模式选择 batch_size
    if batch_size_mode == "auto":
        batch_size = auto_select_batch_size(
            device=device,
            model_type=model_type,
            input_dim=input_dim,
            seq_len=int(config.get("seq_len", 20)),
            memory_fraction=memory_fraction,
            mode="auto",
            manual_batch_size=batch_size_manual,
        )
        logger_info = f"自动选择 batch_size={batch_size} (GPU 显存模式)"
    else:
        batch_size = batch_size_manual
        logger_info = f"使用手动 batch_size={batch_size}"
    
    dropout_rate = float(config.get("dropout_rate", 0.3))
    seq_len = int(config.get("seq_len", 20))

    # 数据加载优化参数（可通过配置文件调整）
    num_workers = int(config.get("num_workers", 2))  # 数据加载工作进程数
    prefetch_factor = int(config.get("prefetch_factor", 2))  # 预取 batch 数

    # 打印 batch_size 选择信息
    try:
        import logging
        logger = logging.getLogger(__name__)
        logger.info("[GPU 优化] %s", logger_info)
        if device.type == "cuda":
            mem_info = get_gpu_memory_info(device)
            logger.info("[GPU 优化] GPU 显存：总计=%.2f MB, 可用=%.2f MB", mem_info['total'], mem_info['available'])
    except Exception:
        pass

    if model_type == "mlp":
        model = MLP(input_dim=input_dim, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        train_loader = create_sequence_dataset(
            X_train, y_train,
            seq_len=1,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
        val_loader = None
        if X_val is not None and y_val is not None:
            val_loader = create_sequence_dataset(
                X_val, y_val,
                seq_len=1,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
            )

    elif model_type == "lstm":
        num_layers = int(config.get("num_layers", 2))
        bidirectional = bool(config.get("bidirectional", False))
        model = LSTM(
            input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers,
            dropout_rate=dropout_rate, bidirectional=bidirectional,
        )
        train_loader = create_sequence_dataset(
            X_train, y_train,
            seq_len=seq_len,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
        val_loader = None
        if X_val is not None and y_val is not None:
            val_loader = create_sequence_dataset(
                X_val, y_val,
                seq_len=seq_len,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
            )

    elif model_type == "gru":
        num_layers = int(config.get("num_layers", 2))
        bidirectional = bool(config.get("bidirectional", False))
        model = GRU(
            input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers,
            dropout_rate=dropout_rate, bidirectional=bidirectional,
        )
        train_loader = create_sequence_dataset(
            X_train, y_train,
            seq_len=seq_len,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
        val_loader = None
        if X_val is not None and y_val is not None:
            val_loader = create_sequence_dataset(
                X_val, y_val,
                seq_len=seq_len,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
            )

    elif model_type == "cnn":
        num_filters = int(config.get("num_filters", 64))
        kernel_sizes = config.get("kernel_sizes", [2, 3, 5])
        model = CNN1D(
            input_dim=input_dim, seq_len=seq_len, num_filters=num_filters,
            kernel_sizes=kernel_sizes, dropout_rate=dropout_rate,
        )
        train_loader = create_sequence_dataset(
            X_train, y_train,
            seq_len=seq_len,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
        val_loader = None
        if X_val is not None and y_val is not None:
            val_loader = create_sequence_dataset(
                X_val, y_val,
                seq_len=seq_len,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
            )

    elif model_type == "transformer":
        nhead = int(config.get("nhead", 4))
        num_layers = int(config.get("num_layers", 2))
        dim_feedforward = int(config.get("dim_feedforward", 128))
        model = Transformer(
            input_dim=input_dim, d_model=d_model, nhead=nhead, num_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout_rate=dropout_rate,
        )
        train_loader = create_sequence_dataset(
            X_train, y_train,
            seq_len=seq_len,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
        val_loader = None
        if X_val is not None and y_val is not None:
            val_loader = create_sequence_dataset(
                X_val, y_val,
                seq_len=seq_len,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
            )

    else:
        raise ValueError(f"未知的 model_type: {model_type}")

    model = model.to(device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )

    use_amp: bool = (device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)

    best_monitor_loss = float("inf")
    best_state_dict = None
    patience_counter = 0
    max_patience = 10

    for epoch in range(epochs):
        # ---------- 训练阶段 ----------
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            if len(batch) < 2:
                continue
            X_batch, y_batch = batch
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            num_batches += 1

        train_loss = total_loss / max(num_batches, 1)

        # ---------- 验证阶段（有 val_loader 时才执行）----------
        if val_loader is not None:
            model.eval()
            val_total_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) < 2:
                        continue
                    X_batch, y_batch = batch
                    X_batch = X_batch.to(device, non_blocking=True)
                    y_batch = y_batch.to(device, non_blocking=True)
                    with autocast(device_type=device.type, enabled=use_amp):
                        logits = model(X_batch)
                        loss = criterion(logits, y_batch)
                    val_total_loss += loss.item()
                    val_batches += 1
            monitor_loss = val_total_loss / max(val_batches, 1)
        else:
            # 没有验证集时退化为原逻辑（用训练 loss 监控）
            monitor_loss = train_loss

        scheduler.step(monitor_loss)

        # ---------- 早停 + 最优权重 checkpoint ----------
        if monitor_loss < best_monitor_loss:
            best_monitor_loss = monitor_loss
            # 深拷贝当前权重，作为最优 checkpoint
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                break

    # 恢复最优权重（防止返回过拟合的最后一个 epoch 权重）
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    model._seq_len = seq_len
    return model


# =========================================================
# PyTorch 推理函数
# =========================================================
def predict_pytorch_model(
    model: nn.Module,
    X: pd.DataFrame,
    device: torch.device,
    seq_len: int = 20,
    use_amp: bool = True,
) -> pd.Series:
    """
    使用 PyTorch 模型进行批量推理。

    **性能优化：**
        - 混合精度推理（AMP）：默认启用，减少显存占用，提高推理速度
        - 批量推理：分批处理大数据集，避免显存溢出

    Args:
        model: 训练好的模型
        X: 特征 DataFrame
        device: 计算设备
        seq_len: 序列长度
        use_amp: 是否启用混合精度推理（默认 True）

    Returns:
        预测概率 Series
    """
    _require_torch("predict_pytorch_model")
    model.eval()
    X_tensor = torch.tensor(X.values, dtype=torch.float32)

    with torch.no_grad():
        if isinstance(model, (LSTM, GRU, CNN1D, Transformer)):
            n_samples = X_tensor.shape[0]
            if n_samples - seq_len + 1 <= 0:
                raise ValueError(f"seq_len ({seq_len}) 超过样本数 ({n_samples})，无法进行序列推理。")

            X_seq_all = X_tensor.unfold(0, seq_len, 1).permute(0, 2, 1).contiguous()
            X_seq_all = X_seq_all.to(device, non_blocking=True)
            
            # 混合精度推理
            if use_amp and device.type == "cuda":
                with autocast(device_type=device.type):
                    logits = model(X_seq_all)
            else:
                logits = model(X_seq_all)
            
            probs = _logits_to_probs(logits).cpu().numpy().flatten()
            valid_indices = X.index[seq_len - 1:]
            return pd.Series(probs, index=valid_indices, name="dpoint")

        else:
            # MLP 或其他模型：批量推理
            batch_size = 1024
            all_probs = []
            for start in range(0, X_tensor.shape[0], batch_size):
                batch = X_tensor[start : start + batch_size].to(device, non_blocking=True)
                
                # 混合精度推理
                if use_amp and device.type == "cuda":
                    with autocast(device_type=device.type):
                        logits = model(batch)
                else:
                    logits = model(batch)
                    
                all_probs.append(_logits_to_probs(logits).cpu())
            probs = torch.cat(all_probs, dim=0).numpy().flatten()
            return pd.Series(probs, index=X.index, name="dpoint")


# =========================================================
# sklearn 模型构建
# =========================================================
LOGREG_MAX_ITER: int = 8000
SGD_MAX_ITER: int = 3000
SGD_TOL: float = 1e-3


def _try_import_xgboost() -> Any:
    """尝试导入 xgboost，失败返回 None。"""
    try:
        import xgboost as xgb
        return xgb
    except Exception:
        return None


def make_model(candidate: Dict[str, Any], seed: int) -> Any:
    """
    根据 candidate["model_config"] 构建未拟合的 sklearn 兼容模型或 PyTorch 模型。

    支持的 model_type:
        logreg  — LogisticRegression + StandardScaler Pipeline
        sgd     — SGDClassifier(log_loss) + StandardScaler Pipeline
        xgb     — XGBClassifier
        mlp     — PyTorch MLP
    """
    model_type = str(candidate["model_config"]["model_type"])
    model_config = candidate["model_config"]

    if model_type == "logreg":
        C = float(model_config["C"])
        penalty = str(model_config["penalty"])
        solver = str(model_config["solver"])
        class_weight = model_config.get("class_weight", None)
        l1_ratio = model_config.get("l1_ratio", None)
        max_iter = int(model_config.get("max_iter", LOGREG_MAX_ITER))
        clf = LogisticRegression(
            C=C, penalty=penalty, solver=solver, l1_ratio=l1_ratio,
            class_weight=class_weight, max_iter=max_iter, random_state=seed,
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    if model_type == "sgd":
        alpha = float(model_config["alpha"])
        penalty = str(model_config["penalty"])
        l1_ratio = float(model_config.get("l1_ratio", 0.15))
        class_weight = model_config.get("class_weight", None)
        max_iter = int(model_config.get("max_iter", SGD_MAX_ITER))
        tol = float(model_config.get("tol", SGD_TOL))
        clf = SGDClassifier(
            loss="log_loss", alpha=alpha, penalty=penalty,
            l1_ratio=l1_ratio if penalty == "elasticnet" else None,
            class_weight=class_weight, max_iter=max_iter, tol=tol, random_state=seed,
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    if model_type == "xgb":
        xgb = _try_import_xgboost()
        if xgb is None:
            raise RuntimeError("xgboost_not_installed")
        xgb_params = dict(model_config["params"])
        xgb_params.pop("verbose", None)
        if "eval_metric" not in xgb_params:
            xgb_params["eval_metric"] = "logloss"
        clf = xgb.XGBClassifier(**xgb_params)
        return clf

    if model_type == "mlp":
        _require_torch("make_model(model_type='mlp')")
        input_dim = int(model_config["input_dim"])
        hidden_dim = int(model_config["hidden_dim"])
        dropout_rate = float(model_config.get("dropout_rate", 0.5))
        return MLP(input_dim, hidden_dim, dropout_rate=dropout_rate)

    raise ValueError(f"Unknown model_type: {model_type}")


# ✅ 改后：统一处理所有 PyTorch 模型
def predict_dpoint(model: Any, X: pd.DataFrame) -> pd.Series:
    # 所有 PyTorch 模型统一走这里
    if isinstance(model, (MLP, LSTM, GRU, CNN1D, Transformer)):
        device = _get_device()
        # MLP 序列长度为 1，序列模型从训练时保存的属性读取，找不到则默认 20
        if isinstance(model, MLP):
            seq_len = 1
        else:
            seq_len = getattr(model, "_seq_len", 20)
        return predict_pytorch_model(model, X, device, seq_len=seq_len)

    # sklearn Pipeline / XGBClassifier
    if hasattr(model, "predict_proba"):
        if isinstance(model, Pipeline):
            proba = model.predict_proba(X.values)[:, 1]
        else:
            proba = model.predict_proba(X)[:, 1]
        return pd.Series(proba, index=X.index, name="dpoint")

    raise ValueError(
        f"Unsupported model type: {type(model)}. "
        "Must be a PyTorch model (MLP/LSTM/GRU/CNN1D/Transformer) "
        "or sklearn model with predict_proba."
    )