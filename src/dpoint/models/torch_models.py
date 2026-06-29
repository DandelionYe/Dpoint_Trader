# torch_models.py
"""
PyTorch 深度学习模型定义：MLP / LSTM / GRU / CNN1D / Transformer。
合并自两个项目的模型实现。
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _require_torch():
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for deep learning models. Install with: pip install torch"
        )


# =========================================================
# MLP
# =========================================================


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout_rate: float = 0.3,
        output_dim: int = 1,
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# =========================================================
# LSTM
# =========================================================


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout_rate: float = 0.3,
        bidirectional: bool = False,
        output_dim: int = 1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        fc_input = hidden_dim * 2 if bidirectional else hidden_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(fc_input, output_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, features)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(self.dropout(last))


# =========================================================
# GRU
# =========================================================


class GRUModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout_rate: float = 0.3,
        bidirectional: bool = False,
        output_dim: int = 1,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        fc_input = hidden_dim * 2 if bidirectional else hidden_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(fc_input, output_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.fc(self.dropout(last))


# =========================================================
# CNN1D
# =========================================================


class CNN1D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        kernel_sizes: list = None,
        dropout_rate: float = 0.3,
        output_dim: int = 1,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [2, 3, 5]
        self.convs = nn.ModuleList(
            [nn.Conv1d(input_dim, hidden_dim, kernel_size=k, padding=k // 2) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * len(kernel_sizes), output_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, features) -> treat as seq_len=1
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        conv_outs = []
        for conv in self.convs:
            c = torch.relu(conv(x))
            c = c.max(dim=2).values  # global max pooling
            conv_outs.append(c)
        out = torch.cat(conv_outs, dim=1)
        return self.fc(self.dropout(out))


# =========================================================
# Transformer
# =========================================================


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout_rate: float = 0.3,
        output_dim: int = 1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x.mean(dim=1)  # mean pooling
        return self.fc(self.dropout(x))


# =========================================================
# 模型工厂
# =========================================================

DL_MODEL_REGISTRY = {
    "mlp": MLP,
    "lstm": LSTMModel,
    "gru": GRUModel,
    "cnn": CNN1D,
    "transformer": TransformerModel,
}


def create_dl_model(
    model_type: str,
    input_dim: int,
    config: Dict[str, Any],
    output_dim: int = 1,
) -> nn.Module:
    """创建 PyTorch 模型。"""
    _require_torch()
    if model_type not in DL_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown DL model type: {model_type}. Available: {list(DL_MODEL_REGISTRY.keys())}"
        )

    cls = DL_MODEL_REGISTRY[model_type]
    kwargs = {"input_dim": input_dim, "output_dim": output_dim}

    if model_type == "mlp":
        kwargs.update(
            hidden_dim=config.get("hidden_dim", 64),
            num_layers=config.get("num_layers", 2),
            dropout_rate=config.get("dropout_rate", 0.3),
        )
    elif model_type in ("lstm", "gru"):
        kwargs.update(
            hidden_dim=config.get("hidden_dim", 64),
            num_layers=config.get("num_layers", 2),
            dropout_rate=config.get("dropout_rate", 0.3),
            bidirectional=config.get("bidirectional", False),
        )
    elif model_type == "cnn":
        kwargs.update(
            hidden_dim=config.get("hidden_dim", 64), dropout_rate=config.get("dropout_rate", 0.3)
        )
    elif model_type == "transformer":
        kwargs.update(
            d_model=config.get("hidden_dim", 64),
            nhead=config.get("nhead", 4),
            num_layers=config.get("num_layers", 2),
            dropout_rate=config.get("dropout_rate", 0.3),
        )

    return cls(**kwargs)
