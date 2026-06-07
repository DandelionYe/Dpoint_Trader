# trainer.py
"""
统一训练循环：支持 sklearn 和 PyTorch 模型。
来自 Ver1.0 的自动 batch 调优和 AMP 支持。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def train_sklearn_model(model, X_train: np.ndarray, y_train: np.ndarray) -> Any:
    """训练 sklearn 模型。"""
    model.fit(X_train, y_train)
    return model


def predict_sklearn_model(model, X: np.ndarray) -> np.ndarray:
    """sklearn 模型预测概率。"""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]  # 二分类取正类概率
        return proba
    return model.predict(X)


def train_pytorch_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    *,
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    patience: int = 10,
    device: str = "auto",
    use_amp: bool = True,
) -> dict:
    """
    训练 PyTorch 模型。

    Returns:
        训练历史 dict
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        raise ImportError("PyTorch is required for DL models")

    # 设备选择
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    model = model.to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = nn.BCEWithLogitsLoss()

    # 数据准备
    X_t = torch.as_tensor(X_train, dtype=torch.float32)
    y_t = torch.as_tensor(y_train, dtype=torch.float32)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=(dev.type == "cuda"))

    X_v = y_v = None
    if X_val is not None and y_val is not None:
        X_v = torch.as_tensor(X_val, dtype=torch.float32).to(dev)
        y_v = torch.as_tensor(y_val, dtype=torch.float32).to(dev)

    # AMP
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and dev.type == "cuda"))
    amp_ctx = torch.amp.autocast("cuda", enabled=(use_amp and dev.type == "cuda"))

    # 训练循环
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(dev, non_blocking=True)
            y_batch = y_batch.to(dev, non_blocking=True)

            optimizer.zero_grad()
            with amp_ctx:
                logits = model(X_batch).squeeze(-1)
                loss = criterion(logits, y_batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())

        avg_train = float(np.mean(train_losses))
        history["train_loss"].append(avg_train)

        # 验证
        if X_v is not None:
            model.eval()
            with torch.no_grad(), amp_ctx:
                val_logits = model(X_v).squeeze(-1)
                val_loss = criterion(val_logits, y_v).item()
            history["val_loss"].append(val_loss)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

    # 恢复最优权重
    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def predict_pytorch_model(model, X: np.ndarray, device: str = "auto") -> np.ndarray:
    """PyTorch 模型预测概率。"""
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    model = model.to(dev)
    model.eval()
    X_t = torch.as_tensor(X, dtype=torch.float32).to(dev)

    with torch.no_grad():
        logits = model(X_t).squeeze(-1)
        proba = torch.sigmoid(logits).cpu().numpy()

    return proba
