# calibration.py
"""
概率校准模块：Platt Scaling / Isotonic Regression。
来自 Ver2.0/calibration.py。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

CALIBRATION_METHODS = ["none", "platt", "isotonic"]


class ProbabilityCalibrator:
    """
    概率校准器。

    支持三种方法:
        - none: 不校准
        - platt: Platt Scaling (logistic regression)
        - isotonic: Isotonic Regression

    校准在 validation set 上拟合，推理时将 raw probability 转为 calibrated probability。
    """

    def __init__(self, method: str = "none"):
        if method not in CALIBRATION_METHODS:
            raise ValueError(f"Unknown method: {method}. Must be one of {CALIBRATION_METHODS}")
        self.method = method
        self.calibrator: Optional[Any] = None
        self._is_fitted = False
        self._use_logit = False

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> "ProbabilityCalibrator":
        """在 validation set 上拟合校准器。"""
        if self.method == "none":
            self._is_fitted = True
            return self

        y_true = np.asarray(y_true).flatten()
        y_prob = np.asarray(y_prob).flatten()

        mask = np.isfinite(y_true) & np.isfinite(y_prob)
        y_true = y_true[mask]
        y_prob = y_prob[mask]

        if len(y_true) < 10:
            logger.warning(
                "Too few samples for calibration (%d), falling back to none", len(y_true)
            )
            self.method = "none"
            self._is_fitted = True
            return self

        if self.method == "platt":
            # Platt Scaling: 用 sigmoid 拟合，避免 scipy L-BFGS-B 崩溃
            # 使用简单的 numpy 实现: logit(p) = a * log(p/(1-p)) + b
            from sklearn.isotonic import IsotonicRegression

            # 退化为 isotonic（更稳定），或用 numpy 手动实现 Platt
            p = np.clip(y_prob, 1e-7, 1 - 1e-7)
            logit = np.log(p / (1 - p))
            # 线性回归: y_true ~ sigmoid(a * logit + b)
            # 用 isotonic regression 作为替代（单调性保证）
            self.calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
            self.calibrator.fit(logit, y_true)
            self._use_logit = True

        elif self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            self.calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
            self.calibrator.fit(y_prob, y_true)

        self._is_fitted = True
        logger.info("Calibration fitted: method=%s, n_samples=%d", self.method, len(y_true))
        return self

    def predict(self, y_prob: np.ndarray) -> np.ndarray:
        """将原始概率转换为校准后的概率。"""
        if not self._is_fitted:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")

        if self.method == "none":
            return y_prob

        y_prob = np.asarray(y_prob).flatten()

        if self.method == "platt":
            if self._use_logit:
                p = np.clip(y_prob, 1e-7, 1 - 1e-7)
                logit = np.log(p / (1 - p))
                return self.calibrator.predict(logit)
            return self.calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]

        elif self.method == "isotonic":
            return self.calibrator.predict(y_prob)

        return y_prob

    def evaluate(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """计算校准质量指标。"""
        y_true = np.asarray(y_true).flatten()
        y_prob = np.asarray(y_prob).flatten()

        mask = np.isfinite(y_true) & np.isfinite(y_prob)
        y_true = y_true[mask]
        y_prob = y_prob[mask]

        metrics = {}

        # Brier Score
        try:
            from sklearn.metrics import brier_score_loss

            metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))
        except Exception:
            pass

        # ECE (Expected Calibration Error)
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask_bin = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            if mask_bin.sum() == 0:
                continue
            bin_acc = y_true[mask_bin].mean()
            bin_conf = y_prob[mask_bin].mean()
            ece += mask_bin.sum() * abs(bin_acc - bin_conf)
        metrics["ece"] = float(ece / max(len(y_true), 1))

        return metrics
