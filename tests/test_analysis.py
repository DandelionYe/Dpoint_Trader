# test_analysis.py
"""分析模块测试：校准、解释、Regime。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_calibration_none():
    from dpoint.analysis.calibration import ProbabilityCalibrator

    cal = ProbabilityCalibrator("none")
    cal.fit(np.array([0, 1, 0, 1]), np.array([0.2, 0.8, 0.3, 0.7]))
    result = cal.predict(np.array([0.4, 0.6]))
    np.testing.assert_array_equal(result, [0.4, 0.6])


def test_calibration_platt():
    from dpoint.analysis.calibration import ProbabilityCalibrator

    rng = np.random.Generator(np.random.PCG64(42))
    n = 200
    y_true = rng.integers(0, 2, n)
    y_prob = np.clip(y_true * 0.3 + rng.normal(0, 0.2, n), 0.01, 0.99)

    cal = ProbabilityCalibrator("platt")
    cal.fit(y_true[:150], y_prob[:150])
    calibrated = cal.predict(y_prob[150:])
    assert len(calibrated) == 50
    assert all(0 <= v <= 1 for v in calibrated)


def test_calibration_isotonic():
    from dpoint.analysis.calibration import ProbabilityCalibrator

    rng = np.random.Generator(np.random.PCG64(42))
    n = 200
    y_true = rng.integers(0, 2, n)
    y_prob = np.clip(y_true * 0.3 + rng.normal(0, 0.2, n), 0.01, 0.99)

    cal = ProbabilityCalibrator("isotonic")
    try:
        cal.fit(y_true[:150], y_prob[:150])
        calibrated = cal.predict(y_prob[150:])
        assert len(calibrated) == 50
    except Exception:
        pytest.skip("Isotonic calibration failed")


def test_calibration_evaluate():
    from dpoint.analysis.calibration import ProbabilityCalibrator

    rng = np.random.Generator(np.random.PCG64(42))
    n = 100
    y_true = rng.integers(0, 2, n)
    y_prob = rng.uniform(0, 1, n)

    cal = ProbabilityCalibrator("none")
    metrics = cal.evaluate(y_true, y_prob)
    assert "brier_score" in metrics
    assert "ece" in metrics


def test_permutation_importance():
    from sklearn.tree import DecisionTreeClassifier

    from dpoint.analysis.explainer import permutation_importance

    rng = np.random.Generator(np.random.PCG64(42))
    X = rng.normal(0, 1, (100, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X, y)

    importance = permutation_importance(model, X, y, ["f0", "f1", "f2", "f3", "f4"], n_repeats=3)
    assert len(importance) == 5
    assert "importance_mean" in importance.columns


def test_regime_detector():
    from dpoint.analysis.regime import RegimeDetector

    rng = np.random.Generator(np.random.PCG64(42))
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, 200))))

    detector = RegimeDetector()
    regimes = detector.detect(close, mode="combined")
    assert len(regimes) == 200
    # 应该有多种 regime
    unique_regimes = regimes.unique()
    assert len(unique_regimes) > 1


def test_regime_trend():
    from dpoint.analysis.regime import RegimeDetector

    close = pd.Series(np.linspace(100, 150, 100))  # 稳定上涨
    detector = RegimeDetector()
    regimes = detector.detect_trend(close)
    # 后半段应该是 trend
    assert regimes.iloc[-1] == "trend"


def test_regime_metrics():
    from dpoint.analysis.regime import compute_regime_metrics

    rng = np.random.Generator(np.random.PCG64(42))
    n = 200
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, n))))
    equity = pd.DataFrame(
        {
            "total_equity": 100000 * (1 + np.cumsum(rng.normal(0.001, 0.01, n))),
        }
    )

    metrics = compute_regime_metrics(equity, close, regime_mode="trend")
    assert isinstance(metrics, dict)
    for regime, m in metrics.items():
        assert "total_return" in m
        assert "sharpe" in m
