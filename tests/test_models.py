# test_models.py
"""模型创建和基本前向传播测试。"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dpoint.models.registry import make_model, ML_MODELS, DL_MODELS
from dpoint.models.sklearn_models import create_sklearn_model
from dpoint.core.tasks import LabelSpec


def test_sklearn_logreg():
    # 注意: LogisticRegression 使用 scipy L-BFGS-B，在某些 Windows 环境下会崩溃
    # 此处仅测试模型创建，不调用 fit
    model = create_sklearn_model("logreg", {"C": 1.0, "penalty": "l2"})
    assert model is not None
    assert hasattr(model, "fit")


def test_sklearn_sgd():
    model = create_sklearn_model("sgd", {"alpha": 1e-4})
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    model.fit(X, y)
    pred = model.predict(X[:10])
    assert len(pred) == 10


def test_sklearn_sgd_predict_proba():
    model = create_sklearn_model("sgd", {"alpha": 1e-4})
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    model.fit(X, y)
    proba = model.predict_proba(X[:10])
    assert proba.shape[0] == 10


def test_torch_mlp():
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")

    from dpoint.models.torch_models import create_dl_model

    model = create_dl_model("mlp", input_dim=10, config={"hidden_dim": 32, "num_layers": 2})
    X = torch.randn(8, 10)
    out = model(X)
    assert out.shape == (8, 1)


def test_torch_lstm():
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")

    from dpoint.models.torch_models import create_dl_model

    model = create_dl_model("lstm", input_dim=10, config={"hidden_dim": 32, "seq_len": 5})
    X = torch.randn(8, 5, 10)
    out = model(X)
    assert out.shape == (8, 1)


def test_torch_transformer():
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")

    from dpoint.models.torch_models import create_dl_model

    model = create_dl_model(
        "transformer", input_dim=10, config={"hidden_dim": 32, "nhead": 4, "num_layers": 1}
    )
    X = torch.randn(8, 5, 10)
    out = model(X)
    assert out.shape == (8, 1)


def test_make_model_sklearn():
    label_spec = LabelSpec(task_type="binary_classification", label_mode="binary_next_close_up")
    model, kind = make_model("logreg", input_dim=10, config={}, label_spec=label_spec)
    assert kind == "sklearn"


def test_make_model_torch():
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")

    label_spec = LabelSpec(task_type="binary_classification", label_mode="binary_next_close_up")
    model, kind = make_model("lstm", input_dim=10, config={"hidden_dim": 32}, label_spec=label_spec)
    assert kind == "torch"
