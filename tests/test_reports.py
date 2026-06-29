# test_reports.py
"""报告生成测试。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dpoint.reports.excel_reporter import escape_excel_formulas, save_excel_report
from dpoint.reports.metrics import compute_rank_ic, compute_ranking_metrics


def test_escape_excel_formulas():
    df = pd.DataFrame({"col": ["=SUM(A1)", "normal", "+cmd", "-ref", "@func"]})
    result = escape_excel_formulas(df)
    assert result.iloc[0]["col"] == "'=SUM(A1)"
    assert result.iloc[1]["col"] == "normal"
    assert result.iloc[2]["col"] == "'+cmd"


def test_rank_ic_basic():
    rng = np.random.Generator(np.random.PCG64(42))
    n = 100
    df = pd.DataFrame(
        {
            "date": pd.date_range("2021-01-01", periods=n, freq="B").repeat(5),
            "ticker": list("ABCDE") * n,
            "score": rng.uniform(0, 1, n * 5),
            "label": rng.choice([0, 1], n * 5).astype(float),
        }
    )
    ic_series = compute_rank_ic(df)
    assert len(ic_series) > 0


def test_ranking_metrics():
    rng = np.random.Generator(np.random.PCG64(42))
    dates = pd.date_range("2021-01-01", periods=50, freq="B")
    frames = []
    for dt in dates:
        for t in ["A", "B", "C", "D", "E"]:
            frames.append(
                {"date": dt, "ticker": t, "score": rng.uniform(), "label": rng.choice([0, 1])}
            )
    df = pd.DataFrame(frames)
    metrics = compute_ranking_metrics(df, top_k=2, n_layers=2)
    assert metrics.rank_ic_mean is not None
    assert metrics.layered_returns is not None


def test_save_excel_report(tmp_path):
    ec = pd.DataFrame(
        {
            "date": pd.date_range("2021-01-01", periods=10),
            "total_equity": np.linspace(100000, 110000, 10),
        }
    )
    trades = pd.DataFrame(
        {
            "date": pd.date_range("2021-01-01", periods=4),
            "action": ["BUY", "SELL", "BUY", "SELL"],
            "price": [10, 11, 10.5, 11.5],
            "shares": [1000, 1000, 1000, 1000],
        }
    )
    risk = {"total_return": 0.1, "sharpe": 1.5, "max_drawdown": -0.05}

    path = save_excel_report(
        tmp_path / "test_report.xlsx",
        equity_curve=ec,
        trades=trades,
        risk_metrics=risk,
        config={"model": "lstm", "seed": 42},
    )

    assert path.exists()

    # 验证 Excel 内容
    xl = pd.ExcelFile(path)
    assert "RiskMetrics" in xl.sheet_names
    assert "EquityCurve" in xl.sheet_names
    assert "Trades" in xl.sheet_names
    assert "Config" in xl.sheet_names
