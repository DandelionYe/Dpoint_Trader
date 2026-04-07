# conftest.py
"""
Pytest fixtures for A-share Dpoint Trader tests (Basket Mode Only).
Provides minimal sample data fixtures for testing basket mode.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
import os
import tempfile


@pytest.fixture
def minimal_price_data():
    """
    Minimal price data for testing (100 trading days).
    Note: This is for single-stock backtester tests only.
    """
    np.random.seed(42)
    n = 100
    base_price = 10.0

    dates = pd.date_range(start="2020-01-01", periods=n, freq="B")

    returns = np.random.normal(0.0005, 0.02, n)
    close_prices = base_price * np.exp(np.cumsum(returns))
    open_prices = close_prices * (1 + np.random.uniform(-0.01, 0.01, n))
    high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(np.random.uniform(0, 0.02, n)))
    low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(np.random.uniform(0, 0.02, n)))
    volumes = np.random.uniform(1_000_000, 10_000_000, n)
    amounts = volumes * close_prices

    df = pd.DataFrame({
        "date": dates,
        "open_qfq": open_prices,
        "high_qfq": high_prices,
        "low_qfq": low_prices,
        "close_qfq": close_prices,
        "volume": volumes,
        "amount": amounts,
    })

    return df


@pytest.fixture
def basket_sample_data():
    """
    Sample basket data for testing (3 stocks, 100 days each).
    Returns a dict of {stock_code: DataFrame}.
    """
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start="2020-01-01", periods=n, freq="B")

    stock_dict = {}
    for code in ["000001", "000002", "000003"]:
        base_price = 10.0 + np.random.uniform(-2, 2)
        returns = np.random.normal(0.0005, 0.02, n)
        close_prices = base_price * np.exp(np.cumsum(returns))
        open_prices = close_prices * (1 + np.random.uniform(-0.01, 0.01, n))
        high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(np.random.uniform(0, 0.02, n)))
        low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(np.random.uniform(0, 0.02, n)))
        volumes = np.random.uniform(1_000_000, 10_000_000, n)

        stock_dict[code] = pd.DataFrame({
            "date": dates,
            "open_qfq": open_prices,
            "high_qfq": high_prices,
            "low_qfq": low_prices,
            "close_qfq": close_prices,
            "volume": volumes,
        })

    return stock_dict


@pytest.fixture
def basket_csv_dir(tmp_path):
    """
    Create a temporary basket directory with CSV files.
    Returns the directory path.
    """
    basket_dir = tmp_path / "basket_test"
    basket_dir.mkdir()

    np.random.seed(42)
    n = 100
    dates = pd.date_range(start="2020-01-01", periods=n, freq="B")

    stocks = [
        ("000001", "20200101"),
        ("000002", "20200101"),
        ("000003", "20200101"),
    ]

    for code, listing_date in stocks:
        base_price = 10.0 + np.random.uniform(-2, 2)
        returns = np.random.normal(0.0005, 0.02, n)
        close_prices = base_price * np.exp(np.cumsum(returns))
        open_prices = close_prices * (1 + np.random.uniform(-0.01, 0.01, n))
        high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(np.random.uniform(0, 0.02, n)))
        low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(np.random.uniform(0, 0.02, n)))
        volumes = np.random.uniform(1_000_000, 10_000_000, n)

        df = pd.DataFrame({
            "Date": dates.strftime("%Y-%m-%d"),
            "Open (CNY, qfq)": open_prices,
            "High (CNY, qfq)": high_prices,
            "Low (CNY, qfq)": low_prices,
            "Close (CNY, qfq)": close_prices,
            "Volume (shares)": volumes,
        })

        csv_path = basket_dir / f"{code}_{listing_date}.csv"
        df.to_csv(csv_path, index=False)

    return str(basket_dir)


@pytest.fixture
def price_data_with_trend():
    """
    Price data with clear uptrend for testing.
    """
    n = 200
    base_price = 10.0

    dates = pd.date_range(start="2020-01-01", periods=n, freq="B")

    trend = np.linspace(0, 0.3, n)
    returns = trend + np.random.normal(0, 0.01, n)
    close_prices = base_price * np.exp(np.cumsum(returns))
    open_prices = close_prices * (1 + np.random.uniform(-0.005, 0.005, n))
    high_prices = np.maximum(open_prices, close_prices) * (1 + np.random.uniform(0, 0.01, n))
    low_prices = np.minimum(open_prices, close_prices) * (1 - np.random.uniform(0, 0.01, n))
    volumes = np.random.uniform(1_000_000, 10_000_000, n)

    df = pd.DataFrame({
        "date": dates,
        "open_qfq": open_prices,
        "high_qfq": high_prices,
        "low_qfq": low_prices,
        "close_qfq": close_prices,
        "volume": volumes,
    })
    df = df.set_index("date")

    return df


@pytest.fixture
def sample_dpoint_series():
    """
    Sample Dpoint series for testing.
    """
    n = 100
    dates = pd.date_range(start="2020-01-01", periods=n, freq="B")

    np.random.seed(42)
    dpoint_values = np.random.uniform(0.3, 0.7, n)
    dpoint_values[10:20] = 0.8
    dpoint_values[50:60] = 0.2

    dpoint = pd.Series(dpoint_values, index=dates, name="dpoint")
    return dpoint


@pytest.fixture
def sample_dpoint_matrix():
    """
    Sample Dpoint matrix for basket mode testing.
    Returns a dict of {stock_code: pd.Series}.
    """
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start="2020-01-01", periods=n, freq="B")

    dpoint_matrix = {}
    for code in ["000001", "000002", "000003"]:
        dpoint_values = np.random.uniform(0.3, 0.7, n)
        dpoint_matrix[code] = pd.Series(dpoint_values, index=dates, name="dpoint")

    return dpoint_matrix


@pytest.fixture
def minimal_trade_config():
    """
    Minimal trade configuration for testing.
    """
    return {
        "initial_cash": 100000.0,
        "buy_threshold": 0.6,
        "sell_threshold": 0.4,
        "confirm_days": 2,
        "min_hold_days": 1,
        "max_hold_days": 20,
    }


@pytest.fixture
def minimal_basket_config():
    """
    Minimal basket configuration for testing.
    """
    return {
        "feature_config": {
            "windows": [5, 10],
            "use_momentum": True,
            "use_volatility": True,
            "use_volume": True,
            "use_candle": True,
            "use_turnover": False,
            "use_ta_indicators": False,
        },
        "model_config": {
            "model_type": "logreg",
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
        },
        "trade_config": {
            "initial_cash": 1000000.0,
            "buy_threshold": 0.6,
            "sell_threshold": 0.4,
            "confirm_days": 2,
            "min_hold_days": 1,
            "max_hold_days": 20,
            "top_k": 2,
        },
    }


@pytest.fixture
def sample_trades():
    """
    Sample trades for testing metrics calculations.
    """
    return pd.DataFrame({
        "buy_signal_date": pd.to_datetime(["2020-01-10", "2020-02-15", "2020-03-20"]),
        "buy_exec_date": pd.to_datetime(["2020-01-11", "2020-02-18", "2020-03-23"]),
        "buy_price": [10.0, 11.0, 12.0],
        "buy_shares": [1000, 900, 800],
        "buy_cost": [10030.0, 9900.0, 9720.0],
        "sell_signal_date": pd.to_datetime(["2020-02-10", "2020-03-15", pd.NaT]),
        "sell_exec_date": pd.to_datetime(["2020-02-11", "2020-03-16", pd.NaT]),
        "sell_price": [11.0, 13.0, np.nan],
        "sell_shares": [1000, 900, np.nan],
        "sell_proceeds": [10857.0, 11664.0, np.nan],
        "pnl": [827.0, 1764.0, np.nan],
        "return": [0.0825, 0.1782, np.nan],
        "success": [True, True, np.nan],
        "status": ["CLOSED", "CLOSED", "OPEN"],
    })


@pytest.fixture
def sample_equity_curve():
    """
    Sample equity curve for testing metrics.
    """
    dates = pd.date_range(start="2020-01-01", periods=100, freq="B")
    initial_cash = 100000.0

    equity = initial_cash * (1 + np.cumsum(np.random.normal(0.001, 0.01, 100)))
    equity = np.maximum(equity, initial_cash * 0.9)

    return pd.DataFrame({
        "date": dates,
        "total_equity": equity,
        "cash": equity * 0.3,
        "shares": equity * 0.7 / 10.0,
    })


@pytest.fixture
def sample_portfolio_equity_curve():
    """
    Sample portfolio equity curve for testing basket mode metrics.
    """
    dates = pd.date_range(start="2020-01-01", periods=100, freq="B")
    initial_cash = 1000000.0

    equity = initial_cash * (1 + np.cumsum(np.random.normal(0.001, 0.01, 100)))
    equity = np.maximum(equity, initial_cash * 0.9)

    return pd.DataFrame({
        "date": dates,
        "total_equity": equity,
        "cash": equity * 0.3,
        "market_value": equity * 0.7,
        "n_positions": np.random.randint(1, 5, 100),
    })


@pytest.fixture
def price_data_limit_up():
    """
    Price data with limit up days for testing.
    """
    n = 50
    dates = pd.date_range(start="2020-01-01", periods=n, freq="B")

    prices = [10.0]
    for i in range(n - 1):
        if i == 10:
            prices.append(prices[-1] * 1.10)
        elif i == 20:
            prices.append(prices[-1] * 0.90)
        else:
            prices.append(prices[-1] * (1 + np.random.uniform(-0.02, 0.02)))

    df = pd.DataFrame({
        "date": dates,
        "open_qfq": prices,
        "close_qfq": prices,
        "volume": [5_000_000] * n,
    })
    df = df.set_index("date")
    return df


@pytest.fixture
def price_data_suspended():
    """
    Price data with suspended days for testing.
    """
    n = 50
    dates = pd.date_range(start="2020-01-01", periods=n, freq="B")

    prices = [10.0] * n
    prices[10] = 0.0
    prices[15] = 0.0

    df = pd.DataFrame({
        "date": dates,
        "open_qfq": prices,
        "close_qfq": prices,
        "volume": [5_000_000] * n,
    })
    df = df.set_index("date")
    return df


@pytest.fixture
def temp_output_dir(tmp_path):
    """
    Temporary output directory for test artifacts.
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return str(output_dir)
