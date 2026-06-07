# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dpoint_Trader is a deep-learning quantitative trading research framework for A-shares (Chinese stock market). It supports two modes: single-stock strategies (`dpoint single`) and multi-stock basket/factor strategies (`dpoint basket`). All code comments, docstrings, and CLI help text are in Chinese.

## Setup & Common Commands

```bash
# Install in development mode
pip install -e .

# Install with all optional dependencies
pip install -e ".[all]"

# Run tests
pytest tests/ -v --tb=short

# Lint/format checks (matches CI)
black --check src/dpoint tests
isort --check-only src/dpoint tests
flake8 src/dpoint --count --select=E9,F63,F7,F82 --show-source --statistics

# Auto-format
black src/dpoint tests
isort src/dpoint tests
```

## Source Layout

The package uses **src layout**: all source code lives under `src/dpoint/`, not `dpoint/` at root. The `pyproject.toml` declares `where = ["src"]` for package discovery.

CLI entry point: `dpoint` command → `dpoint.cli.main:main`

## Architecture

The system is a **pipeline** that flows through these stages:

1. **Data loading** (`data/`) — Excel/CSV loader for single-stock; folder-of-CSVs basket loader builds a date×ticker panel
2. **Feature engineering** (`features/`) — time-series features (momentum, volatility, volume, candle, turnover, TA indicators); cross-sectional rank features in basket mode; label construction
3. **Hyperparameter search** (`search/`) — random search over feature/model/trade config space; decoupled evaluate callback pattern
4. **Model training** (`models/`) — factory pattern via `make_model()`; sklearn models (logreg, sgd, xgb) and PyTorch DL models (mlp, lstm, gru, cnn, transformer)
5. **Splitting** (`splits/`) — Walk-Forward / Embargo WF / Final Holdout
6. **Backtesting** (`backtester/`) — A-share execution constraints (limit up/down, T+1, 100-share lots, commission, stamp duty, slippage)
7. **Reports** (`reports/`) — Excel + interactive HTML (plotly optional); ranking metrics (Rank IC, ICIR, layered returns)

Configuration is entirely via dataclasses in `core/config.py`: `FeatureConfig`, `ModelConfig`, `TradeConfig`, `SearchConfig`, `SplitConfig`, `PortfolioConfig`, `RunConfig`.

## Key Design Decisions

- **Optional dependencies**: PyTorch, XGBoost, SHAP, and plotly are all guarded by `try/except ImportError` with graceful fallbacks. XGBoost falls back to sklearn's GradientBoosting.
- **Reproducibility**: Uses `numpy.random.Generator(PCG64)` (not legacy `np.random.seed`). Seed management, data hashing, and experiment contracts (`core/contract.py`) ensure reproducible runs.
- **No cross-ticker leakage**: Rolling features are computed per-ticker via `groupby(ticker)`.
- **Standard column names**: `date`, `ticker`, `open_qfq`, `high_qfq`, `low_qfq`, `close_qfq`, `volume`, `amount`, `turnover_rate` (defined in `core/constants.py`).
- **All `.py` files** use `from __future__ import annotations`.

## Formatting & Style

- **Black**: line-length 100, target Python 3.10
- **isort**: profile "black", known_first_party ["dpoint"]
- **flake8**: max-line-length 120 (secondary check)
