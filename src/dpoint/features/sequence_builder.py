# sequence_builder.py
"""
序列构建器：将面板数据按 ticker 分组构建滑动窗口序列。
来自 DpointTrader_deeplearning_Ver1.0/sequence_builder.py。
供 LSTM/GRU/CNN/Transformer 使用。
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PanelSequenceStore:
    """
    按 ticker 分组存储特征和标签的序列数据。
    避免跨 ticker 边界的序列泄露。
    """

    def __init__(
        self,
        panel_df: pd.DataFrame,
        feature_names: List[str],
        label_col: str = "label",
        seq_len: int = 10,
        date_col: str = "date",
        ticker_col: str = "ticker",
    ):
        self.seq_len = seq_len
        self.feature_names = feature_names
        self.label_col = label_col

        self._sequences = []
        self._labels = []
        self._tickers = []
        self._end_dates = []

        for ticker, group in panel_df.groupby(ticker_col, sort=False):
            group = group.sort_values(date_col)
            X = group[feature_names].values
            y = group[label_col].values if label_col in group.columns else np.zeros(len(group))
            dates = group[date_col].values

            for i in range(seq_len - 1, len(group)):
                if np.isnan(y[i]):
                    continue
                window = X[i - seq_len + 1 : i + 1]
                if np.any(np.isnan(window)):
                    continue
                self._sequences.append(window)
                self._labels.append(y[i])
                self._tickers.append(ticker)
                self._end_dates.append(dates[i])

        self._sequences = np.array(self._sequences, dtype=np.float32)
        self._labels = np.array(self._labels, dtype=np.float32)

        logger.info(
            "Built %d sequences (seq_len=%d) from %d tickers",
            len(self._sequences),
            seq_len,
            panel_df[ticker_col].nunique(),
        )

    def __len__(self) -> int:
        return len(self._sequences)

    @property
    def X(self) -> np.ndarray:
        return self._sequences

    @property
    def y(self) -> np.ndarray:
        return self._labels

    @property
    def tickers(self) -> list:
        return self._tickers

    @property
    def end_dates(self) -> list:
        return self._end_dates


def build_sequences_from_panel(
    panel_df: pd.DataFrame,
    feature_names: List[str],
    label_col: str = "label",
    seq_len: int = 10,
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> PanelSequenceStore:
    """从面板数据构建序列存储。"""
    return PanelSequenceStore(
        panel_df,
        feature_names,
        label_col,
        seq_len,
        date_col,
        ticker_col,
    )
