# src/dpoint/data/fetch/__init__.py
"""数据获取模块：从 QMT 获取价格数据，从国泰安获取行业分类。

用法:
    dpoint fetch single --code 000001.SZ
    dpoint fetch basket --industry C27
"""
from __future__ import annotations

from dpoint.data.fetch.formatter import (
    generate_csv_filename,
    qmt_to_dpoint_csv,
    qmt_to_dpoint_single,
)

__all__ = [
    "generate_csv_filename",
    "qmt_to_dpoint_csv",
    "qmt_to_dpoint_single",
]
