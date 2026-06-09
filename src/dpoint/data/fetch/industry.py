"""
行业分类数据库查询。

从本仓库 data/csmar_industry.sqlite 查询股票分类信息。
支持 7 个维度筛选：4 级行业 + 省份 + 城市 + 所有权。

用法:
    from dpoint.data.fetch.industry import IndustryDB

    with IndustryDB() as db:
        # 列出某维度的可选值
        industries = db.list_values("ind4")

        # 按维度筛选股票
        codes = db.query_stocks(ind4="C27", province="广东省")

        # 查询单只股票的分类信息
        info = db.resolve_stock("000001")
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# 默认路径：本仓库 data/ 目录
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
DEFAULT_DB_PATH = _PROJECT_ROOT / "data" / "csmar_industry.sqlite"

# list_values 支持的维度及其代码/名称列
_DIMENSION_COLUMNS = {
    "ind1": ("ind1_code", "ind1_name"),
    "ind2": ("ind2_code", "ind2_name"),
    "ind3": ("ind3_code", "ind3_name"),
    "ind4": ("ind4_code", "ind4_name"),
    "province": ("province_code", "province"),
    "city": ("city_code", "city"),
    "ownership": ("ownership_code", "ownership"),
}

# query_stocks 支持的筛选维度 → 列名
_FILTER_COLUMNS = {
    "ind1": "ind1_code",
    "ind2": "ind2_code",
    "ind3": "ind3_code",
    "ind4": "ind4_code",
    "province": "province",
    "city": "city",
    "ownership": "ownership",
}


@dataclass
class DimensionValue:
    """某维度的一个可选值。"""

    code: str
    name: str
    count: int


class IndustryDB:
    """行业分类数据库查询接口。"""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"行业分类数据库不存在: {self.db_path}\n"
                f"请先运行 python scripts/build_industry_db.py 构建数据库。"
            )
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self) -> None:
        """关闭数据库连接（幂等）。"""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def list_values(self, dimension: str) -> list[DimensionValue]:
        """
        列出某维度的所有可选值及其股票数量。

        Args:
            dimension: "ind1" | "ind2" | "ind3" | "ind4" | "province" | "city" | "ownership"

        Returns:
            按数量降序排列的 DimensionValue 列表
        """
        if dimension not in _DIMENSION_COLUMNS:
            raise ValueError(
                f"未知维度: {dimension}，可选: {list(_DIMENSION_COLUMNS.keys())}"
            )

        code_col, name_col = _DIMENSION_COLUMNS[dimension]
        sql = f"""
            SELECT {code_col} AS code, {name_col} AS name, COUNT(*) AS cnt
            FROM stocks
            WHERE {code_col} IS NOT NULL
            GROUP BY {code_col}, {name_col}
            ORDER BY cnt DESC
        """
        rows = self._conn.execute(sql).fetchall()
        return [
            DimensionValue(code=r["code"], name=r["name"], count=r["cnt"]) for r in rows
        ]

    def query_stocks(self, **filters: str) -> list[str]:
        """
        按维度筛选股票，返回 6 位代码列表。所有条件取交集。

        Args:
            **filters: 维度名=值，如 ind4="C27", province="广东省"

        Returns:
            排序后的股票代码列表
        """
        conditions = []
        params: list[str] = []

        for dim, value in filters.items():
            if dim not in _FILTER_COLUMNS:
                raise ValueError(
                    f"未知筛选维度: {dim}，可选: {list(_FILTER_COLUMNS.keys())}"
                )
            col = _FILTER_COLUMNS[dim]
            # 先尝试按 code 精确匹配，再按 name 匹配
            conditions.append(f"({col} = ? OR {col.replace('_code', '_name')} = ?)")
            params.extend([value, value])

        where = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT code FROM stocks WHERE {where} ORDER BY code"

        rows = self._conn.execute(sql, params).fetchall()
        return [r["code"] for r in rows]

    def resolve_stock(self, code: str) -> dict:
        """
        查询单只股票的全部分类信息。

        Args:
            code: 6 位股票代码，如 "000001"

        Returns:
            包含所有分类字段的字典，未找到返回空字典
        """
        code = code.split(".")[0] if "." in code else code
        code = code.zfill(6)

        row = self._conn.execute(
            "SELECT * FROM stocks WHERE code = ?", (code,)
        ).fetchone()
        if not row:
            return {}
        return dict(row)
