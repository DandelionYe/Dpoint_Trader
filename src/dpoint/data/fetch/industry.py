"""
行业分类数据库查询。

从国泰安 CSMAR SQLite 数据库查询行业成员股票列表。
数据库来源: J:\\Dandelions_investment_agent\\storage\\reference\\csmar_industry.sqlite

该数据库由 Dandelions_investment_agent 的 scripts/build_csmar_industry_reference.py 从
TRD_Co.csv 构建，包含两个表:
- securities: 每只股票的行业归属
- industry_members: 行业代码到股票的扁平映射
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = r"J:\Dandelions_investment_agent\storage\reference\csmar_industry.sqlite"


@dataclass
class IndustryInfo:
    """行业信息。"""

    code: str  # 行业代码，如 "C27"
    name: str  # 行业名称，如 "医药制造业"
    count: int  # 成员股票数量


class IndustryDB:
    """行业分类数据库查询接口。"""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"行业分类数据库不存在: {self.db_path}\n"
                f"请确认 Dandelions_investment_agent 仓库路径正确，"
                f"或通过 --db 参数指定数据库路径。"
            )
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self) -> None:
        """关闭数据库连接。"""
        self._conn.close()

    def list_industries(self, industry_level: str = "CSMAR_ZX") -> List[IndustryInfo]:
        """
        列出所有行业代码、名称和成员数量。
        """
        sql = """
            SELECT industry_code, industry_name, COUNT(*) as cnt
            FROM industry_members
            WHERE industry_level = ? AND is_active = 1
            GROUP BY industry_code, industry_name
            ORDER BY cnt DESC
        """
        rows = self._conn.execute(sql, (industry_level,)).fetchall()
        return [
            IndustryInfo(code=r["industry_code"], name=r["industry_name"], count=r["cnt"])
            for r in rows
        ]

    def get_industry_members(
        self,
        industry_code: str,
        industry_level: str = "CSMAR_ZX",
        active_only: bool = True,
        exclude_st: bool = True,
    ) -> List[str]:
        """
        获取指定行业的所有股票代码。
        """
        conditions = ["industry_code = ?", "industry_level = ?"]
        params: list = [industry_code, industry_level]

        if active_only:
            conditions.append("is_active = 1")
        if exclude_st:
            conditions.append("is_st_name = 0")

        sql = f"""
            SELECT symbol
            FROM industry_members
            WHERE {" AND ".join(conditions)}
            ORDER BY symbol
        """
        rows = self._conn.execute(sql, params).fetchall()
        return [r["symbol"] for r in rows]

    def resolve_stock_industry(
        self,
        symbol: str,
        industry_level: str = "CSMAR_ZX",
    ) -> dict:
        """
        查询单只股票的行业归属。
        """
        sql = """
            SELECT primary_industry_code, primary_industry_name,
                   industry_section_code, industry_section_name
            FROM securities
            WHERE symbol = ?
        """
        row = self._conn.execute(sql, (symbol,)).fetchone()
        if not row:
            return {}

        if industry_level == "CSMAR_ZX":
            return {
                "industry_code": row["primary_industry_code"],
                "industry_name": row["primary_industry_name"],
            }
        else:
            return {
                "industry_code": row["industry_section_code"],
                "industry_name": row["industry_section_name"],
            }
