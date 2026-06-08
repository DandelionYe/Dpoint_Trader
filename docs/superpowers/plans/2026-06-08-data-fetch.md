# Data Fetch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `dpoint fetch single` and `dpoint fetch basket` CLI commands to automatically download price data from QMT (XtMiniQMT) and save it in Dpoint_Trader's required format.

**Architecture:** New `src/dpoint/data/fetch/` module with three components: `qmt_client.py` (QMT API wrapper), `industry.py` (CSMAR industry DB query), `formatter.py` (QMT→Dpoint_Trader column mapping). CLI registration in `cli/main.py`.

**Tech Stack:** `xtquant` (optional, for QMT data), `sqlite3` (stdlib, for industry DB), `pandas`, `openpyxl`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/dpoint/data/fetch/__init__.py` | Create | Module init, public API exports |
| `src/dpoint/data/fetch/qmt_client.py` | Create | QMT connection, download, read historical data |
| `src/dpoint/data/fetch/industry.py` | Create | SQLite industry DB query, get industry members |
| `src/dpoint/data/fetch/formatter.py` | Create | Column mapping, date conversion, file output |
| `src/dpoint/cli/main.py:28-93` | Modify | Register `fetch` subcommand with `single`/`basket` sub-subcommands |
| `src/dpoint/cli/main.py:790-807` | Modify | Add `fetch` dispatch in `main()` |
| `tests/test_fetch.py` | Create | Unit tests for formatter, industry, CLI |

---

## Task 1: formatter.py — Column Mapping & File Output

**Files:**
- Create: `src/dpoint/data/fetch/formatter.py`
- Create: `tests/test_fetch.py`

The formatter is the core translation layer. It converts QMT DataFrames (columns: `time, open, high, low, close, volume, amount`) into Dpoint_Trader format (columns: `date, open_qfq, high_qfq, low_qfq, close_qfq, volume, amount`).

- [ ] **Step 1: Write failing tests for formatter**

```python
# tests/test_fetch.py
"""数据获取模块的单元测试。"""
from __future__ import annotations

import pandas as pd
import pytest


class TestQmtToDpointSingle:
    """测试 qmt_to_dpoint_single 转换函数。"""

    def test_column_mapping(self):
        """QMT 列名应正确映射为 Dpoint_Trader 内部列名。"""
        from dpoint.data.fetch.formatter import qmt_to_dpoint_single

        raw = pd.DataFrame({
            "time": [1609459200000, 1609545600000],  # 2021-01-01, 2021-01-02
            "open": [10.0, 10.5],
            "high": [10.5, 11.0],
            "low": [9.5, 10.0],
            "close": [10.2, 10.8],
            "volume": [100000, 120000],
            "amount": [1020000.0, 1296000.0],
        })
        df = qmt_to_dpoint_single(raw)

        assert "date" in df.columns
        assert "open_qfq" in df.columns
        assert "high_qfq" in df.columns
        assert "low_qfq" in df.columns
        assert "close_qfq" in df.columns
        assert "volume" in df.columns
        assert "amount" in df.columns

    def test_date_conversion(self):
        """毫秒时间戳应转换为 datetime。"""
        from dpoint.data.fetch.formatter import qmt_to_dpoint_single

        raw = pd.DataFrame({
            "time": [1609459200000],
            "open": [10.0], "high": [10.5], "low": [9.5], "close": [10.2],
            "volume": [100000], "amount": [1020000.0],
        })
        df = qmt_to_dpoint_single(raw)

        assert pd.api.types.is_datetime64_any_dtype(df["date"])
        assert df["date"].iloc[0] == pd.Timestamp("2021-01-01")

    def test_date_is_sorted(self):
        """输出 DataFrame 应按日期升序排列。"""
        from dpoint.data.fetch.formatter import qmt_to_dpoint_single

        raw = pd.DataFrame({
            "time": [1609545600000, 1609459200000],  # 反序
            "open": [10.5, 10.0],
            "high": [11.0, 10.5],
            "low": [10.0, 9.5],
            "close": [10.8, 10.2],
            "volume": [120000, 100000],
            "amount": [1296000.0, 1020000.0],
        })
        df = qmt_to_dpoint_single(raw)

        assert df["date"].is_monotonic_increasing

    def test_empty_dataframe(self):
        """空 DataFrame 应返回空结果且不报错。"""
        from dpoint.data.fetch.formatter import qmt_to_dpoint_single

        raw = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume", "amount"])
        df = qmt_to_dpoint_single(raw)

        assert len(df) == 0
        assert "open_qfq" in df.columns


class TestQmtToDpointCsv:
    """测试 qmt_to_dpoint_csv 转换函数（篮子 CSV 格式）。"""

    def test_csv_column_names(self):
        """篮子 CSV 应使用 Dpoint_Trader 外部列名。"""
        from dpoint.data.fetch.formatter import qmt_to_dpoint_csv

        raw = pd.DataFrame({
            "time": [1609459200000],
            "open": [10.0], "high": [10.5], "low": [9.5], "close": [10.2],
            "volume": [100000], "amount": [1020000.0],
        })
        df = qmt_to_dpoint_csv(raw)

        assert "Date" in df.columns
        assert "Open (CNY, qfq)" in df.columns
        assert "High (CNY, qfq)" in df.columns
        assert "Low (CNY, qfq)" in df.columns
        assert "Close (CNY, qfq)" in df.columns
        assert "Volume (shares)" in df.columns

    def test_csv_date_format(self):
        """日期格式应为 YYYY/M/D（无前导零）。"""
        from dpoint.data.fetch.formatter import qmt_to_dpoint_csv

        raw = pd.DataFrame({
            "time": [1609459200000],  # 2021-01-01
            "open": [10.0], "high": [10.5], "low": [9.5], "close": [10.2],
            "volume": [100000], "amount": [1020000.0],
        })
        df = qmt_to_dpoint_csv(raw)

        assert df["Date"].iloc[0] == "2021/1/1"


class TestGenerateCsvFilename:
    """测试 CSV 文件名生成。"""

    def test_standard_format(self):
        """文件名格式应为 {6位代码}_{日期}.csv。"""
        from dpoint.data.fetch.formatter import generate_csv_filename

        name = generate_csv_filename("000001.SZ", "19910403")
        assert name == "000001_19910403.csv"

    def test_code_without_suffix(self):
        """无后缀代码应直接使用。"""
        from dpoint.data.fetch.formatter import generate_csv_filename

        name = generate_csv_filename("600519", "20010827")
        assert name == "600519_20010827.csv"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd J:/DpointTrader/Dpoint_Trader && python -m pytest tests/test_fetch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dpoint.data.fetch'`

- [ ] **Step 3: Implement formatter.py**

```python
# src/dpoint/data/fetch/__init__.py
"""数据获取模块：从 QMT 获取价格数据，从国泰安获取行业分类。"""
from __future__ import annotations
```

```python
# src/dpoint/data/fetch/formatter.py
"""
数据格式转换：将 QMT DataFrame 转换为 Dpoint_Trader 所需格式。

QMT 返回格式:
    列: time(ms), open, high, low, close, volume, amount

Dpoint_Trader 单股格式:
    列: date, open_qfq, high_qfq, low_qfq, close_qfq, volume, amount

Dpoint_Trader 篮子 CSV 格式:
    列: Date, Open (CNY, qfq), High (CNY, qfq), Low (CNY, qfq), Close (CNY, qfq), Volume (shares)
    日期格式: YYYY/M/D
"""
from __future__ import annotations

import pandas as pd


def qmt_to_dpoint_single(df: pd.DataFrame) -> pd.DataFrame:
    """
    转换 QMT DataFrame 为 Dpoint_Trader 单股格式。

    Args:
        df: QMT 返回的 DataFrame，列名为 time/open/high/low/close/volume/amount

    Returns:
        Dpoint_Trader 格式 DataFrame，列为 date/open_qfq/high_qfq/low_qfq/close_qfq/volume/amount
    """
    if df.empty:
        return pd.DataFrame(columns=["date", "open_qfq", "high_qfq", "low_qfq", "close_qfq", "volume", "amount"])

    result = pd.DataFrame()
    result["date"] = pd.to_datetime(df["time"], unit="ms", errors="coerce")
    result["open_qfq"] = df["open"].values
    result["high_qfq"] = df["high"].values
    result["low_qfq"] = df["low"].values
    result["close_qfq"] = df["close"].values
    result["volume"] = df["volume"].values
    if "amount" in df.columns:
        result["amount"] = df["amount"].values

    # 按日期排序
    result = result.sort_values("date").reset_index(drop=True)
    return result


def qmt_to_dpoint_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    转换 QMT DataFrame 为 Dpoint_Trader 篮子 CSV 格式。

    列名使用 Dpoint_Trader 的外部映射名（参见 core/constants.py DEFAULT_COLUMN_MAP）。
    日期格式为 YYYY/M/D（无前导零），与现有 basket_1/*.csv 一致。

    Args:
        df: QMT 返回的 DataFrame

    Returns:
        篮子 CSV 格式 DataFrame
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "Date", "Open (CNY, qfq)", "High (CNY, qfq)",
            "Low (CNY, qfq)", "Close (CNY, qfq)", "Volume (shares)",
        ])

    dates = pd.to_datetime(df["time"], unit="ms", errors="coerce")

    result = pd.DataFrame()
    result["Date"] = dates.dt.strftime("%Y/%-m/%-d")
    result["Open (CNY, qfq)"] = df["open"].values
    result["High (CNY, qfq)"] = df["high"].values
    result["Low (CNY, qfq)"] = df["low"].values
    result["Close (CNY, qfq)"] = df["close"].values
    result["Volume (shares)"] = df["volume"].astype(float).values

    return result


def generate_csv_filename(stock_code: str, start_date: str) -> str:
    """
    生成篮子 CSV 文件名。

    格式: {6位代码}_{日期}.csv
    示例: 000001_19910403.csv

    Args:
        stock_code: 股票代码，如 "000001.SZ" 或 "000001"
        start_date: 起始日期，格式 "YYYYMMDD"

    Returns:
        文件名字符串
    """
    code = stock_code.split(".")[0] if "." in stock_code else stock_code
    return f"{code}_{start_date}.csv"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd J:/DpointTrader/Dpoint_Trader && python -m pytest tests/test_fetch.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/dpoint/data/fetch/__init__.py src/dpoint/data/fetch/formatter.py tests/test_fetch.py
git commit -m "feat(fetch): add formatter for QMT→Dpoint_Trader column mapping"
```

---

## Task 2: industry.py — CSMAR Industry DB Query

**Files:**
- Create: `src/dpoint/data/fetch/industry.py`
- Modify: `tests/test_fetch.py`

The industry module queries the existing CSMAR SQLite database at `J:\Dandelions_investment_agent\storage\reference\csmar_industry.sqlite`. It reads the `industry_members` table to get stock lists for a given industry code.

- [ ] **Step 1: Write failing tests for industry module**

Append to `tests/test_fetch.py`:

```python
class TestIndustryDB:
    """测试行业分类数据库查询。"""

    def test_list_industries_returns_list(self):
        """list_industries 应返回行业列表。"""
        from dpoint.data.fetch.industry import IndustryDB

        # 使用 mock 或跳过（需要真实 DB）
        db_path = r"J:\Dandelions_investment_agent\storage\reference\csmar_industry.sqlite"
        if not Path(db_path).exists():
            pytest.skip("CSMAR SQLite not found")

        db = IndustryDB(db_path)
        industries = db.list_industries()

        assert len(industries) > 0
        assert "code" in industries[0]
        assert "name" in industries[0]
        assert "count" in industries[0]

    def test_get_industry_members_returns_list(self):
        """get_industries_members 应返回股票代码列表。"""
        from dpoint.data.fetch.industry import IndustryDB

        db_path = r"J:\Dandelions_investment_agent\storage\reference\csmar_industry.sqlite"
        if not Path(db_path).exists():
            pytest.skip("CSMAR SQLite not found")

        db = IndustryDB(db_path)
        members = db.get_industry_members("C27")

        assert len(members) > 0
        # 所有代码应为 CODE.MARKET 格式
        for code in members:
            assert "." in code, f"Expected CODE.MARKET format, got: {code}"

    def test_invalid_industry_code(self):
        """无效行业代码应返回空列表。"""
        from dpoint.data.fetch.industry import IndustryDB

        db_path = r"J:\Dandelions_investment_agent\storage\reference\csmar_industry.sqlite"
        if not Path(db_path).exists():
            pytest.skip("CSMAR SQLite not found")

        db = IndustryDB(db_path)
        members = db.get_industry_members("ZZZZ99")

        assert len(members) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd J:/DpointTrader/Dpoint_Trader && python -m pytest tests/test_fetch.py::TestIndustryDB -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dpoint.data.fetch.industry'`

- [ ] **Step 3: Implement industry.py**

```python
# src/dpoint/data/fetch/industry.py
"""
行业分类数据库查询。

从国泰安 CSMAR SQLite 数据库查询行业成员股票列表。
数据库来源: J:\Dandelions_investment_agent\storage\reference\csmar_industry.sqlite

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

# 默认数据库路径（复用 Dandelions 仓库的已构建数据库）
DEFAULT_DB_PATH = r"J:\Dandelions_investment_agent\storage\reference\csmar_industry.sqlite"


@dataclass
class IndustryInfo:
    """行业信息。"""
    code: str       # 行业代码，如 "C27"
    name: str       # 行业名称，如 "医药制造业"
    count: int      # 成员股票数量


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

    def close(self) -> None:
        """关闭数据库连接。"""
        self._conn.close()

    def list_industries(self, industry_level: str = "CSMAR_ZX") -> List[IndustryInfo]:
        """
        列出所有行业代码、名称和成员数量。

        Args:
            industry_level: 行业分类级别，"CSMAR_ZX"（中信）或 "CSMAR_SECTION"（大类）

        Returns:
            IndustryInfo 列表，按成员数量降序排列
        """
        sql = """
            SELECT industry_code, industry_name, COUNT(*) as cnt
            FROM industry_members
            WHERE industry_level = ? AND is_active = 1
            GROUP BY industry_code, industry_name
            ORDER BY cnt DESC
        """
        rows = self._conn.execute(sql, (industry_level,)).fetchall()
        return [IndustryInfo(code=r["industry_code"], name=r["industry_name"], count=r["cnt"]) for r in rows]

    def get_industry_members(
        self,
        industry_code: str,
        industry_level: str = "CSMAR_ZX",
        active_only: bool = True,
        exclude_st: bool = True,
    ) -> List[str]:
        """
        获取指定行业的所有股票代码。

        Args:
            industry_code: 行业代码，如 "C27"
            industry_level: 行业分类级别
            active_only: 仅返回活跃股票
            exclude_st: 排除 ST 股票

        Returns:
            股票代码列表，格式为 "CODE.MARKET"（如 "000001.SZ"）
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

        Args:
            symbol: 股票代码，格式 "CODE.MARKET"
            industry_level: 行业分类级别

        Returns:
            包含 industry_code, industry_name 等字段的字典
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd J:/DpointTrader/Dpoint_Trader && python -m pytest tests/test_fetch.py::TestIndustryDB -v`
Expected: All 3 tests PASS (or SKIP if SQLite not found)

- [ ] **Step 5: Commit**

```bash
git add src/dpoint/data/fetch/industry.py tests/test_fetch.py
git commit -m "feat(fetch): add CSMAR industry DB query module"
```

---

## Task 3: qmt_client.py — QMT Data Client

**Files:**
- Create: `src/dpoint/data/fetch/qmt_client.py`
- Modify: `tests/test_fetch.py`

The QMT client wraps `xtquant.xtdata` to provide a clean interface for downloading and reading historical price data. It handles connection checking, data download, and format conversion.

- [ ] **Step 1: Write failing tests for QMT client**

Append to `tests/test_fetch.py`:

```python
class TestQMTClient:
    """测试 QMT 客户端（需要 XtMiniQMT 运行）。"""

    def test_import_xtquant(self):
        """应能导入 xtquant 库。"""
        try:
            from xtquant import xtdata  # noqa: F401
        except ImportError:
            pytest.skip("xtquant not installed (requires QMT)")

    def test_fetch_single_stock(self):
        """应能获取单只股票的历史数据。"""
        from dpoint.data.fetch.qmt_client import QMTClient

        client = QMTClient()
        df = client.fetch_daily_history("000001.SZ", start_date="20240101", end_date="20240110")

        assert not df.empty
        assert "time" in df.columns
        assert "open" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns

    def test_fetch_batch(self):
        """应能批量获取多只股票数据。"""
        from dpoint.data.fetch.qmt_client import QMTClient

        client = QMTClient()
        result = client.fetch_batch(
            ["000001.SZ", "600519.SH"],
            start_date="20240101",
            end_date="20240110",
        )

        assert isinstance(result, dict)
        assert len(result) == 2
        assert "000001.SZ" in result
        assert "600519.SH" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd J:/DpointTrader/Dpoint_Trader && python -m pytest tests/test_fetch.py::TestQMTClient -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement qmt_client.py**

```python
# src/dpoint/data/fetch/qmt_client.py
"""
QMT 数据获取客户端。

封装 xtquant.xtdata API，提供历史价格数据获取接口。
前提条件: XtMiniQMT 必须在后台运行并登录。

典型用法:
    client = QMTClient()
    df = client.fetch_daily_history("000001.SZ", start_date="20200101")
"""
from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


class QMTClient:
    """QMT 数据获取客户端。"""

    def __init__(self):
        """
        初始化 QMT 客户端。

        Raises:
            ImportError: xtquant 未安装
            RuntimeError: 无法连接到 XtMiniQMT
        """
        try:
            from xtquant import xtdata
            self._xtdata = xtdata
        except ImportError:
            raise ImportError(
                "xtquant 未安装。请确认已安装 QMT/MiniQMT 客户端，"
                "并将 xtquant 所在目录添加到 Python 路径。"
            )

    def fetch_daily_history(
        self,
        stock_code: str,
        period: str = "1d",
        start_date: str = "",
        end_date: str = "",
        dividend_type: str = "front",
    ) -> pd.DataFrame:
        """
        获取单只股票的历史日线数据。

        流程:
        1. download_history_data() 下载数据到本地缓存
        2. get_market_data_ex() 从缓存读取

        Args:
            stock_code: 股票代码，格式 "CODE.MARKET"（如 "000001.SZ"）
            period: K线周期，默认 "1d"（日线）
            start_date: 起始日期，格式 "YYYYMMDD"，默认获取所有可用数据
            end_date: 结束日期，格式 "YYYYMMDD"，默认获取到最新
            dividend_type: 复权类型，默认 "front"（前复权）

        Returns:
            DataFrame，列为 time/open/high/low/close/volume/amount
        """
        logger.info("Fetching %s (%s ~ %s, %s)", stock_code, start_date or "all", end_date or "latest", period)

        # Step 1: 下载到本地缓存
        try:
            self._xtdata.download_history_data(
                stock_code=stock_code,
                period=period,
                start_time=start_date,
                end_time=end_date,
            )
        except TypeError:
            # 兼容不同版本的 xtquant：部分版本不支持 keyword-only 参数
            self._xtdata.download_history_data(stock_code, period, start_date, end_date)

        # Step 2: 从缓存读取
        data = self._xtdata.get_market_data_ex(
            field_list=[],
            stock_list=[stock_code],
            period=period,
            start_time=start_date,
            end_time=end_date,
            count=-1,
            dividend_type=dividend_type,
            fill_data=True,
        )

        if stock_code not in data or data[stock_code].empty:
            logger.warning("No data returned for %s", stock_code)
            return pd.DataFrame()

        df = data[stock_code].copy()

        # 标准化列名（xtquant 可能返回混合大小写）
        df.columns = [str(c).lower() for c in df.columns]

        # 确保 time 列存在
        if "time" not in df.columns and df.index.name in ("time", "timetag"):
            df = df.reset_index()

        logger.info("Fetched %d rows for %s", len(df), stock_code)
        return df

    def fetch_batch(
        self,
        stock_codes: List[str],
        period: str = "1d",
        start_date: str = "",
        end_date: str = "",
        dividend_type: str = "front",
    ) -> Dict[str, pd.DataFrame]:
        """
        批量获取多只股票的历史数据。

        Args:
            stock_codes: 股票代码列表
            period: K线周期
            start_date: 起始日期
            end_date: 结束日期
            dividend_type: 复权类型

        Returns:
            {stock_code: DataFrame} 字典
        """
        logger.info("Batch fetching %d stocks", len(stock_codes))

        result = {}
        for i, code in enumerate(stock_codes, 1):
            logger.info("[%d/%d] Fetching %s", i, len(stock_codes), code)
            try:
                df = self.fetch_daily_history(code, period, start_date, end_date, dividend_type)
                if not df.empty:
                    result[code] = df
                else:
                    logger.warning("Empty data for %s, skipping", code)
            except Exception as e:
                logger.error("Failed to fetch %s: %s", code, e)

        logger.info("Batch complete: %d/%d succeeded", len(result), len(stock_codes))
        return result
```

- [ ] **Step 4: Run tests**

Run: `cd J:/DpointTrader/Dpoint_Trader && python -m pytest tests/test_fetch.py::TestQMTClient -v`
Expected: PASS (if QMT running) or SKIP (if xtquant not installed)

- [ ] **Step 5: Commit**

```bash
git add src/dpoint/data/fetch/qmt_client.py tests/test_fetch.py
git commit -m "feat(fetch): add QMT data client with download+read API"
```

---

## Task 4: CLI Registration — `dpoint fetch` Command

**Files:**
- Modify: `src/dpoint/cli/main.py:28-93` (add fetch subcommand parser)
- Modify: `src/dpoint/cli/main.py:790-807` (add fetch dispatch)

- [ ] **Step 1: Add fetch subcommand parser in `build_parser()`**

After line 91 (end of resume parser), add:

```python
    # === dpoint fetch ===
    fetch = subparsers.add_parser("fetch", help="自动获取价格数据（需要 XtMiniQMT 运行）")
    fetch_sub = fetch.add_subparsers(dest="fetch_mode", help="获取模式")

    # dpoint fetch single
    fetch_single = fetch_sub.add_parser("single", help="获取单只股票历史数据")
    fetch_single.add_argument("--code", required=True, help="股票代码，如 000001.SZ")
    fetch_single.add_argument("--start", default="", help="起始日期 YYYYMMDD（默认6年前）")
    fetch_single.add_argument("--end", default="", help="结束日期 YYYYMMDD（默认今天）")
    fetch_single.add_argument("--output", default="", help="输出文件路径")
    fetch_single.add_argument("--format", default="xlsx", choices=["xlsx", "csv"], help="输出格式")

    # dpoint fetch basket
    fetch_basket = fetch_sub.add_parser("basket", help="获取行业篮子数据")
    fetch_basket.add_argument("--industry", required=True, help="行业代码，如 C27")
    fetch_basket.add_argument("--start", default="", help="起始日期 YYYYMMDD（默认6年前）")
    fetch_basket.add_argument("--end", default="", help="结束日期 YYYYMMDD（默认今天）")
    fetch_basket.add_argument("--output", default="", help="输出目录路径")
    fetch_basket.add_argument("--format", default="csv", choices=["xlsx", "csv"], help="输出格式")
    fetch_basket.add_argument("--db", default="", help="行业分类 SQLite 路径")
```

- [ ] **Step 2: Add `run_fetch_single()` function**

Add before `main()`:

```python
def run_fetch_single(args) -> int:
    """获取单只股票历史数据。"""
    logger = logging.getLogger("dpoint.fetch.single")

    from dpoint.data.fetch.formatter import qmt_to_dpoint_single
    from dpoint.data.fetch.qmt_client import QMTClient

    # 确定输出路径
    if args.output:
        output_path = Path(args.output)
    else:
        from datetime import datetime, timedelta
        end = args.end or datetime.now().strftime("%Y%m%d")
        if args.start:
            start = args.start
        else:
            start = (datetime.now() - timedelta(days=365 * 6)).strftime("%Y%m%d")
        code_clean = args.code.replace(".", "_")
        ext = "xlsx" if args.format == "xlsx" else "csv"
        output_path = Path("data") / f"{code_clean}_{start}_{end}.{ext}"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 获取数据
    logger.info("Fetching %s from QMT...", args.code)
    try:
        client = QMTClient()
    except ImportError as e:
        logger.error(str(e))
        return 1

    raw_df = client.fetch_daily_history(args.code, start_date=args.start, end_date=args.end)
    if raw_df.empty:
        logger.error("未获取到 %s 的数据", args.code)
        return 1

    # 转换格式
    df = qmt_to_dpoint_single(raw_df)
    logger.info("Converted to Dpoint_Trader format: %d rows", len(df))

    # 保存
    if output_path.suffix in (".xlsx", ".xls"):
        df.to_excel(output_path, index=False, engine="openpyxl")
    else:
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

    logger.info("Saved to: %s", output_path)
    logger.info("可直接用于: dpoint single --data_path %s", output_path)
    return 0
```

- [ ] **Step 3: Add `run_fetch_basket()` function**

```python
def run_fetch_basket(args) -> int:
    """获取行业篮子数据。"""
    logger = logging.getLogger("dpoint.fetch.basket")

    from dpoint.data.fetch.formatter import generate_csv_filename, qmt_to_dpoint_csv
    from dpoint.data.fetch.industry import DEFAULT_DB_PATH, IndustryDB
    from dpoint.data.fetch.qmt_client import QMTClient

    # 确定数据库路径
    db_path = args.db if args.db else DEFAULT_DB_PATH

    # 查询行业成员
    try:
        db = IndustryDB(db_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    members = db.get_industry_members(args.industry)
    if not members:
        logger.error("行业代码 '%s' 未找到任何股票", args.industry)
        logger.info("可用行业示例:")
        for info in db.list_industries()[:10]:
            logger.info("  %s %s (%d只)", info.code, info.name, info.count)
        return 1

    logger.info("行业 %s 共 %d 只股票", args.industry, len(members))

    # 确定输出目录
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path("data") / f"basket_{args.industry}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 批量获取
    try:
        client = QMTClient()
    except ImportError as e:
        logger.error(str(e))
        return 1

    # 确定起始日期（用于文件名）
    from datetime import datetime, timedelta
    start = args.start or (datetime.now() - timedelta(days=365 * 6)).strftime("%Y%m%d")

    data = client.fetch_batch(members, start_date=args.start, end_date=args.end)

    # 保存
    saved = 0
    for code, raw_df in data.items():
        df = qmt_to_dpoint_csv(raw_df)
        filename = generate_csv_filename(code, start)
        filepath = output_dir / filename
        df.to_csv(filepath, index=False, encoding="utf-8-sig")
        saved += 1

    logger.info("Saved %d stocks to: %s", saved, output_dir)
    logger.info("可直接用于: dpoint basket --basket_path %s", output_dir)
    return 0
```

- [ ] **Step 4: Add fetch dispatch in `main()`**

In the `main()` function, after the `elif args.command == "resume":` block (line 805), add:

```python
    elif args.command == "fetch":
        if not args.fetch_mode:
            parser.parse_args(["fetch", "--help"])
            return 1
        if args.fetch_mode == "single":
            return run_fetch_single(args)
        elif args.fetch_mode == "basket":
            return run_fetch_basket(args)
```

- [ ] **Step 5: Verify CLI help works**

Run: `cd J:/DpointTrader/Dpoint_Trader && python -m dpoint fetch --help`
Expected: Shows fetch subcommand help with `single` and `basket` options

Run: `cd J:/DpointTrader/Dpoint_Trader && python -m dpoint fetch single --help`
Expected: Shows single stock fetch options (--code, --start, --end, --output, --format)

- [ ] **Step 6: Commit**

```bash
git add src/dpoint/cli/main.py
git commit -m "feat(fetch): register dpoint fetch single/basket CLI commands"
```

---

## Task 5: Integration Test — End-to-End Verification

**Files:**
- Modify: `tests/test_fetch.py`

This task adds integration tests that verify the full pipeline works with real QMT data.

- [ ] **Step 1: Write integration test**

Append to `tests/test_fetch.py`:

```python
class TestFetchIntegration:
    """端到端集成测试（需要 XtMiniQMT 运行）。"""

    def test_single_stock_end_to_end(self):
        """完整流程: 获取 → 转换 → 验证格式。"""
        from dpoint.data.fetch.formatter import qmt_to_dpoint_single
        from dpoint.data.fetch.qmt_client import QMTClient

        client = QMTClient()
        raw = client.fetch_daily_history("000001.SZ", start_date="20240101", end_date="20240131")
        if raw.empty:
            pytest.skip("QMT returned empty data")

        df = qmt_to_dpoint_single(raw)

        # 验证 Dpoint_Trader 所需列存在
        for col in ["date", "open_qfq", "high_qfq", "low_qfq", "close_qfq", "volume"]:
            assert col in df.columns, f"Missing column: {col}"

        # 验证数据类型
        assert pd.api.types.is_datetime64_any_dtype(df["date"])
        assert pd.api.types.is_numeric_dtype(df["open_qfq"])

    def test_basket_csv_roundtrip(self):
        """验证生成的 CSV 能被 basket_loader 正确加载。"""
        import tempfile

        from dpoint.data.fetch.formatter import generate_csv_filename, qmt_to_dpoint_csv
        from dpoint.data.fetch.qmt_client import QMTClient

        client = QMTClient()
        raw = client.fetch_daily_history("600519.SH", start_date="20240101", end_date="20240131")
        if raw.empty:
            pytest.skip("QMT returned empty data")

        csv_df = qmt_to_dpoint_csv(raw)

        # 写入临时文件
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / generate_csv_filename("600519.SH", "20240101")
            csv_df.to_csv(filepath, index=False, encoding="utf-8-sig")

            # 用 csv_loader 加载验证
            from dpoint.data.csv_loader import load_single_csv
            loaded_df, report = load_single_csv(filepath, ticker="600519")
            assert len(loaded_df) > 0
            assert "close_qfq" in loaded_df.columns
```

- [ ] **Step 2: Run all tests**

Run: `cd J:/DpointTrader/Dpoint_Trader && python -m pytest tests/test_fetch.py -v`
Expected: All tests PASS or SKIP (if QMT not running)

- [ ] **Step 3: Manual verification with real QMT**

```bash
# 单股测试
cd J:/DpointTrader/Dpoint_Trader
python -m dpoint fetch single --code 000001.SZ --start 20240101 --end 20240601

# 验证生成的文件
python -c "
import pandas as pd
df = pd.read_excel('data/000001_SZ_20240101_20240601.xlsx')
print(df.head())
print('Columns:', df.columns.tolist())
print('Shape:', df.shape)
"

# 篮子测试
python -m dpoint fetch basket --industry C27 --start 20240101 --end 20240601

# 验证生成的目录
ls data/basket_C27/
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_fetch.py
git commit -m "test(fetch): add integration tests for fetch pipeline"
```

---

## Task 6: Final Cleanup & Documentation

**Files:**
- Modify: `src/dpoint/data/fetch/__init__.py`

- [ ] **Step 1: Update __init__.py with public API**

```python
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
```

- [ ] **Step 2: Run full test suite**

Run: `cd J:/DpointTrader/Dpoint_Trader && python -m pytest tests/ -v --tb=short`
Expected: All existing tests + new fetch tests PASS

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete dpoint fetch command for automatic data acquisition"
```
