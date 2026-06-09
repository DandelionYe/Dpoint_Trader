# 篮子模式多维度分类筛选 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 重构篮子数据获取的行业分类系统，从单一行业维度扩展为 7 个可组合维度（4 级行业 + 省份 + 城市 + 所有权）。

**Architecture:** 从 TRD_Co.csv 构建 SQLite 数据库（单表扁平设计），重写 IndustryDB 类支持多维度查询，CLI 新增 7 个可选筛选参数。

**Tech Stack:** Python 3.10+, sqlite3, pandas, argparse

---

## File Structure

| 文件 | 职责 |
|------|------|
| `scripts/build_industry_db.py` | 从 TRD_Co.csv 构建 SQLite，清洗+补零+建表+索引 |
| `data/csmar_industry.sqlite` | 生成的 SQLite 数据库 |
| `src/dpoint/data/fetch/industry.py` | 重写：新 IndustryDB 接口（list_values, query_stocks, resolve_stock） |
| `src/dpoint/cli/main.py` | 修改：新增筛选参数 + list 命令 + 重构 run_fetch_basket |
| `tests/test_industry.py` | 新建：IndustryDB 新接口单元测试 |

---

### Task 1: 创建 SQLite 构建脚本

**Files:**
- Create: `scripts/build_industry_db.py`

- [ ] **Step 1: 创建 scripts 目录并写入构建脚本**

```python
"""
从 TRD_Co.csv 构建行业分类 SQLite 数据库。

用法:
    python scripts/build_industry_db.py

输入: data/TRD_Co.csv
输出: data/csmar_industry.sqlite
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

# 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "TRD_Co.csv"
DB_PATH = PROJECT_ROOT / "data" / "csmar_industry.sqlite"

# CSV 列名 → SQLite 列名映射
COLUMN_MAP = {
    "Stkcd": "code",
    "Stknme": "name",
    "Indcd": "ind1_code",
    "Indnme": "ind1_name",
    "Nindcd": "ind2_code",
    "Nindnme": "ind2_name",
    "Nnindcd": "ind3_code",
    "Nnindnme": "ind3_name",
    "IndcdZX": "ind4_code",
    "IndnmeZX": "ind4_name",
    "PROVINCE": "province",
    "PROVINCECODE": "province_code",
    "CITY": "city",
    "CITYCODE": "city_code",
    "OWNERSHIPTYPE": "ownership",
    "OWNERSHIPTYPECODE": "ownership_code",
}

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS stocks (
    code           TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    ind1_code      TEXT,
    ind1_name      TEXT,
    ind2_code      TEXT,
    ind2_name      TEXT,
    ind3_code      TEXT,
    ind3_name      TEXT,
    ind4_code      TEXT,
    ind4_name      TEXT,
    province       TEXT,
    province_code  TEXT,
    city           TEXT,
    city_code      TEXT,
    ownership      TEXT,
    ownership_code TEXT
);

CREATE INDEX IF NOT EXISTS idx_ind1 ON stocks(ind1_code);
CREATE INDEX IF NOT EXISTS idx_ind2 ON stocks(ind2_code);
CREATE INDEX IF NOT EXISTS idx_ind3 ON stocks(ind3_code);
CREATE INDEX IF NOT EXISTS idx_ind4 ON stocks(ind4_code);
CREATE INDEX IF NOT EXISTS idx_province ON stocks(province);
CREATE INDEX IF NOT EXISTS idx_city ON stocks(city);
CREATE INDEX IF NOT EXISTS idx_ownership ON stocks(ownership);
"""


def build() -> None:
    """读取 CSV，清洗，写入 SQLite。"""
    print(f"读取: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig", dtype=str)

    # 只保留需要的列
    csv_cols = list(COLUMN_MAP.keys())
    missing = set(csv_cols) - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少列: {missing}")
    df = df[csv_cols].rename(columns=COLUMN_MAP)

    # 清洗：过滤空代码，补零到 6 位
    df = df.dropna(subset=["code"])
    df["code"] = df["code"].str.strip().str.zfill(6)

    # 将 NaN 替换为 None（SQLite NULL）
    df = df.where(df.notna(), None)

    print(f"有效记录: {len(df)}")

    # 写入 SQLite
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript(CREATE_SQL)

    # 批量插入
    cols = list(COLUMN_MAP.values())
    placeholders = ", ".join(["?"] * len(cols))
    insert_sql = f"INSERT INTO stocks ({', '.join(cols)}) VALUES ({placeholders})"

    rows = df[cols].values.tolist()
    conn.executemany(insert_sql, rows)
    conn.commit()

    # 验证
    count = conn.execute("SELECT COUNT(*) FROM stocks").fetchone()[0]
    print(f"已写入 {count} 条记录到: {DB_PATH}")

    # 统计各维度
    for dim in ["ind1", "ind2", "ind3", "ind4", "province", "city", "ownership"]:
        n = conn.execute(f"SELECT COUNT(DISTINCT {dim}_code) FROM stocks").fetchone()[0]
        print(f"  {dim}: {n} 个分类")

    conn.close()
    print("完成。")


if __name__ == "__main__":
    build()
```

- [ ] **Step 2: 运行构建脚本生成 SQLite**

Run: `cd J:/DpointTrader/Dpoint_Trader && python scripts/build_industry_db.py`

Expected output:
```
读取: data/TRD_Co.csv
有效记录: 5963
已写入 5963 条记录到: data/csmar_industry.sqlite
  ind1: 6 个分类
  ind2: 72 个分类
  ...
完成。
```

- [ ] **Step 3: 验证生成的数据库**

Run: `python -c "import sqlite3; conn = sqlite3.connect('data/csmar_industry.sqlite'); print(conn.execute('SELECT code, name, ind4_code, province FROM stocks LIMIT 5').fetchall()); conn.close()"`

Expected: 6 位代码 + 名称 + 行业代码 + 省份的元组列表

- [ ] **Step 4: 提交**

```bash
git add scripts/build_industry_db.py data/csmar_industry.sqlite
git commit -m "feat: add industry SQLite build script and generated database"
```

---

### Task 2: 重写 IndustryDB

**Files:**
- Rewrite: `src/dpoint/data/fetch/industry.py`

- [ ] **Step 1: 写入新的 industry.py**

```python
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
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
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
```

- [ ] **Step 2: 运行基础验证**

Run: `python -c "
from dpoint.data.fetch.industry import IndustryDB
with IndustryDB() as db:
    vals = db.list_values('ind4')
    print(f'ind4 维度: {len(vals)} 个值')
    print(f'前3: {[(v.code, v.name, v.count) for v in vals[:3]]}')
    codes = db.query_stocks(ind4='C27')
    print(f'C27 成分股: {len(codes)} 只')
    codes2 = db.query_stocks(ind4='C27', province='广东省')
    print(f'C27+广东省: {len(codes2)} 只')
    info = db.resolve_stock('000001')
    print(f'000001: {info.get(\"name\")}, {info.get(\"ind4_name\")}, {info.get(\"province\")}')
"`

Expected: 行业列表、筛选结果、单股信息均正常返回

- [ ] **Step 3: 提交**

```bash
git add src/dpoint/data/fetch/industry.py
git commit -m "feat: rewrite IndustryDB with 7-dimension query support"
```

---

### Task 3: 更新 CLI — 新增筛选参数和 list 命令

**Files:**
- Modify: `src/dpoint/cli/main.py:92-114` (fetch 子命令定义)
- Modify: `src/dpoint/cli/main.py:876-938` (run_fetch_basket 函数)
- Modify: `src/dpoint/cli/main.py:957-973` (main dispatch)

- [ ] **Step 1: 更新 fetch 子命令参数定义**

将 `main.py` 中 `build_parser()` 的 fetch basket 部分（约 105-113 行）替换为：

```python
    # dpoint fetch basket
    fetch_basket = fetch_sub.add_parser("basket", help="获取篮子数据（多维度筛选）")
    # 筛选参数（全部可选）
    fetch_basket.add_argument("--ind1", default="", help="一级行业代码或名称")
    fetch_basket.add_argument("--ind2", default="", help="二级行业代码或名称")
    fetch_basket.add_argument("--ind3", default="", help="三级行业代码或名称")
    fetch_basket.add_argument("--ind4", default="", help="四级行业代码或名称（中信）")
    fetch_basket.add_argument("--industry", default="", help="别名，等同于 --ind4（向后兼容）")
    fetch_basket.add_argument("--province", default="", help="省份名称")
    fetch_basket.add_argument("--city", default="", help="城市名称")
    fetch_basket.add_argument("--ownership", default="", help="所有权类型")
    # 其他选项
    fetch_basket.add_argument("--start", default="", help="起始日期 YYYYMMDD（默认6年前）")
    fetch_basket.add_argument("--end", default="", help="结束日期 YYYYMMDD（默认今天）")
    fetch_basket.add_argument("--output", default="", help="输出目录路径")
    fetch_basket.add_argument("--format", default="csv", choices=["xlsx", "csv"], help="输出格式")
    fetch_basket.add_argument("--db", default="", help="行业分类 SQLite 路径")

    # dpoint fetch list-industries
    fetch_list_ind = fetch_sub.add_parser("list-industries", help="列出可用行业")
    fetch_list_ind.add_argument("--level", type=int, choices=[1, 2, 3, 4], help="行业级别（默认全部）")

    # dpoint fetch list-provinces
    fetch_sub.add_parser("list-provinces", help="列出可用省份")

    # dpoint fetch list-cities
    fetch_list_cities = fetch_sub.add_parser("list-cities", help="列出可用城市")
    fetch_list_cities.add_argument("--province", default="", help="按省份筛选")

    # dpoint fetch list-ownership
    fetch_sub.add_parser("list-ownership", help="列出可用所有权类型")
```

- [ ] **Step 2: 添加辅助 list 命令的实现函数**

在 `run_fetch_basket` 函数之前添加：

```python
def run_fetch_list(args) -> int:
    """列出分类维度的可选值。"""
    from dpoint.data.fetch.industry import DEFAULT_DB_PATH, IndustryDB

    db_path = args.db if hasattr(args, "db") and args.db else DEFAULT_DB_PATH

    try:
        with IndustryDB(db_path) as db:
            fetch_mode = args.fetch_mode

            if fetch_mode == "list-industries":
                levels = [args.level] if args.level else [1, 2, 3, 4]
                for level in levels:
                    dim = f"ind{level}"
                    values = db.list_values(dim)
                    print(f"\n=== {level} 级行业 ({dim}) === 共 {len(values)} 个")
                    for v in values:
                        print(f"  {v.code:8s} {v.name:20s} ({v.count} 只)")

            elif fetch_mode == "list-provinces":
                values = db.list_values("province")
                print(f"\n=== 省份 === 共 {len(values)} 个")
                for v in values:
                    print(f"  {v.code:8s} {v.name:10s} ({v.count} 只)")

            elif fetch_mode == "list-cities":
                if args.province:
                    codes = db.query_stocks(province=args.province)
                    # 用 province 筛选后列出城市
                    values = db.list_values("city")
                    # 过滤属于该省份的城市（通过查询股票数量）
                    print(f"\n=== {args.province} 的城市 ===")
                    for v in values:
                        city_codes = db.query_stocks(city=v.name, province=args.province)
                        if city_codes:
                            print(f"  {v.code:8s} {v.name:10s} ({len(city_codes)} 只)")
                else:
                    values = db.list_values("city")
                    print(f"\n=== 城市 === 共 {len(values)} 个")
                    for v in values:
                        print(f"  {v.code:8s} {v.name:10s} ({v.count} 只)")

            elif fetch_mode == "list-ownership":
                values = db.list_values("ownership")
                print(f"\n=== 所有权类型 === 共 {len(values)} 个")
                for v in values:
                    print(f"  {v.code:8s} {v.name:10s} ({v.count} 只)")

    except FileNotFoundError as e:
        print(f"错误: {e}")
        return 1

    return 0
```

- [ ] **Step 3: 重写 run_fetch_basket 函数**

将 `run_fetch_basket`（约 876-938 行）替换为：

```python
def run_fetch_basket(args) -> int:
    """获取篮子数据（多维度筛选）。"""
    logger = logging.getLogger("dpoint.fetch.basket")

    from dpoint.data.fetch.formatter import (
        generate_csv_filename,
        qmt_to_dpoint_csv,
        qmt_to_dpoint_single,
    )
    from dpoint.data.fetch.industry import DEFAULT_DB_PATH, IndustryDB

    # 确定数据库路径
    db_path = args.db if args.db else DEFAULT_DB_PATH

    # 构建筛选条件
    filters: dict[str, str] = {}
    # --industry 是 --ind4 的别名
    ind4_value = args.ind4 or args.industry
    if ind4_value:
        filters["ind4"] = ind4_value
    for dim in ("ind1", "ind2", "ind3", "province", "city", "ownership"):
        val = getattr(args, dim, "")
        if val:
            filters[dim] = val

    if not filters:
        logger.error("至少需要指定一个筛选条件，如 --ind4 C27 或 --province 广东省")
        return 1

    # 查询股票
    try:
        with IndustryDB(db_path) as db:
            members = db.query_stocks(**filters)
            if not members:
                logger.error("筛选条件 %s 未找到任何股票", filters)
                return 1
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    # 构建输出目录名
    if args.output:
        output_dir = Path(args.output)
    else:
        parts = [f"{dim}_{val}" for dim, val in filters.items()]
        output_dir = Path("data") / f"basket_{'_'.join(parts)}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("筛选条件: %s, 共 %d 只股票", filters, len(members))

    # 批量获取
    client = _make_qmt_client()
    if client is None:
        return 1

    start, end = _default_date_range(args)
    data = client.fetch_batch(members, start_date=start, end_date=end)

    # 保存
    saved = 0
    for code, raw_df in data.items():
        if args.format == "xlsx":
            df = qmt_to_dpoint_single(raw_df)
            stem = generate_csv_filename(code, start).replace(".csv", "")
            filepath = output_dir / f"{stem}.xlsx"
            df.to_excel(filepath, index=False, engine="openpyxl")
        else:
            df = qmt_to_dpoint_csv(raw_df)
            filename = generate_csv_filename(code, start)
            filepath = output_dir / filename
            df.to_csv(filepath, index=False, encoding="utf-8-sig")
        saved += 1

    logger.info("Saved %d stocks to: %s", saved, output_dir)
    logger.info("可直接用于: dpoint basket --basket_path %s", output_dir)
    return 0
```

- [ ] **Step 4: 更新 main dispatch**

将 `main()` 函数中的 fetch dispatch 部分（约 957-973 行）替换为：

```python
    elif args.command == "fetch":
        if not args.fetch_mode:
            try:
                parser.parse_args(["fetch", "--help"])
            except SystemExit:
                pass
            return 1
        if args.fetch_mode == "single":
            return run_fetch_single(args)
        elif args.fetch_mode == "basket":
            return run_fetch_basket(args)
        elif args.fetch_mode in (
            "list-industries",
            "list-provinces",
            "list-cities",
            "list-ownership",
        ):
            return run_fetch_list(args)
        else:
            logger.error("未知的 fetch 模式: %s", args.fetch_mode)
            return 1
```

- [ ] **Step 5: 验证 CLI help 输出**

Run: `python -m dpoint.cli.main fetch basket --help`

Expected: 显示 --ind1 到 --ind4、--province、--city、--ownership 等参数

Run: `python -m dpoint.cli.main fetch list-industries --help`

Expected: 显示 --level 参数

- [ ] **Step 6: 提交**

```bash
git add src/dpoint/cli/main.py
git commit -m "feat: add multi-dimensional basket filtering to CLI"
```

---

### Task 4: 编写 IndustryDB 单元测试

**Files:**
- Create: `tests/test_industry.py`

- [ ] **Step 1: 写入测试文件**

```python
"""IndustryDB 多维度查询接口的单元测试。"""
from __future__ import annotations

from pathlib import Path

import pytest

# 测试用 SQLite 路径
_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "csmar_industry.sqlite"


@pytest.fixture
def db():
    """提供 IndustryDB 实例，测试后自动关闭。"""
    if not _DB_PATH.exists():
        pytest.skip("csmar_industry.sqlite not found, run scripts/build_industry_db.py first")
    from dpoint.data.fetch.industry import IndustryDB

    conn = IndustryDB(_DB_PATH)
    yield conn
    conn.close()


class TestListValues:
    """测试 list_values 方法。"""

    def test_list_ind1(self, db):
        """一级行业应有 6 个分类。"""
        values = db.list_values("ind1")
        assert len(values) == 6
        codes = {v.code for v in values}
        assert "1" in codes  # 金融
        assert all(v.count > 0 for v in values)

    def test_list_ind4(self, db):
        """四级行业应有 83 个分类。"""
        values = db.list_values("ind4")
        assert len(values) == 83

    def test_list_province(self, db):
        """省份应有 34 个分类。"""
        values = db.list_values("province")
        assert len(values) == 34

    def test_list_ownership(self, db):
        """所有权类型应有 8 个分类。"""
        values = db.list_values("ownership")
        assert len(values) == 8

    def test_invalid_dimension(self, db):
        """无效维度应抛出 ValueError。"""
        with pytest.raises(ValueError, match="未知维度"):
            db.list_values("invalid_dim")

    def test_returns_sorted_by_count(self, db):
        """结果应按数量降序排列。"""
        values = db.list_values("ind1")
        counts = [v.count for v in values]
        assert counts == sorted(counts, reverse=True)


class TestQueryStocks:
    """测试 query_stocks 方法。"""

    def test_query_by_ind4_code(self, db):
        """按四级行业代码筛选。"""
        codes = db.query_stocks(ind4="C27")
        assert len(codes) > 0
        assert all(isinstance(c, str) and len(c) == 6 for c in codes)

    def test_query_by_ind4_name(self, db):
        """按四级行业名称筛选。"""
        # 先获取 C27 的名称
        values = db.list_values("ind4")
        c27_name = next(v.name for v in values if v.code == "C27")
        codes = db.query_stocks(ind4=c27_name)
        assert len(codes) > 0

    def test_query_by_province(self, db):
        """按省份筛选。"""
        codes = db.query_stocks(province="广东省")
        assert len(codes) > 0

    def test_query_by_ownership(self, db):
        """按所有权类型筛选。"""
        codes = db.query_stocks(ownership="私营企业")
        assert len(codes) > 0

    def test_query_multi_dimension(self, db):
        """多维度组合筛选应取交集。"""
        codes_all = db.query_stocks(ind4="C27")
        codes_gd = db.query_stocks(ind4="C27", province="广东省")
        assert len(codes_gd) <= len(codes_all)
        assert len(codes_gd) > 0

    def test_query_no_result(self, db):
        """不存在的筛选条件应返回空列表。"""
        codes = db.query_stocks(ind4="ZZZZ99")
        assert codes == []

    def test_query_invalid_dimension(self, db):
        """无效维度应抛出 ValueError。"""
        with pytest.raises(ValueError, match="未知筛选维度"):
            db.query_stocks(invalid="test")

    def test_query_no_filters(self, db):
        """无筛选条件应返回全部股票。"""
        codes = db.query_stocks()
        assert len(codes) > 5000  # 总共约 5963 只


class TestResolveStock:
    """测试 resolve_stock 方法。"""

    def test_resolve_existing_stock(self, db):
        """查询存在的股票应返回完整信息。"""
        info = db.resolve_stock("000001")
        assert info["code"] == "000001"
        assert info["name"] == "平安银行"
        assert info["ind4_code"] is not None
        assert info["province"] is not None

    def test_resolve_with_suffix(self, db):
        """带后缀的代码应自动去除后缀。"""
        info = db.resolve_stock("000001.SZ")
        assert info["code"] == "000001"

    def test_resolve_nonexistent(self, db):
        """不存在的代码应返回空字典。"""
        info = db.resolve_stock("999999")
        assert info == {}

    def test_resolve_padded(self, db):
        """不补零的代码应自动补零。"""
        info = db.resolve_stock("1")
        assert info["code"] == "000001"


class TestIndustryDBContextManager:
    """测试上下文管理器。"""

    def test_context_manager(self):
        """with 语句应自动关闭连接。"""
        if not _DB_PATH.exists():
            pytest.skip("csmar_industry.sqlite not found")
        from dpoint.data.fetch.industry import IndustryDB

        with IndustryDB(_DB_PATH) as db:
            values = db.list_values("ind1")
            assert len(values) > 0

    def test_file_not_found(self):
        """不存在的数据库应抛出 FileNotFoundError。"""
        from dpoint.data.fetch.industry import IndustryDB

        with pytest.raises(FileNotFoundError, match="不存在"):
            IndustryDB("/nonexistent/path.sqlite")
```

- [ ] **Step 2: 运行测试**

Run: `pytest tests/test_industry.py -v`

Expected: 全部 PASS（如果 SQLite 不存在则 SKIP）

- [ ] **Step 3: 提交**

```bash
git add tests/test_industry.py
git commit -m "test: add IndustryDB multi-dimensional query tests"
```

---

### Task 5: 端到端验证

- [ ] **Step 1: 运行全部测试确保无回归**

Run: `pytest tests/ -v --tb=short`

Expected: 全部 PASS 或 SKIP（QMT 相关测试可能 SKIP）

- [ ] **Step 2: 验证 CLI list 命令**

Run: `python -m dpoint.cli.main fetch list-industries --level 4`

Expected: 列出 83 个四级行业

- [ ] **Step 3: 验证向后兼容（--industry 参数）**

Run: `python -m dpoint.cli.main fetch basket --industry C27 --help`

Expected: 正常解析（不做实际获取，只验证参数解析）

- [ ] **Step 4: 最终提交**

```bash
git add -A
git commit -m "feat: complete basket multi-dimensional classification system"
```
