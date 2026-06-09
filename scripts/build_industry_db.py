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
