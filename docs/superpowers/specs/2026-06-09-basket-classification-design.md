# 篮子模式多维度分类筛选设计

**日期**: 2026-06-09
**状态**: 已批准
**范围**: 重构篮子数据获取的行业分类系统，从单一行业维度扩展为 7 个可组合维度

## 背景

当前 `dpoint fetch basket --industry C27` 仅支持按中信四级行业分类筛选。用户将国泰安 TRD_Co.csv 原始数据放入本仓库（`data/TRD_Co.csv`），需要：

1. 从 CSV 构建 SQLite 数据库，清理无关字段
2. 支持按 4 级行业 + 省份 + 城市 + 所有权共 7 个维度筛选
3. 所有维度可选，可组合取交集
4. 将数据源从外部仓库迁移到本仓库

## 数据分析

TRD_Co.csv 共 5963 行（股票），分类维度统计：

| 维度 | 字段 | 唯一值数量 |
|------|------|-----------|
| 一级行业 | Indcd / Indnme | 6 |
| 二级行业 | Nindcd / Nindnme | 72 |
| 三级行业 | Nnindcd / Nnindnme | 82 |
| 四级行业（中信） | IndcdZX / IndnmeZX | 83 |
| 省份 | PROVINCE / PROVINCECODE | 34 |
| 城市 | CITY / CITYCODE | 434 |
| 所有权 | OWNERSHIPTYPE / OWNERSHIPTYPECODE | 8 |

一级行业示例：金融、公用事业、房地产、综合、工业、商业。

## SQLite 数据库设计

**路径**: `J:\DpointTrader\Dpoint_Trader\data\csmar_industry.sqlite`

**表结构**: 单表 `stocks`，扁平设计。

```sql
CREATE TABLE stocks (
    code           TEXT PRIMARY KEY,  -- 6位代码，如 "000001"
    name           TEXT NOT NULL,     -- 股票名称，如 "平安银行"
    -- 四级行业
    ind1_code      TEXT,  -- 一级行业代码，如 "1"
    ind1_name      TEXT,  -- 一级行业名称，如 "金融"
    ind2_code      TEXT,  -- 二级行业代码，如 "J"
    ind2_name      TEXT,  -- 二级行业名称，如 "银行业"
    ind3_code      TEXT,  -- 三级行业代码，如 "J66"
    ind3_name      TEXT,  -- 三级行业名称，如 "货币金融服务"
    ind4_code      TEXT,  -- 四级行业代码（中信），如 "C27"
    ind4_name      TEXT,  -- 四级行业名称（中信），如 "医药制造业"
    -- 地理
    province       TEXT,  -- 省份，如 "广东省"
    province_code  TEXT,  -- 省份代码，如 "440000"
    city           TEXT,  -- 城市，如 "深圳市"
    city_code      TEXT,  -- 城市代码，如 "440300"
    -- 所有权
    ownership      TEXT,  -- 所有权类型，如 "私营企业"
    ownership_code TEXT   -- 所有权代码，如 "P0306"
);

-- 索引：每个筛选维度
CREATE INDEX idx_ind1 ON stocks(ind1_code);
CREATE INDEX idx_ind2 ON stocks(ind2_code);
CREATE INDEX idx_ind3 ON stocks(ind3_code);
CREATE INDEX idx_ind4 ON stocks(ind4_code);
CREATE INDEX idx_province ON stocks(province);
CREATE INDEX idx_city ON stocks(city);
CREATE INDEX idx_ownership ON stocks(ownership);
```

**数据清理规则**:
- `Stkcd` 补零到 6 位（`1` → `000001`）
- 只保留上述字段，丢弃其余 20+ 列
- 过滤掉 `Stkcd` 为空的行

## CLI 接口设计

### 获取命令

```bash
dpoint fetch basket [筛选参数...] [选项...]
```

**筛选参数（全部可选，可组合取交集）**:

| 参数 | 说明 | 示例 |
|------|------|------|
| `--ind1` | 一级行业代码或名称 | `--ind1 金融` 或 `--ind1 1` |
| `--ind2` | 二级行业代码或名称 | `--ind2 银行业` 或 `--ind2 J` |
| `--ind3` | 三级行业代码或名称 | `--ind3 货币金融服务` 或 `--ind3 J66` |
| `--ind4` | 四级行业代码或名称（中信） | `--ind4 医药制造业` 或 `--ind4 C27` |
| `--province` | 省份名称 | `--province 广东省` |
| `--city` | 城市名称 | `--city 深圳市` |
| `--ownership` | 所有权类型 | `--ownership 私营企业` |
| `--industry` | 别名，等同于 `--ind4`（向后兼容） | `--industry C27` |

**其他选项**（保持不变）:
- `--start` / `--end` — 日期范围
- `--output` — 输出目录
- `--format` — 输出格式
- `--db` — SQLite 路径（默认 `data/csmar_industry.sqlite`）

**输出目录命名规则**:
- 单维度：`data/basket_{维度}_{值}`
- 多维度：`data/basket_{维度1}_{值1}_{维度2}_{值2}`
- 用户指定 `--output` 时覆盖默认命名

**示例**:
```bash
# 按四级行业（向后兼容）
dpoint fetch basket --industry C27

# 按省份
dpoint fetch basket --province 广东省

# 组合筛选
dpoint fetch basket --ind4 C27 --province 广东省 --ownership 私营企业
```

### 辅助查询命令

```bash
# 列出行业（可按级别筛选）
dpoint fetch list-industries [--level 1|2|3|4]

# 列出省份
dpoint fetch list-provinces

# 列出城市（可按省份筛选）
dpoint fetch list-cities [--province 广东省]

# 列出所有权类型
dpoint fetch list-ownership
```

## IndustryDB 重构

**文件**: `src/dpoint/data/fetch/industry.py`

**新接口**:

```python
class IndustryDB:
    def __init__(self, db_path: str | Path):
        """连接到 SQLite 数据库。"""

    def list_values(self, dimension: str) -> list[dict]:
        """列出某维度的所有可选值。
        dimension: "ind1" | "ind2" | "ind3" | "ind4" | "province" | "city" | "ownership"
        返回: [{"code": "1", "name": "金融", "count": 120}, ...]
        """

    def query_stocks(self, **filters) -> list[str]:
        """按维度筛选股票，返回代码列表。
        filters: ind1="金融", ind4="C27", province="广东省", ...
        所有过滤条件取交集。
        """

    def resolve_stock(self, code: str) -> dict:
        """查询单只股票的全部分类信息。"""
```

**SQL 构建**: `query_stocks` 动态构建 WHERE 子句，每个非空 filter 加一个 AND 条件。

**代码/名称匹配逻辑**: 当用户传入 `--ind1 金融` 时，先尝试按 `_code` 列精确匹配，若无结果则按 `_name` 列精确匹配。不做模糊匹配。

**路径变更**: `DEFAULT_DB_PATH` 从外部仓库改为 `data/csmar_industry.sqlite`。

## 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `data/TRD_Co.csv` | 已有 | 原始数据，不修改 |
| `data/csmar_industry.sqlite` | 新建 | 由构建脚本生成 |
| `scripts/build_industry_db.py` | 新建 | SQLite 构建脚本 |
| `src/dpoint/data/fetch/industry.py` | 重写 | 新 IndustryDB，新路径，新接口 |
| `src/dpoint/cli/main.py` | 修改 | 新增筛选参数 + list 命令 + 重构 run_fetch_basket |
| `tests/test_industry.py` | 新建 | 新接口单元测试 |

**不在范围内**:
- 不修改 `qmt_client.py`（底层获取逻辑不变）
- 不修改 `formatter.py`（格式转换不变）
- 不修改单股模式

## 实现顺序

1. 构建脚本 `scripts/build_industry_db.py` → 生成 `data/csmar_industry.sqlite`
2. 重写 `industry.py` → 新 IndustryDB 接口
3. 修改 `main.py` → CLI 参数 + run_fetch_basket + list 命令
4. 测试 `tests/test_industry.py`

## 股票代码格式说明

QMT 客户端接受两种格式：`000001.SZ` 和 `000001`。本仓库不涉及 ETF，全部是 A 股，用 6 位纯数字代码即可唯一定位。TRD_Co.csv 中的 `Stkcd` 字段不带后缀且不补零（如 `1`），构建 SQLite 时补零到 6 位即可。
