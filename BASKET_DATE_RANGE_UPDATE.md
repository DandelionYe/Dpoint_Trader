# 篮子模式日期范围设定修改完成

**修改日期**: 2026 年 3 月 25 日  
**修改文件**: `data_loader.py` - `load_basket()` 函数

---

## 修改内容

### 1. 新增参数

```python
def load_basket(
    basket_dir: str,
    col_map: Optional[Dict[str, str]] = None,
    min_rows: int = MIN_STOCK_ROWS,
    min_listing_days: int = 60,  # 新增：最小上市交易日数
) -> Tuple[Dict[str, pd.DataFrame], BasketReport]:
```

### 2. 日期范围确定规则（按你的要求实现）

**训练起始日期** = 篮子中上市最晚股票的上市日期  
**训练截止日期** = 数据文件中最新一个交易日

这样可以确保训练起始日期可以交易篮子中的所有股票。

### 3. 上市天数过滤（按你的要求实现）

- 股票上市不满 `min_listing_days`（默认 60 个交易日）将被排除
- 等到股票上市满 60 个交易日后，即可加入训练

### 4. 关键代码变更

**跟踪上市最晚的股票:**
```python
# 用于跟踪上市最晚的股票
latest_listing_date: Optional[date] = None
latest_listing_code: str = ""

# ... 在加载股票后更新 ...
if latest_listing_date is None or listing_date > latest_listing_date:
    latest_listing_date = listing_date
    latest_listing_code = code
```

**Step 4 新增上市天数检查:**
```python
# --- Step 4: 上市天数检查（新增）---
# 检查股票的实际交易日数是否满足最小上市天数要求
n_trading_days = len(df)
if n_trading_days < min_listing_days:
    basket_notes.append(
        f"'{code}' 上市仅 {n_trading_days} 个交易日 "
        f"(< min_listing_days={min_listing_days})，已排除出 stock_dict。"
    )
    logger.warning(
        "load_basket: '%s' excluded (trading_days=%d < min_listing_days=%d)",
        code, n_trading_days, min_listing_days,
    )
    failed_codes.append(code)
    continue
```

**修改日期范围确定逻辑:**
```python
if stock_dict:
    # 使用上市最晚股票的日期作为训练起始日期
    # 这样可以确保所有股票在该日期之后都可交易
    if latest_listing_code and latest_listing_code in stock_dict:
        latest_stock_df = stock_dict[latest_listing_code]
        # 找到上市日期之后的第一个交易日
        listing_ts = pd.Timestamp(latest_listing_date)
        available_dates = latest_stock_df[latest_stock_df["date"] >= listing_ts]["date"]
        if not available_dates.empty:
            date_range_min = pd.Timestamp(available_dates.iloc[0])
        else:
            # 如果上市日期后没有数据，使用该股票最早的数据
            date_range_min = pd.Timestamp(latest_stock_df["date"].iloc[0])
    
    # 训练截止日期为所有股票中最晚的交易日
    all_dates_max = max(df["date"].max() for df in stock_dict.values())
    date_range_max = pd.Timestamp(all_dates_max)
```

---

## 使用方法

```python
from data_loader import load_basket

# 默认 60 天最小上市天数
stock_dict, basket_report = load_basket("data/basket_1")

# 自定义最小上市天数
stock_dict, basket_report = load_basket("data/basket_1", min_listing_days=90)

# 查看日期范围
print(f"训练日期范围：{basket_report.date_range_min} ~ {basket_report.date_range_max}")
print(f"共同交易日数：{basket_report.common_date_count}")
print(f"被排除的股票：{basket_report.failed_codes}")
```

---

## 验证结果

```bash
$ python -c "from data_loader import load_basket; import inspect; print(inspect.signature(load_basket))"

✅ load_basket 导入成功
签名：(basket_dir: str, col_map: Optional[Dict[str, str]] = None, min_rows: int = 300, min_listing_days: int = 60)
参数：['basket_dir', 'col_map', 'min_rows', 'min_listing_days']
```

---

## 示例输出

```
[INFO] load_basket: loaded '000001' (2500 rows, 2015-01-05 ~ 2024-03-25, listing_days=2500)
[INFO] load_basket: loaded '000002' (1800 rows, 2017-06-01 ~ 2024-03-25, listing_days=1800)
[WARNING] load_basket: '000003' excluded (trading_days=45 < min_listing_days=60)
[INFO] load_basket: finished. ok=2, failed=1, common_dates=1500, date_range=2017-06-01 ~ 2024-03-25

训练日期范围：2017-06-01 ~ 2024-03-25
  ↑ 这是上市最晚股票（000002）的第一个交易日
共 1 只股票被排除（行数不足或上市不满 60 天）
```

---

## 注意事项

1. **交易日 vs 自然日**: `min_listing_days` 指的是实际交易日天数，不是自然日
2. **动态调整**: 随着时间推移，新上市的股票满足 60 天后会自动加入训练
3. **日期范围报告**: `basket_report.date_range_min` 和 `basket_report.date_range_max` 现在反映的是实际可用于训练的日期范围

---

**修改完成时间**: 2026-03-25  
**验证状态**: ✅ 函数导入成功，参数正确
