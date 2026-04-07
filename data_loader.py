# data_loader.py
"""
数据加载与切分模块 (Data Loading and Splitting Module)
========================================================

本模块提供 A 股量化交易研究框架的核心数据处理功能。

**Basket CSV 加载:**
    - parse_basket_filename: 解析 CSV 文件名，提取股票代码与上市日期
    - load_single_csv: 加载单只股票的 CSV 文件，统一列名并补全衍生字段
    - load_basket: 遍历 basket 目录，批量加载所有 CSV，返回 {code: df} 字典
    - build_panel_dataframe: 将多只股票 DataFrame 合并为面板结构
    - BasketReport: Basket 级别的加载质量报告

**数据切分:**
    - walkforward_splits: 标准 Walk-Forward 时序切分
    - walkforward_splits_with_embargo: 带 embargo gap 的 Walk-Forward 切分
    - nested_walkforward_splits: 嵌套 Walk-Forward 切分
    - final_holdout_split: 最终 holdout 集切分
    - recommend_n_folds: 根据数据量自适应推算合理折数

**使用示例 — Basket 加载流程:**
    >>> from data_loader import load_basket, build_panel_dataframe
    >>> stock_dict, basket_report = load_basket("data/basket_1")
    >>> basket_report.summary()           # 打印加载质量报告
    >>> panel_df = build_panel_dataframe(stock_dict)
    >>> # panel_df 含列: date, stock_code, open_qfq, ..., amount_proxy, listing_days

**切分策略说明:**
    Walk-Forward 是一种时序交叉验证方法，适用于金融时间序列数据。
    与传统的 K-Fold 不同，Walk-Forward 保持时间顺序，避免未来信息泄露。

    标准 Walk-Forward (n_folds=4, train_start_ratio=0.5)::

        折 1: train=[0%~50%]   val=[50%~62.5%]
        折 2: train=[0%~62%]   val=[62.5%~75%]
        折 3: train=[0%~75%]   val=[75%~87.5%]
        折 4: train=[0%~87%]   val=[87.5%~100%]
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from constants import (
    CSV_COL_MAP,
    CSV_REQUIRED_COLS,
    COL_AMOUNT_PROXY,
    COL_LISTING_DAYS,
    MIN_STOCK_ROWS,
    MIN_BASKET_SIZE,
)

logger = logging.getLogger(__name__)


# =========================================================
# 数据类：DataReport
# =========================================================

@dataclass
class DataReport:
    """数据质量报告的数据类（单只股票）。

    Attributes:
        rows_raw: 原始数据行数
        rows_after_dropna_core: 去除核心 OHLC 缺失值后的行数
        rows_after_filters: 应用所有过滤器后的最终行数
        duplicate_dates: 发现的重复日期数量
        bad_ohlc_rows: OHLC 不一致的行数
        sheet_used: 使用的 sheet 名称（CSV 加载时为文件名）
        notes: 处理过程中的注释和警告信息列表
    """
    rows_raw: int
    rows_after_dropna_core: int
    rows_after_filters: int
    duplicate_dates: int
    bad_ohlc_rows: int
    sheet_used: str
    notes: List[str]


# =========================================================
# Basket 级别质量报告
# =========================================================

@dataclass
class BasketReport:
    """Basket 批量加载的质量汇总报告。

    Attributes:
        basket_dir: 被加载的 basket 目录路径
        total_files: 目录内发现的 CSV 文件总数
        loaded_ok: 成功加载的股票数
        loaded_failed: 加载失败的股票数（解析错误或数据不足）
        failed_codes: 加载失败的股票代码列表
        stock_reports: 各股票的 DataReport，key 为股票代码
        date_range_min: 所有股票中最早的交易日
        date_range_max: 所有股票中最晚的交易日
        common_date_count: 所有股票共同拥有的交易日数量
        notes: basket 级别的警告和信息
    """
    basket_dir: str
    total_files: int
    loaded_ok: int
    loaded_failed: int
    failed_codes: List[str]
    stock_reports: Dict[str, DataReport]
    date_range_min: Optional[pd.Timestamp]
    date_range_max: Optional[pd.Timestamp]
    common_date_count: int
    notes: List[str]

    def summary(self) -> None:
        """打印 basket 加载摘要到 stdout。"""
        print("=" * 60)
        print(f"Basket Load Report: {self.basket_dir}")
        print("=" * 60)
        print(f"  CSV 文件总数   : {self.total_files}")
        print(f"  成功加载       : {self.loaded_ok}")
        print(f"  加载失败       : {self.loaded_failed}")
        if self.failed_codes:
            print(f"  失败股票       : {', '.join(self.failed_codes)}")
        if self.date_range_min and self.date_range_max:
            print(f"  日期范围       : {self.date_range_min.date()} ~ {self.date_range_max.date()}")
        print(f"  共同交易日数   : {self.common_date_count}")
        for note in self.notes:
            print(f"  [NOTE] {note}")
        print("-" * 60)
        for code, rep in self.stock_reports.items():
            status = "✓" if rep.rows_after_filters >= MIN_STOCK_ROWS else "⚠"
            print(
                f"  {status} {code}: {rep.rows_after_filters} 行"
                f"  ({rep.rows_raw} raw → {rep.rows_after_filters} clean)"
            )
            for n in rep.notes:
                print(f"      · {n}")
        print("=" * 60)


# =========================================================
# P-basket 新增：文件名解析
# =========================================================

# 匹配格式 "{code}_{YYYYMMDD}.csv"，代码允许任意非下划线字符
_BASKET_FILENAME_RE = re.compile(
    r"^(?P<code>[^_]+)_(?P<ymd>\d{8})\.csv$",
    re.IGNORECASE,
)


def parse_basket_filename(filename: str) -> Tuple[str, date]:
    """解析 basket CSV 文件名，提取股票代码与上市日期。

    文件名格式要求: ``{股票代码}_{上市日期YYYYMMDD}.csv``

    示例::

        >>> parse_basket_filename("300299_20120319.csv")
        ('300299', datetime.date(2012, 3, 19))

        >>> parse_basket_filename("002555_20110302.csv")
        ('002555', datetime.date(2011, 3, 2))

    Args:
        filename: 文件名（不含目录路径），如 "300299_20120319.csv"

    Returns:
        Tuple[str, date]: (股票代码字符串, 上市日期)

    Raises:
        ValueError: 文件名不符合约定格式
    """
    basename = os.path.basename(filename)
    m = _BASKET_FILENAME_RE.match(basename)
    if m is None:
        raise ValueError(
            f"文件名 '{basename}' 不符合 basket 命名约定 "
            f"'{{股票代码}}_{{YYYYMMDD}}.csv'。"
        )
    code = m.group("code")
    ymd = m.group("ymd")
    try:
        listing_date = datetime.strptime(ymd, "%Y%m%d").date()
    except ValueError as e:
        raise ValueError(f"文件名 '{basename}' 中日期 '{ymd}' 无法解析: {e}") from e
    return code, listing_date


# =========================================================
# P-basket 新增：单只 CSV 加载
# =========================================================

def load_single_csv(
    csv_path: str,
    listing_date: date,
    col_map: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, DataReport]:
    """加载单只股票的 CSV 文件，统一列名并补全衍生字段。

    本函数执行数据清洗（OHLC 一致性、
    负值过滤、重复日期去重等），并额外计算两个衍生字段：

    - ``amount_proxy``: 成交额代理值，公式为 ``(O+H+L+C)/4 × Volume``。
      由于 CSV 原始数据不含成交额，此字段用于替代 ``amount`` 参与
      ``backtester.check_execution_feasibility`` 中的流动性过滤逻辑。
      注意这只是估算，实际成交额会因盘中价格分布而有所偏差。

    - ``listing_days``: 上市至各交易日的自然日天数，由文件名中解析的
      ``listing_date`` 与各行 ``date`` 相减得到。用于 backtester 中
      "新股上市天数不足"的过滤逻辑（``DEFAULT_MIN_LISTING_DAYS = 60``）。

    以下字段在 CSV 中不可获得，统一设为下游安全的默认值：

    - ``turnover_rate``: 设为 ``NaN``，`feature_dpoint.py` 中 ``use_turnover``
      特征族检测到全 NaN 后会自动降级跳过，不抛异常。
    - ``is_st``: 设为 ``False``，backtester 的 ST 过滤不生效。
    - ``suspended``: 设为 ``False``，backtester 会用开盘价<=0 兜底推断。

    Args:
        csv_path: CSV 文件路径（含文件名），如 "data/basket_1/300299_20120319.csv"
        listing_date: 该股票的上市日期（由调用方通过文件名解析获得）
        col_map: 列名映射字典，None 时使用 constants.CSV_COL_MAP

    Returns:
        Tuple[pd.DataFrame, DataReport]:
            - 清洗并补全后的 DataFrame，含列：
              date, open_qfq, high_qfq, low_qfq, close_qfq, volume,
              amount_proxy, listing_days, turnover_rate, is_st, suspended
            - DataReport（sheet_used 字段填文件名）

    Raises:
        FileNotFoundError: CSV 文件不存在
        ValueError: CSV 缺少核心列，无法继续
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")

    if col_map is None:
        col_map = CSV_COL_MAP

    notes: List[str] = []
    filename = os.path.basename(csv_path)

    # ----------------------------------------------------------
    # 1. 读取 CSV（处理 UTF-8 BOM）
    # ----------------------------------------------------------
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    rows_raw = len(df)

    # 去除列名首尾空格（防止隐式空格导致映射失败）
    df.columns = [str(c).strip() for c in df.columns]

    # ----------------------------------------------------------
    # 2. 列名映射：原始列名 → 内部列名
    # ----------------------------------------------------------
    rename_map = {k: v for k, v in col_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # 检查核心列是否全部存在（映射后）
    missing_required = [c for c in CSV_REQUIRED_COLS if c not in df.columns]
    if missing_required:
        raise ValueError(
            f"[{filename}] 列名映射后仍缺少必需列: {missing_required}。"
            f"当前列: {list(df.columns)}。"
            f"请检查 CSV_COL_MAP 配置是否与数据源列名匹配。"
        )

    # 仅保留需要的列（忽略 CSV 中其他多余列）
    df = df[list(CSV_REQUIRED_COLS)].copy()

    # ----------------------------------------------------------
    # 3. 日期解析与排序
    # ----------------------------------------------------------
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bad_date = int(df["date"].isna().sum())
    if bad_date > 0:
        notes.append(f"Dropped rows with unparseable dates: {bad_date}")
    df = df.dropna(subset=["date"]).copy()
    df = df.sort_values("date").reset_index(drop=True)

    # ----------------------------------------------------------
    # 4. 重复日期去重
    # ----------------------------------------------------------
    duplicate_dates = int(df["date"].duplicated().sum())
    if duplicate_dates > 0:
        notes.append(
            f"Found {duplicate_dates} duplicate dates; keeping last occurrence."
        )
        df = df.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    # ----------------------------------------------------------
    # 5. 数值转换
    # ----------------------------------------------------------
    num_cols = [c for c in CSV_REQUIRED_COLS if c != "date"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ----------------------------------------------------------
    # 6. 核心 OHLC 缺失行删除
    # ----------------------------------------------------------
    core_cols = ["open_qfq", "high_qfq", "low_qfq", "close_qfq"]
    rows_before = len(df)
    df = df.dropna(subset=core_cols).copy()
    rows_after_dropna_core = len(df)
    dropped_core = rows_before - rows_after_dropna_core
    if dropped_core > 0:
        notes.append(f"Dropped {dropped_core} rows with NaN core OHLC.")

    # ----------------------------------------------------------
    # 7. 有效性过滤：非正价格
    # ----------------------------------------------------------
    bad_price_mask = (
        (df["open_qfq"] <= 0)
        | (df["high_qfq"] <= 0)
        | (df["low_qfq"] <= 0)
        | (df["close_qfq"] <= 0)
    )
    bad_price = int(bad_price_mask.sum())
    if bad_price > 0:
        notes.append(f"Dropped {bad_price} rows with non-positive prices.")
        df = df[~bad_price_mask].copy()

    # 负成交量
    bad_vol = int((df["volume"] < 0).sum())
    if bad_vol > 0:
        notes.append(f"Dropped {bad_vol} rows with negative volume.")
        df = df[df["volume"] >= 0].copy()

    # ----------------------------------------------------------
    # 8. OHLC 一致性检验：High >= max(O,C,L) 且 Low <= min(O,C,H)
    # ----------------------------------------------------------
    bad_ohlc_mask = ~(
        (df["high_qfq"] >= df[["open_qfq", "close_qfq", "low_qfq"]].max(axis=1))
        & (df["low_qfq"] <= df[["open_qfq", "close_qfq", "high_qfq"]].min(axis=1))
    )
    bad_ohlc_rows = int(bad_ohlc_mask.sum())
    if bad_ohlc_rows > 0:
        notes.append(f"Dropped {bad_ohlc_rows} rows with OHLC inconsistency.")
        df = df[~bad_ohlc_mask].copy()

    df = df.sort_values("date").reset_index(drop=True)
    rows_after_filters = len(df)

    # ----------------------------------------------------------
    # 9. 数据量告警
    # ----------------------------------------------------------
    if rows_after_filters < MIN_STOCK_ROWS:
        notes.append(
            f"Warning: only {rows_after_filters} rows after cleaning "
            f"(< {MIN_STOCK_ROWS}). ML features may be unstable."
        )

    # ----------------------------------------------------------
    # 10. 补全衍生字段
    # ----------------------------------------------------------

    # 10a. amount_proxy = 典型价 × 成交量（VWAP 代理，用于流动性过滤）
    typical_price = (
        df["open_qfq"] + df["high_qfq"] + df["low_qfq"] + df["close_qfq"]
    ) / 4.0
    df[COL_AMOUNT_PROXY] = (typical_price * df["volume"]).clip(lower=0.0)
    notes.append(
        f"Computed '{COL_AMOUNT_PROXY}' = (O+H+L+C)/4 × Volume "
        f"as proxy for trading amount (no raw amount in CSV)."
    )

    # 10b. listing_days = 各交易日与上市日期之间的自然日天数
    listing_dt = pd.Timestamp(listing_date)
    df[COL_LISTING_DAYS] = (df["date"] - listing_dt).dt.days.clip(lower=0)

    # 早于或等于上市日的行（理论上不应出现，但做防御处理）
    pre_listing = int((df[COL_LISTING_DAYS] == 0).sum())
    if pre_listing > 0:
        notes.append(
            f"Found {pre_listing} rows with listing_days=0 "
            f"(date ≤ listing_date {listing_date}); kept but flagged."
        )

    # 10c. 不可获得字段：设为下游安全的默认值
    #   turnover_rate: NaN → feature_dpoint.py 中 use_turnover 族自动降级
    #   is_st:         False → backtester ST 过滤不生效
    #   suspended:     False → backtester 用开盘价<=0 兜底推断
    df["turnover_rate"] = float("nan")
    df["is_st"] = False
    df["suspended"] = False

    report = DataReport(
        rows_raw=rows_raw,
        rows_after_dropna_core=rows_after_dropna_core,
        rows_after_filters=rows_after_filters,
        duplicate_dates=duplicate_dates,
        bad_ohlc_rows=bad_ohlc_rows,
        sheet_used=filename,   # 复用字段存放文件名
        notes=notes,
    )
    return df, report


# =========================================================
# P-basket 新增：Basket 目录批量加载
# =========================================================

def load_basket(
    basket_dir: str,
    col_map: Optional[Dict[str, str]] = None,
    min_rows: int = MIN_STOCK_ROWS,
    min_listing_days: int = 60,
) -> Tuple[Dict[str, pd.DataFrame], BasketReport]:
    """遍历 basket 目录，批量加载所有 CSV，返回股票字典与质量报告。

    目录内每个符合 ``{code}_{YYYYMMDD}.csv`` 命名的文件都会被加载。
    不符合命名规则的文件会被跳过并记录警告。

    加载成功的条件（以下任一不满足则归入 failed）：
        1. 文件名可被 ``parse_basket_filename`` 解析
        2. CSV 含有全部 ``CSV_REQUIRED_COLS`` 列（经 col_map 重命名后）
        3. 清洗后行数 >= ``min_rows``

    Args:
        basket_dir: basket 目录路径，如 ``"data/basket_1"``
        col_map: 列名映射，None 时使用 constants.CSV_COL_MAP
        min_rows: 清洗后的最小行数；低于此值的股票不纳入返回字典，
                  但会记录在 BasketReport.failed_codes 中

    Returns:
        Tuple[Dict[str, pd.DataFrame], BasketReport]:
            - stock_dict: {股票代码: 清洗后的 DataFrame}
            - basket_report: basket 级别的质量报告

    Raises:
        FileNotFoundError: basket_dir 目录不存在
    """
    if not os.path.isdir(basket_dir):
        raise FileNotFoundError(
            f"Basket 目录不存在：'{basket_dir}'。"
            f"请确认路径正确，或先在该路径下创建目录并放置 CSV 文件。"
        )

    # 收集目录内所有 .csv 文件（不递归子目录）
    all_files = sorted(
        f for f in os.listdir(basket_dir)
        if f.lower().endswith(".csv")
    )
    total_files = len(all_files)
    basket_notes: List[str] = []

    if total_files == 0:
        basket_notes.append(f"目录 '{basket_dir}' 内未找到任何 .csv 文件。")
        return {}, BasketReport(
            basket_dir=basket_dir,
            total_files=0,
            loaded_ok=0,
            loaded_failed=0,
            failed_codes=[],
            stock_reports={},
            date_range_min=None,
            date_range_max=None,
            common_date_count=0,
            notes=basket_notes,
        )

    stock_dict: Dict[str, pd.DataFrame] = {}
    stock_reports: Dict[str, DataReport] = {}
    failed_codes: List[str] = []
    
    # 用于跟踪上市最晚的股票
    latest_listing_date: Optional[date] = None
    latest_listing_code: str = ""

    for filename in all_files:
        csv_path = os.path.join(basket_dir, filename)

        # --- Step 1: 解析文件名 ---
        try:
            code, listing_date = parse_basket_filename(filename)
        except ValueError as e:
            basket_notes.append(f"跳过 '{filename}'：文件名不符合命名约定（{e}）")
            logger.warning("load_basket: skip file '%s': %s", filename, e)
            failed_codes.append(filename)  # 无法提取 code，用文件名代替
            continue

        # --- Step 2: 加载与清洗 ---
        try:
            df, report = load_single_csv(csv_path, listing_date, col_map=col_map)
        except (ValueError, FileNotFoundError) as e:
            basket_notes.append(f"跳过 '{code}'：加载失败（{e}）")
            logger.warning("load_basket: failed to load '%s': %s", filename, e)
            failed_codes.append(code)
            # 仍然记录一个占位 DataReport，方便 BasketReport.summary() 展示
            stock_reports[code] = DataReport(
                rows_raw=0, rows_after_dropna_core=0, rows_after_filters=0,
                duplicate_dates=0, bad_ohlc_rows=0, sheet_used=filename,
                notes=[f"加载失败：{e}"],
            )
            continue

        stock_reports[code] = report

        # --- Step 3: 行数检查 ---
        if report.rows_after_filters < min_rows:
            basket_notes.append(
                f"'{code}' 清洗后仅 {report.rows_after_filters} 行 "
                f"(< min_rows={min_rows})，已排除出 stock_dict。"
            )
            logger.warning(
                "load_basket: '%s' excluded (rows=%d < min_rows=%d)",
                code, report.rows_after_filters, min_rows,
            )
            failed_codes.append(code)
            continue

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

        # --- Step 5: 重复 code 处理 ---
        if code in stock_dict:
            basket_notes.append(
                f"发现重复股票代码 '{code}'，后加载的文件 '{filename}' 将覆盖前者。"
            )
            logger.warning("load_basket: duplicate code '%s', overwriting with '%s'", code, filename)

        stock_dict[code] = df
        
        # 更新上市最晚的股票信息
        if latest_listing_date is None or listing_date > latest_listing_date:
            latest_listing_date = listing_date
            latest_listing_code = code
        
        logger.info(
            "load_basket: loaded '%s' (%d rows, %s ~ %s, listing_days=%d)",
            code,
            report.rows_after_filters,
            df["date"].iloc[0].date(),
            df["date"].iloc[-1].date(),
            n_trading_days,
        )

    loaded_ok = len(stock_dict)
    loaded_failed = len(failed_codes)

    # --- Basket 级别统计 ---
    date_range_min: Optional[pd.Timestamp] = None
    date_range_max: Optional[pd.Timestamp] = None
    common_date_count: int = 0

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
        else:
            # 回退到所有股票的并集最小值
            all_dates_union = set.union(*[set(df["date"]) for df in stock_dict.values()])
            all_dates_sorted = sorted(all_dates_union)
            date_range_min = pd.Timestamp(all_dates_sorted[0]) if all_dates_sorted else None
        
        # 训练截止日期为所有股票中最晚的交易日
        all_dates_max = max(df["date"].max() for df in stock_dict.values())
        date_range_max = pd.Timestamp(all_dates_max)
        
        # 计算共同交易日数（所有股票都有的交易日）
        all_dates_intersection = set.intersection(*[set(df["date"]) for df in stock_dict.values()])
        common_date_count = len(all_dates_intersection)

        if common_date_count == 0:
            basket_notes.append(
                "警告：所有股票之间没有共同交易日！"
                "请检查各股票数据时间跨度是否存在重叠。"
            )
        elif common_date_count < 252:
            basket_notes.append(
                f"共同交易日仅 {common_date_count} 天（< 252），"
                f"面板模型的训练窗口可能不足 1 年。"
            )

    if loaded_ok < MIN_BASKET_SIZE:
        basket_notes.append(
            f"成功加载的股票数 {loaded_ok} 低于建议最小值 {MIN_BASKET_SIZE}，"
            f"横截面排名的统计稳定性可能不足。"
        )

    # 添加上市天数过滤的说明
    if loaded_failed > 0:
        basket_notes.append(
            f"共 {loaded_failed} 只股票被排除（行数不足或上市不满 {min_listing_days} 天）。"
        )

    basket_report = BasketReport(
        basket_dir=basket_dir,
        total_files=total_files,
        loaded_ok=loaded_ok,
        loaded_failed=loaded_failed,
        failed_codes=failed_codes,
        stock_reports=stock_reports,
        date_range_min=date_range_min,
        date_range_max=date_range_max,
        common_date_count=common_date_count,
        notes=basket_notes,
    )

    logger.info(
        "load_basket: finished. ok=%d, failed=%d, common_dates=%d, date_range=%s ~ %s",
        loaded_ok, loaded_failed, common_date_count,
        date_range_min.date() if date_range_min else "N/A",
        date_range_max.date() if date_range_max else "N/A",
    )

    return stock_dict, basket_report
    return stock_dict, basket_report


# =========================================================
# P-basket 新增：合并为面板 DataFrame
# =========================================================

def build_panel_dataframe(
    stock_dict: Dict[str, pd.DataFrame],
    sort: bool = True,
) -> pd.DataFrame:
    """将多只股票 DataFrame 合并为面板结构（长格式）。

    合并后的 DataFrame 为"长格式"（long format），每行代表
    一只股票在一个交易日的记录。这是面板数据机器学习与组合
    回测的标准输入格式。

    结构示意::

        date        stock_code  open_qfq  high_qfq  ...  amount_proxy  listing_days
        2023-01-03  300299      6.10      6.25      ...  1234567.0     3942
        2023-01-03  002555      12.50     12.80     ...  8901234.0     4328
        2023-01-04  300299      6.20      6.30      ...  2345678.0     3943
        ...

    Notes:
        - 不同股票的日期范围可以不同；合并后不对齐/不填充，
          保留各股的原始日期（对齐由下游特征工程或训练切分负责）。
        - ``stock_code`` 列被插入为第二列（紧跟 ``date``），便于
          分组操作（如 ``.groupby("stock_code")``）。
        - 合并结果按 (date, stock_code) 排序（sort=True 时）。

    Args:
        stock_dict: {股票代码: 清洗后的 DataFrame}，
                    通常直接来自 load_basket 的返回值
        sort: 是否按 (date, stock_code) 排序，默认 True

    Returns:
        pd.DataFrame: 面板格式的长表，含 stock_code 列

    Raises:
        ValueError: stock_dict 为空
    """
    if not stock_dict:
        raise ValueError(
            "stock_dict 为空，无法构建面板 DataFrame。"
            "请先调用 load_basket() 加载至少一只股票。"
        )

    frames: List[pd.DataFrame] = []
    for code, df in stock_dict.items():
        df_copy = df.copy()
        # 插入 stock_code 列（放在 date 之后的第二列位置）
        df_copy.insert(1, "stock_code", code)
        frames.append(df_copy)

    panel = pd.concat(frames, axis=0, ignore_index=True)

    if sort:
        panel = panel.sort_values(["date", "stock_code"]).reset_index(drop=True)

    logger.info(
        "build_panel_dataframe: %d stocks, %d total rows, date range %s ~ %s",
        len(stock_dict),
        len(panel),
        panel["date"].min().date(),
        panel["date"].max().date(),
    )

    return panel


# =========================================================
# 原有：数据切分 (Data Splitting) — 签名与行为完全不变
# =========================================================

def walkforward_splits(
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 4,
    train_start_ratio: float = 0.5,
    min_rows: int = 50,
) -> List[Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]]:
    """
    生成 walk-forward 时序切分。

    Walk-forward 是一种时序交叉验证方法，适用于金融时间序列数据。
    验证集不重叠，训练集累积扩展（expanding window）。

    参数说明：
        n_folds          : 验证折数，默认 4。
        train_start_ratio: 第一折训练集占全部数据的比例，默认 0.5。
        min_rows         : 训练集或验证集的最小行数，不足时跳过该折并打印警告。
                           默认 50（原 80），降低以支持较少数据量的场景。

    切分示意（n_folds=4, train_start_ratio=0.5）::

        折 1: train=[0%~50%]   val=[50%~62.5%]
        折 2: train=[0%~62%]   val=[62.5%~75%]
        折 3: train=[0%~75%]   val=[75%~87.5%]
        折 4: train=[0%~87%]   val=[87.5%~100%]
        （共 n_folds = 4 个验证折，首段 0~50% 数据仅用作初始训练集）

    注意：验证集不重叠，训练集累积扩展。

    Args:
        X: 特征 DataFrame
        y: 标签 Series
        n_folds: 验证折数
        train_start_ratio: 初始训练集比例
        min_rows: 最小行数约束

    Returns:
        List[Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]]:
            包含 (训练集，验证集) 元组的列表
    """
    n = len(X)
    cuts = [
        train_start_ratio + (1.0 - train_start_ratio) * i / n_folds
        for i in range(n_folds + 1)
    ]

    splits = []
    for k in range(len(cuts) - 1):
        train_end = int(n * cuts[k])
        val_end = int(n * cuts[k + 1])
        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]

        if len(X_train) < min_rows or len(X_val) < min_rows:
            logger.warning(
                "walkforward_splits: fold %d skipped (train=%d, val=%d, min_rows=%d). "
                "Consider reducing n_folds or min_rows.",
                k + 1, len(X_train), len(X_val), min_rows
            )
            continue
        splits.append(((X_train, y_train), (X_val, y_val)))

    if not splits:
        logger.warning(
            "walkforward_splits: ALL %d folds skipped. Total rows=%d, train_start_ratio=%.2f, min_rows=%d.",
            n_folds, n, train_start_ratio, min_rows
        )
    return splits


def final_holdout_split(
    df: pd.DataFrame,
    holdout_ratio: float = 0.15,
    min_holdout_rows: int = 60,
) -> Tuple[
    pd.DataFrame,  # search_df
    pd.DataFrame,  # holdout_df
]:
    """
    P0: Final holdout split - 从数据末尾切出 holdout 集，确保搜索流程完全不接触。

    三阶段验证流程：
        1. Search OOS: walk-forward splits 在 search 数据上评估
        2. Selection OOS: top-K 候选在 search 数据上重新验证
        3. Final Holdout OOS: 最优配置在 holdout 集上做最终评估

    参数说明：
        holdout_ratio    : holdout 集占总数据的比例，默认 15%
        min_holdout_rows : holdout 集最小行数，不足时抛出异常

    返回：
        (search_df, holdout_df) — 原始 DataFrame 切分结果
        （调用处需自行调用 build_features_and_labels 生成 X, y）

    Args:
        df: 原始 DataFrame
        holdout_ratio: holdout 集比例
        min_holdout_rows: holdout 集最小行数

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (search_df, holdout_df)

    Raises:
        ValueError: 当 holdout_size < min_holdout_rows 时
    """
    n = len(df)
    holdout_size = int(n * holdout_ratio)

    if holdout_size < min_holdout_rows:
        raise ValueError(
            f"holdout_size={holdout_size} < min_holdout_rows={min_holdout_rows}. "
            f"Increase holdout_ratio or use more data."
        )

    split_idx = n - holdout_size
    search_df = df.iloc[:split_idx].copy()
    holdout_df = df.iloc[split_idx:].copy()

    logger.info(
        "P0 Final Holdout Split: search=%d rows, holdout=%d rows (%.1f%%)",
        len(search_df), len(holdout_df), holdout_ratio * 100
    )

    return search_df, holdout_df


def recommend_n_folds(
    n_samples: int,
    train_start_ratio: float = 0.5,
    target_trades_per_fold: int = 4,
    assumed_trade_freq: float = 1.0 / 15.0,
    min_rows: int = 50,
    min_folds: int = 2,
    max_folds: int = 8,
) -> int:
    """
    P3-20：根据数据量自适应推算合理的 walk-forward 折数。

    推算原则：
        在满足以下三个约束的前提下，选取尽可能大的折数：
            ① 每折验证期行数 ≥ min_rows（确保回测有足够交易日）
            ② 每折期望交易次数 ≈ target_trades_per_fold（与 penalty 对齐）
            ③ 折数在 [min_folds, max_folds] 范围内

    Args:
        n_samples: 总样本数
        train_start_ratio: 初始训练集比例
        target_trades_per_fold: 每折目标交易次数
        assumed_trade_freq: 假设的交易频率
        min_rows: 每折最小行数
        min_folds: 最少折数
        max_folds: 最多折数

    Returns:
        int: 推荐的折数
    """
    val_pool = n_samples * (1.0 - train_start_ratio)

    best_n = min_folds
    for n in range(max_folds, min_folds - 1, -1):
        val_rows_per_fold = val_pool / n
        if val_rows_per_fold < min_rows:
            continue
        expected_trades = val_rows_per_fold * assumed_trade_freq
        if expected_trades < target_trades_per_fold:
            continue
        best_n = n
        break

    return max(min_folds, min(max_folds, best_n))


def nested_walkforward_splits(
    X: pd.DataFrame,
    y: pd.Series,
    n_outer_folds: int = 3,
    n_inner_folds: int = 2,
    train_start_ratio: float = 0.5,
    min_rows: int = 60,
    embargo_days: int = 5,
) -> List[Tuple[
    Tuple[pd.DataFrame, pd.Series],
    Tuple[pd.DataFrame, pd.Series],
    List[Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]],
]]:
    """
    P2: 嵌套 Walk-Forward 切分。

    在标准 walk-forward 基础上增加内层切分，避免"用验证集选择模型，
    再用同一验证集评估模型"的前向偏差。

    Args:
        X: 特征 DataFrame
        y: 标签 Series
        n_outer_folds: 外层折数
        n_inner_folds: 内层折数
        train_start_ratio: 外层初始训练集比例
        min_rows: 最小行数约束
        embargo_days: embargo 天数

    Returns:
        List[Tuple[outer_train, outer_val, inner_splits]]
    """
    n = len(X)
    cuts = [
        train_start_ratio + (1.0 - train_start_ratio) * i / n_outer_folds
        for i in range(n_outer_folds + 1)
    ]

    splits = []
    for k in range(len(cuts) - 1):
        outer_train_end = int(n * cuts[k])
        outer_val_end = int(n * cuts[k + 1])

        outer_val_start = outer_train_end + embargo_days
        if outer_val_start >= outer_val_end:
            logger.warning("nested_walkforward: fold %d skipped due to embargo_days=%d", k + 1, embargo_days)
            continue

        X_outer_train = X.iloc[:outer_train_end]
        y_outer_train = y.iloc[:outer_train_end]
        X_outer_val = X.iloc[outer_val_start:outer_val_end]
        y_outer_val = y.iloc[outer_val_start:outer_val_end]

        if len(X_outer_train) < min_rows or len(X_outer_val) < min_rows:
            logger.warning("nested_walkforward: fold %d skipped (train=%d, val=%d)",
                          k + 1, len(X_outer_train), len(X_outer_val))
            continue

        inner_splits = walkforward_splits(
            X_outer_train,
            y_outer_train,
            n_folds=n_inner_folds,
            train_start_ratio=train_start_ratio,
            min_rows=min_rows,
        )

        if not inner_splits:
            logger.warning("nested_walkforward: fold %d skipped (no valid inner splits)", k + 1)
            continue

        splits.append((
            (X_outer_train, y_outer_train),
            (X_outer_val, y_outer_val),
            inner_splits
        ))

    if not splits:
        logger.warning("nested_walkforward: ALL %d folds skipped.", n_outer_folds)
    return splits


def walkforward_splits_with_embargo(
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 4,
    train_start_ratio: float = 0.5,
    min_rows: int = 60,
    embargo_days: int = 5,
) -> List[Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]]:
    """
    P2: 带 embargo 的 Walk-Forward 切分。

    在训练集和验证集之间留出 gap，防止滚动窗口特征导致的信息泄露。

    Args:
        X: 特征 DataFrame
        y: 标签 Series
        n_folds: 验证折数
        train_start_ratio: 初始训练集比例
        min_rows: 最小行数约束
        embargo_days: embargo 天数

    Returns:
        List[Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]]
    """
    n = len(X)
    cuts = [
        train_start_ratio + (1.0 - train_start_ratio) * i / n_folds
        for i in range(n_folds + 1)
    ]

    splits = []
    for k in range(len(cuts) - 1):
        train_end = int(n * cuts[k])
        val_end = int(n * cuts[k + 1])

        val_start = train_end + embargo_days
        if val_start >= val_end:
            logger.warning("walkforward_splits_with_embargo: fold %d skipped (embargo=%d)", k + 1, embargo_days)
            continue

        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_val, y_val = X.iloc[val_start:val_end], y.iloc[val_start:val_end]

        if len(X_train) < min_rows or len(X_val) < min_rows:
            logger.warning("walkforward_splits_with_embargo: fold %d skipped (train=%d, val=%d)",
                          k + 1, len(X_train), len(X_val))
            continue

        splits.append(((X_train, y_train), (X_val, y_val)))

    if not splits:
        logger.warning("walkforward_splits_with_embargo: ALL %d folds skipped.", n_folds)
    return splits


# =========================================================
# 公开 API 导出 (Public API Exports)
# =========================================================
__all__ = [
    # 数据类
    "DataReport",
    "BasketReport",
    # Basket CSV 加载
    "parse_basket_filename",
    "load_single_csv",
    "load_basket",
    "build_panel_dataframe",
    # 数据切分
    "walkforward_splits",
    "walkforward_splits_with_embargo",
    "nested_walkforward_splits",
    "final_holdout_split",
    "recommend_n_folds",
]
