# feature_dpoint.py
"""
特征工程模块 - 篮子模式（多股票组合）专用

本模块专为 Basket（多股票组合）模式设计，支持：

**面板特征构建:**
    build_panel_features(stock_dict, config) → (X_panel, y_panel, meta_dict)

    参数说明:
        stock_dict : {股票代码：清洗后的 DataFrame}，来自 load_basket()
        config     : 特征配置字典
        X_panel    : 特征矩阵，含 ``date`` 和 ``stock_code`` 列
        y_panel    : 标签 Series，与 X_panel 行对齐
        meta_dict  : {股票代码：FeatureMeta}，记录各股的实际特征配置

    面板训练时，stock_code 列可以：
        ① 直接丢弃（pool 训练，把所有股票的样本混合）
        ② 编码为整数/one-hot（让模型感知标的差异）
        ③ 用于 groupby 分组（per-stock 评估、时序切分）

**横截面特征增强（可选）:**
    add_crosssection_features(X_panel, windows) → X_panel_enhanced

    在面板特征的基础上，对每个交易日内的多只股票计算各特征的
    百分位排名，得到"横截面排名特征"。这类特征能让模型感知
    某只股票在当日同类股票中的相对强弱，而非只看绝对值。

    注意：此函数要求 X_panel 含 ``date`` 和 ``stock_code`` 列，
    且每个日期至少有 2 只股票的记录。日期内样本过少（< 2）时跳过。

**技术指标特征族:**
    支持经典技术指标：RSI / MACD / 布林带宽 / OBV

**数据容错:**
    - ``amount`` 列缺失或全零时，自动降级使用 ``amount_proxy`` 列
    - ``turnover_rate`` 列全为 NaN 时，自动将 ``use_turnover`` 降级为 False
    - 降级行为记录在 FeatureMeta.notes 中，便于问题排查
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =========================================================
# 数据类
# =========================================================

@dataclass
class FeatureMeta:
    feature_names: List[str]
    params: Dict[str, object]
    dpoint_explainer: str
    # 记录容错降级行为，便于问题排查
    notes: List[str] = field(default_factory=list)


# =========================================================
# 私有工具函数
# =========================================================

def _safe_log1p(x: pd.Series) -> pd.Series:
    """对序列做 log1p 变换，先 clip 负值为 0，避免对数域报错。"""
    return np.log1p(np.clip(x.astype(float), 0.0, None))


def _rolling_mad(x: pd.Series, window: int) -> pd.Series:
    """滚动中位数绝对偏差（MAD），比标准差更鲁棒的波动率代理。"""
    med = x.rolling(window, min_periods=window).median()
    mad = (x - med).abs().rolling(window, min_periods=window).median()
    return mad


def _rolling_zscore(x: pd.Series, window: int) -> pd.Series:
    """滚动 Z-score 标准化；标准差为 0 时返回 NaN 避免除零。"""
    mu = x.rolling(window, min_periods=window).mean()
    sd = x.rolling(window, min_periods=window).std()
    return (x - mu) / sd.replace(0, np.nan)


# =========================================================
# P3-19：技术指标计算函数
# =========================================================

def _calc_rsi(close: pd.Series, window: int) -> pd.Series:
    """
    计算 RSI（相对强弱指数），归一化到 [0, 1]。

    边界处理（符合 TradingView / Wilder 原始定义）：
        - 全涨（avg_loss == 0）→ RSI = 1.0（极度超买）
        - 全跌（avg_gain == 0）→ RSI = 0.0（极度超卖）
        - 完全横盘（两者均 0） → RSI = 0.5（中性）
    无前向偏差：t 日 RSI 只使用 close[0..t]。
    """
    delta = close.diff(1)
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(com=window - 1, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window, adjust=False).mean()

    with np.errstate(invalid="ignore", divide="ignore"):
        rs = avg_gain / avg_loss.where(avg_loss != 0.0, other=np.nan)

    rsi_raw = 100.0 - 100.0 / (1.0 + rs)

    rsi_filled = np.where(
        avg_loss == 0.0,
        np.where(avg_gain == 0.0, 50.0, 100.0),
        np.where(avg_gain == 0.0, 0.0, rsi_raw),
    )

    return pd.Series(rsi_filled / 100.0, index=close.index)


def _calc_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series]:
    """
    计算 MACD 线和 MACD 柱状图（rolling z-score 归一化）。
    无前向偏差：EMA 只使用 t 日及以前的收盘价。
    """
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    macd_hist = macd_line - signal_line

    norm_window = slow + signal
    macd_line_z = _rolling_zscore(macd_line, norm_window)
    macd_hist_z = _rolling_zscore(macd_hist, norm_window)

    return macd_line_z, macd_hist_z


def _calc_bband_width(close: pd.Series, window: int, n_std: float = 2.0) -> pd.Series:
    """
    计算布林带宽（Bollinger Band Width = 2 × n_std × std / sma）。
    无前向偏差：t 日宽度只使用 close[t-window+1..t]。
    """
    sma = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std()
    bwidth = 2.0 * n_std * std / sma.replace(0, np.nan)
    return bwidth


def _calc_obv(
    close: pd.Series,
    volume: pd.Series,
    zscore_window: int = 20,
) -> pd.Series:
    """
    计算 OBV 并做 rolling z-score 归一化。

    Args:
        zscore_window: 归一化窗口，建议与 ta_windows 中的中位值对齐。
    """
    direction = np.sign(close.diff(1))
    direction.iloc[0] = 0.0
    obv_raw = (direction * volume).cumsum()
    obv_z = _rolling_zscore(obv_raw, window=zscore_window)
    return obv_z


# =========================================================
# P-basket 新增：列可用性探针
# =========================================================

def _detect_available_columns(
    df: pd.DataFrame,
    config: Dict[str, object],
) -> Tuple[
    Optional[pd.Series],   # amount_series  — 可用的成交额/代理序列，无则 None
    bool,                  # amount_is_proxy — True 表示使用的是 amount_proxy
    bool,                  # use_turnover_eff — 实际生效的 use_turnover 开关
    List[str],             # notes — 降级行为记录
]:
    """
    统一探测 df 中与特征构建相关的列可用性，输出实际生效的配置。

    探测逻辑：

    **成交额（amount）：**
        优先使用 ``amount`` 列（Excel 来源）；
        若 ``amount`` 不存在或全为 0/NaN，则降级到 ``amount_proxy``
        （CSV 来源，由 load_single_csv 计算）；
        两者均不可用时返回 None，调用方应禁用依赖成交额的子特征。

    **换手率（turnover_rate）：**
        若 ``turnover_rate`` 列全为 NaN（CSV 来源默认值），
        则将 use_turnover 强制降级为 False。
        只要有一个非 NaN 值，就保留原始配置。

    Args:
        df: 单只股票的清洗后 DataFrame
        config: 特征配置字典（用于读取 use_turnover 原始意图）

    Returns:
        (amount_series, amount_is_proxy, use_turnover_eff, notes)
    """
    notes: List[str] = []

    # ── 成交额探测 ──────────────────────────────────────────
    amount_series: Optional[pd.Series] = None
    amount_is_proxy = False

    if "amount" in df.columns:
        col = df["amount"].astype(float)
        if col.notna().any() and (col > 0).any():
            amount_series = col
        else:
            notes.append(
                "'amount' 列存在但全为 0/NaN，尝试降级到 'amount_proxy'。"
            )

    if amount_series is None and "amount_proxy" in df.columns:
        col = df["amount_proxy"].astype(float)
        if col.notna().any() and (col > 0).any():
            amount_series = col
            amount_is_proxy = True
            notes.append(
                "使用 'amount_proxy'（= (O+H+L+C)/4 × Volume）替代缺失的 'amount'；"
                "流动性过滤精度略有下降，属正常现象。"
            )
        else:
            notes.append(
                "'amount' 和 'amount_proxy' 均不可用；"
                "依赖成交额的成交量子特征（amount_ma/amount_z）将被跳过。"
            )

    # ── 换手率探测 ──────────────────────────────────────────
    use_turnover_cfg = bool(config.get("use_turnover", True))
    use_turnover_eff = use_turnover_cfg

    if use_turnover_cfg:
        if "turnover_rate" not in df.columns:
            use_turnover_eff = False
            notes.append(
                "'turnover_rate' 列不存在，use_turnover 自动降级为 False。"
            )
        else:
            col = df["turnover_rate"].astype(float)
            if col.isna().all():
                use_turnover_eff = False
                notes.append(
                    "'turnover_rate' 列全为 NaN（CSV 数据源不含此字段），"
                    "use_turnover 自动降级为 False。"
                )

    return amount_series, amount_is_proxy, use_turnover_eff, notes


# =========================================================
# 主特征构建函数（单股）
# =========================================================

def build_features_and_labels(
    df: pd.DataFrame,
    config: Dict[str, object],
) -> Tuple[pd.DataFrame, pd.Series, FeatureMeta]:
    """
    为单只股票构建特征矩阵和标签序列。

    Dpoint_t = P(close_{t+1} > close_t | X_t)
    所有特征仅使用 t 日及以前的数据（无前向偏差）。

    **容错处理：**

    - ``amount`` 缺失/全零时，自动降级到 ``amount_proxy``；
      两者均无效时，``use_volume`` 中依赖成交额的子特征（amount_ma_*
      / amount_z_*）被跳过，其余成交量特征正常计算。
    - ``turnover_rate`` 全为 NaN 时，``use_turnover`` 自动降级为 False，
      不抛异常。
    - 以上降级行为均记录在返回的 ``FeatureMeta.notes`` 中。

    **技术指标特征族：**
        ``use_ta_indicators``（RSI / MACD / 布林带宽 / OBV）特征族。
        通过 ``config["use_ta_indicators"] = True`` 启用，默认 False。

    Args:
        df: 含 date / open_qfq / high_qfq / low_qfq / close_qfq / volume
            等列的单只股票日线 DataFrame（来自 load_basket 加载的单只股票数据）
        config: 特征配置字典，支持以下键：
            windows        (List[int]) — 滚动窗口列表，默认 [3,5,10,20]
            use_momentum   (bool)      — 是否构建动量特征，默认 True
            use_volatility (bool)      — 是否构建波动率特征，默认 True
            use_volume     (bool)      — 是否构建成交量特征，默认 True
            use_candle     (bool)      — 是否构建 K 线特征，默认 True
            use_turnover   (bool)      — 是否构建换手率特征，默认 True
            use_ta_indicators (bool)   — 是否构建 TA 指标特征，默认 False
            vol_metric     (str)       — 波动率度量：'std' 或 'mad'，默认 'std'
            liq_transform  (str)       — 流动性变换：'ratio' 或 'zscore'，默认 'ratio'
            ta_windows     (List[int]) — TA 指标窗口，默认 [6,14,20]

    Returns:
        Tuple[pd.DataFrame, pd.Series, FeatureMeta]:
            - X: 特征矩阵，index 为 pd.Timestamp 日期
            - y: 标签（0/1），与 X 对齐
            - meta: 特征元信息，含实际生效的参数和降级 notes
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    windows: List[int] = list(config.get("windows", [3, 5, 10, 20]))

    # 原始特征族开关（初始值，可能被容错逻辑覆盖）
    use_momentum:       bool = bool(config.get("use_momentum",       True))
    use_volatility:     bool = bool(config.get("use_volatility",     True))
    use_volume:         bool = bool(config.get("use_volume",         True))
    use_candle:         bool = bool(config.get("use_candle",         True))
    use_ta_indicators:  bool = bool(config.get("use_ta_indicators",  False))

    vol_metric:    str = str(config.get("vol_metric",    "std")).lower()
    liq_transform: str = str(config.get("liq_transform", "ratio")).lower()
    ta_windows: List[int] = list(config.get("ta_windows", [6, 14, 20]))

    # ── P-basket：列可用性探针 ─────────────────────────────
    amount_series, amount_is_proxy, use_turnover, detect_notes = (
        _detect_available_columns(df, config)
    )
    if detect_notes:
        for note in detect_notes:
            logger.debug("feature_dpoint: %s", note)

    # ── 基础价格序列 ──────────────────────────────────────
    close    = df["close_qfq"].astype(float)
    open_    = df["open_qfq"].astype(float)
    high     = df["high_qfq"].astype(float)
    low      = df["low_qfq"].astype(float)
    volume   = df["volume"].astype(float)
    turnover = df["turnover_rate"].astype(float) if "turnover_rate" in df.columns else None

    feats: Dict[str, pd.Series] = {}

    # ── 基础收益率（所有配置均计算）────────────────────────
    ret1 = close.pct_change(1)
    feats["ret_1"] = ret1

    # ── 动量特征 ─────────────────────────────────────────
    if use_momentum:
        for k in windows:
            feats[f"ret_{k}"] = close.pct_change(k)
            ma = close.rolling(k, min_periods=k).mean()
            feats[f"ma_{k}_ratio"] = close / ma - 1.0

    # ── 波动率特征 ───────────────────────────────────────
    if use_volatility:
        feats["hl_range"] = (high - low) / close.replace(0, np.nan)

        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1
        ).max(axis=1)
        feats["true_range_norm"] = tr / close.replace(0, np.nan)

        for k in windows:
            if vol_metric == "mad":
                feats[f"vol_mad_{k}"] = _rolling_mad(ret1, k)
            else:
                feats[f"vol_std_{k}"] = ret1.rolling(k, min_periods=k).std()

    # ── 成交量/成交额特征 ─────────────────────────────────
    if use_volume:
        feats["log_volume"] = _safe_log1p(volume)

        # 成交额：优先使用探针返回的可用序列
        if amount_series is not None:
            feats["log_amount"] = _safe_log1p(amount_series)

        for k in windows:
            if liq_transform == "zscore":
                feats[f"volume_z_{k}"] = _rolling_zscore(volume, k)
                if amount_series is not None:
                    feats[f"amount_z_{k}"] = _rolling_zscore(amount_series, k)
            else:  # ratio
                vma = volume.rolling(k, min_periods=k).mean()
                feats[f"volume_ma_{k}_ratio"] = volume / vma.replace(0, np.nan)
                if amount_series is not None:
                    ama = amount_series.rolling(k, min_periods=k).mean()
                    feats[f"amount_ma_{k}_ratio"] = amount_series / ama.replace(0, np.nan)

    # ── 换手率特征（仅在 use_turnover_eff=True 时计算）─────
    if use_turnover and turnover is not None:
        feats["turnover"] = turnover
        for k in windows:
            if liq_transform == "zscore":
                feats[f"turnover_z_{k}"] = _rolling_zscore(turnover, k)
            else:
                feats[f"turnover_ma_{k}"]  = turnover.rolling(k, min_periods=k).mean()
                feats[f"turnover_std_{k}"] = turnover.rolling(k, min_periods=k).std()

    # ── K 线形态特征 ─────────────────────────────────────
    if use_candle:
        feats["body"]         = (close - open_) / open_.replace(0, np.nan)
        feats["upper_shadow"] = (high - np.maximum(open_, close)) / close.replace(0, np.nan)
        feats["lower_shadow"] = (np.minimum(open_, close) - low)  / close.replace(0, np.nan)

    # ── P3-19：技术指标特征族 ─────────────────────────────
    if use_ta_indicators:
        for w in ta_windows:
            feats[f"rsi_{w}"] = _calc_rsi(close, window=w)

        macd_line_z, macd_hist_z = _calc_macd(close, fast=12, slow=26, signal=9)
        feats["macd_line_z"] = macd_line_z
        feats["macd_hist_z"] = macd_hist_z

        for w in ta_windows:
            feats[f"bband_width_{w}"] = _calc_bband_width(close, window=w)

        obv_window = int(np.median(ta_windows))
        feats["obv_z"] = _calc_obv(close, volume, zscore_window=obv_window)

    # ── 组装特征矩阵 ─────────────────────────────────────
    X = pd.DataFrame(feats)

    # ── 构建标签：t+1 收盘是否高于 t 收盘 ────────────────
    y_diff = close.shift(-1) - close
    y = (y_diff > 0).astype(int)

    # ── 过滤：X 全部非 NaN 且 y_diff 有效（排除末行）──────
    valid = X.notna().all(axis=1) & y_diff.notna()
    X = X.loc[valid].copy()
    y = y.loc[valid].copy()

    X.index = df.loc[X.index, "date"].values
    y.index = X.index

    # ── 组装 FeatureMeta ──────────────────────────────────
    obv_window_val = int(np.median(ta_windows)) if use_ta_indicators else None

    meta = FeatureMeta(
        feature_names=list(X.columns),
        params={
            "windows":            windows,
            "use_momentum":       use_momentum,
            "use_volatility":     use_volatility,
            "use_volume":         use_volume,
            "use_candle":         use_candle,
            "use_turnover":       use_turnover,        # 实际生效值（可能已降级）
            "use_turnover_cfg":   bool(config.get("use_turnover", True)),  # 原始配置值
            "vol_metric":         vol_metric,
            "liq_transform":      liq_transform,
            "use_ta_indicators":  use_ta_indicators,
            "ta_windows":         ta_windows,
            "obv_zscore_window":  obv_window_val,
            "amount_is_proxy":    amount_is_proxy,     # 记录是否使用了代理成交额
        },
        dpoint_explainer=(
            "Dpoint_t = P(close_{t+1} > close_t | X_t). "
            "X_t is built from OHLCV/amount/turnover data up to t only (no future leakage). "
            "P3-19: optional TA indicators (RSI, MACD, BB-width, OBV) available via use_ta_indicators=True. "
            "P-basket: amount_proxy and turnover_rate fallback supported."
        ),
        notes=detect_notes,
    )
    return X, y, meta


# =========================================================
# P-basket 新增：面板特征构建入口
# =========================================================

def build_panel_features(
    stock_dict: Dict[str, pd.DataFrame],
    config: Dict[str, object],
    skip_on_error: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, FeatureMeta]]:
    """
    对 basket 内所有股票批量构建特征，拼接为面板格式。

    本函数对 stock_dict 中每只股票调用 ``build_features_and_labels``，
    然后将结果垂直拼接为"长格式"面板，并插入 ``stock_code`` 列。

    **面板格式说明：**

        返回的 X_panel 为普通 DataFrame（非 MultiIndex），含列::

            stock_code  ret_1  ret_3  ...（所有特征列）

        index 为整数（reset 过），date 信息存储在 X_panel.index 的
        ``name`` 属性中为 None——因为各股日期已经被保存在
        ``X_panel`` 的行索引里（通过 ``y_panel.index`` 可以回溯到
        ``(date, stock_code)`` 对）。

        实际上，**date 被保存为 X_panel 的一个额外列**，
        名为 ``"date"``，这样下游可以直接按日期进行 groupby / 切分。

        完整列顺序::

            ["date", "stock_code", feat_1, feat_2, ...]

    **特征列对齐：**

        不同股票可能因容错降级而产生不同的特征列集合
        （例如有换手率的股票多出 turnover_* 列）。
        本函数取所有股票特征列的 **并集**，缺失列用 ``NaN`` 填充。
        这样可以保证面板矩阵列一致，下游 imputer/fillna 可以处理。

    **y_panel 格式：**

        pd.Series，整数 index（与 X_panel 对齐），值为 0/1。

    **meta_dict：**

        {股票代码：FeatureMeta}，记录各股实际生效的特征配置，
        可用于调试（如发现某只股票的 use_turnover 被降级）。

    Args:
        stock_dict: {股票代码：单股 DataFrame}，来自 load_basket()
        config: 特征配置字典（与 build_features_and_labels 相同接口）
        skip_on_error: True 时某只股票特征构建失败则跳过并记录；
                       False 时直接抛出异常

    Returns:
        Tuple[pd.DataFrame, pd.Series, Dict[str, FeatureMeta]]:
            (X_panel, y_panel, meta_dict)

    Raises:
        ValueError: stock_dict 为空，或所有股票均构建失败
        RuntimeError: skip_on_error=False 时某只股票构建抛出的异常
    """
    if not stock_dict:
        raise ValueError(
            "stock_dict 为空，无法构建面板特征。"
            "请先调用 load_basket() 加载至少一只股票。"
        )

    x_frames: List[pd.DataFrame] = []
    y_list:   List[pd.Series]    = []
    meta_dict: Dict[str, FeatureMeta] = {}
    failed_codes: List[str] = []

    for code, df in stock_dict.items():
        try:
            X_i, y_i, meta_i = build_features_and_labels(df, config)
        except Exception as e:
            msg = f"[{code}] build_features_and_labels 失败：{e}"
            if skip_on_error:
                logger.warning("build_panel_features: %s（已跳过）", msg)
                failed_codes.append(code)
                continue
            else:
                raise RuntimeError(msg) from e

        if X_i.empty:
            logger.warning(
                "build_panel_features: [%s] 特征矩阵为空（有效样本数不足），已跳过。",
                code,
            )
            failed_codes.append(code)
            continue

        # 将 date（index）和 stock_code 注入为普通列
        X_with_meta = X_i.copy()
        X_with_meta.insert(0, "stock_code", code)
        X_with_meta.insert(0, "date", X_i.index)    # index 是 pd.Timestamp

        x_frames.append(X_with_meta)
        y_list.append(y_i)
        meta_dict[code] = meta_i

        if meta_i.notes:
            logger.debug(
                "build_panel_features: [%s] 容错降级 notes: %s",
                code, meta_i.notes,
            )

    if not x_frames:
        raise ValueError(
            f"所有 {len(stock_dict)} 只股票均构建特征失败（failed: {failed_codes}）。"
            "请检查数据质量和特征配置。"
        )

    if failed_codes:
        logger.warning(
            "build_panel_features: %d 只股票构建失败，已跳过：%s",
            len(failed_codes), failed_codes,
        )

    # ── 特征列取并集，缺失列填 NaN ────────────────────────
    # 确定所有特征列（排除 date 和 stock_code）
    meta_cols = {"date", "stock_code"}
    all_feat_cols: List[str] = []
    seen: set = set()
    for frame in x_frames:
        for col in frame.columns:
            if col not in meta_cols and col not in seen:
                all_feat_cols.append(col)
                seen.add(col)

    # 统一列顺序后拼接
    full_cols = ["date", "stock_code"] + all_feat_cols
    aligned_frames = []
    for frame in x_frames:
        # 补全缺失的特征列为 NaN
        for col in all_feat_cols:
            if col not in frame.columns:
                frame = frame.copy()
                frame[col] = np.nan
        aligned_frames.append(frame[full_cols])

    X_panel = pd.concat(aligned_frames, axis=0, ignore_index=True)
    y_panel = pd.concat(y_list, axis=0).reset_index(drop=True)

    # ── 按 (date, stock_code) 排序，方便后续按时间切分 ─────
    X_panel = X_panel.sort_values(["date", "stock_code"]).reset_index(drop=True)
    y_panel = y_panel.iloc[X_panel.index].reset_index(drop=True)

    # 重新排序后 y 的 index 已经 reset，需要与 X 重新对齐
    # 通过重建来保证对齐（concat 后各 y_i 的 index 是日期，需要 reset）
    # 更安全的做法：把 y 也放进面板 df，排序，再拆出
    X_panel["_y_tmp"] = y_panel.values
    X_panel = X_panel.sort_values(["date", "stock_code"]).reset_index(drop=True)
    y_panel = X_panel.pop("_y_tmp")

    n_stocks = len(meta_dict)
    n_rows   = len(X_panel)
    n_feats  = len(all_feat_cols)
    date_min = X_panel["date"].min()
    date_max = X_panel["date"].max()

    logger.info(
        "build_panel_features: %d stocks, %d rows, %d features, date %s ~ %s",
        n_stocks, n_rows, n_feats,
        date_min.date() if pd.notna(date_min) else "N/A",
        date_max.date() if pd.notna(date_max) else "N/A",
    )

    # 检查特征列并集中 NaN 填充比例，超过阈值时给出警告
    feat_nan_ratios = X_panel[all_feat_cols].isna().mean()
    high_nan_feats = feat_nan_ratios[feat_nan_ratios > 0.5]
    if not high_nan_feats.empty:
        logger.warning(
            "build_panel_features: 以下特征列有超过 50%% 的 NaN（可能因部分股票降级导致）: %s",
            high_nan_feats.to_dict(),
        )

    return X_panel, y_panel, meta_dict


# =========================================================
# P-basket 新增：横截面特征增强（可选）
# =========================================================

def add_crosssection_features(
    X_panel: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    rank_suffix: str = "_xrank",
    min_stocks_per_date: int = 2,
) -> pd.DataFrame:
    """
    在面板特征基础上，对每个交易日内的股票计算各特征的百分位排名。

    **横截面排名特征的意义：**

        原始特征反映的是每只股票自身的绝对水平（如 RSI=0.7 表示超买）。
        横截面排名特征反映的是相对水平（如 RSI 在当日所有股票中排名前
        10%），能让模型捕捉"当日最强/最弱"的信号，对纯多头选股框架
        尤为重要。

    **排名规则：**

        使用 pandas ``rank(pct=True)``（百分位排名，范围 [0, 1]），
        ``method="average"`` 处理并列，``na_option="keep"`` 保留 NaN。

    **新增列命名：**

        原特征 ``ret_5`` → 新增 ``ret_5_xrank``（等等）。
        原始特征列保留，排名列以 ``rank_suffix`` 为后缀追加。

    **性能说明：**

        对大量特征列做 groupby-transform 会有一定开销。
        如有性能顾虑，可只传入 ``feature_cols`` 中的少数关键列。

    Args:
        X_panel: 含 ``date``、``stock_code`` 及特征列的面板 DataFrame
                 （来自 build_panel_features 的返回值）
        feature_cols: 要计算横截面排名的特征列列表；
                      None 时自动选取所有非 date/stock_code 的数值列
        rank_suffix: 排名列名后缀，默认 ``"_xrank"``
        min_stocks_per_date: 某日有效股票数低于此值时跳过该日排名
                             （NaN 填充），默认 2

    Returns:
        pd.DataFrame: 原 X_panel 加上排名列，index 与输入相同

    Raises:
        ValueError: X_panel 缺少 ``date`` 或 ``stock_code`` 列
    """
    required = {"date", "stock_code"}
    missing = required - set(X_panel.columns)
    if missing:
        raise ValueError(
            f"add_crosssection_features: X_panel 缺少必需列 {missing}。"
            "请确认传入的是 build_panel_features 的返回值。"
        )

    X_out = X_panel.copy()

    # 自动推断要排名的特征列
    if feature_cols is None:
        feature_cols = [
            c for c in X_panel.columns
            if c not in {"date", "stock_code"}
            and pd.api.types.is_numeric_dtype(X_panel[c])
        ]

    if not feature_cols:
        logger.warning("add_crosssection_features: 没有找到可排名的数值列，直接返回原 panel。")
        return X_out

    # 预分配排名列（全 NaN，之后逐日填充）
    rank_col_names = [f"{c}{rank_suffix}" for c in feature_cols]
    for rc in rank_col_names:
        X_out[rc] = np.nan

    # 按日期分组计算百分位排名
    for date_val, grp_idx in X_panel.groupby("date").groups.items():
        grp = X_out.loc[grp_idx, feature_cols]
        n_valid = grp.notna().any(axis=1).sum()

        if n_valid < min_stocks_per_date:
            # 该日有效股票太少，排名无意义，保留 NaN
            continue

        # rank(pct=True) 对每列独立排名，na_option='keep' 保留 NaN
        ranked = grp.rank(pct=True, method="average", na_option="keep")
        ranked.columns = rank_col_names
        X_out.loc[grp_idx, rank_col_names] = ranked.values

    n_new_cols = len(rank_col_names)
    logger.info(
        "add_crosssection_features: 新增 %d 个横截面排名列（suffix='%s'）",
        n_new_cols, rank_suffix,
    )

    return X_out


# =========================================================
# 公开 API 导出列表
# =========================================================
__all__ = [
    # 数据类
    "FeatureMeta",
    # 单股入口（保留，用于 build_panel_features 内部调用）
    "build_features_and_labels",
    # P-basket 新增
    "build_panel_features",
    "add_crosssection_features",
    # 内部工具（供测试）
    "_detect_available_columns",
    "_calc_rsi",
    "_calc_macd",
    "_calc_bband_width",
    "_calc_obv",
]
