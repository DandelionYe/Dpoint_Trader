# constants.py
"""
全局常量集中管理。
所有模块从此处 import，避免跨模块依赖业务逻辑文件。

变更历史:
    原版: Walk-forward 训练约束 / 持久化文件名
    P-basket: 新增 CSV 列名映射、Basket 数据目录约定、组合构建默认参数
"""
from __future__ import annotations

from typing import Dict

# ==============================================================
# Walk-forward 训练约束 / 惩罚项
# （trainer.py 和 reporter.py 共同使用）
# ==============================================================

# 硬约束：每折至少需要的已平仓交易数
# 说明：如果某折的已平仓交易数低于此值，该折的评估结果将被视为无效。
#       这是为了确保每折都有足够的统计显著性。
# 调优建议：对于低频策略（如月频调仓），可降低到 1；
#           对于高频策略（如日频调仓），可提高到 5-10。
MIN_CLOSED_TRADES_PER_FOLD: int = 2

# 软目标：惩罚项对齐的目标已平仓交易数
# 说明：在超参数搜索中，如果某折的交易数偏离此目标，会施加惩罚项。
#       目的是避免搜索到过度交易或交易不足的参数。
# 调优建议：
#   - 过度交易惩罚：如果希望减少交易频率，可提高此值（如 6-8）
#   - 交易不足惩罚：如果希望增加交易频率，可降低此值（如 2-3）
#   - 惩罚项公式：penalty = LAMBDA_TRADE_PENALTY × |actual - target|
TARGET_CLOSED_TRADES_PER_FOLD: int = 4

# 惩罚项强度系数
# 说明：控制交易数偏离目标时对最终指标的惩罚力度。
#       最终指标 = 原始指标 - LAMBDA_TRADE_PENALTY × |actual_trades - target|
# 调优建议：
#   - 0.01-0.03：轻度惩罚，允许一定程度的交易数偏离
#   - 0.05-0.10：中度惩罚，强烈偏好接近目标交易数的配置
#   - > 0.10：重度惩罚，几乎完全由交易数决定优劣（不推荐）
# 示例：
#   假设原始指标 = 1.50，实际交易数 = 8，目标 = 4，lambda = 0.03
#   惩罚后指标 = 1.50 - 0.03 × |8 - 4| = 1.50 - 0.12 = 1.38
LAMBDA_TRADE_PENALTY: float = 0.03


# ==============================================================
# 持久化文件名
# （trainer.py 使用）
# ==============================================================
BEST_SO_FAR_FILENAME: str = "best_so_far.json"
BEST_POOL_FILENAME: str = "best_pool.json"


# ==============================================================
# P-basket: CSV 列名映射
# 将各数据源 CSV 的原始列名统一映射到项目内部列名。
#
# 背景：
#   新版数据来源从 Excel（含 amount/turnover_rate 等富字段）
#   切换为精简版 CSV，仅包含 OHLCV 五列。列名含括号和空格，
#   需要在 load_single_csv() 中做一次集中重命名。
#
# 内部列名约定（与 feature_dpoint.py / backtester.py 保持一致）：
#   date        — 交易日（datetime）
#   open_qfq    — 后复权开盘价
#   high_qfq    — 后复权最高价
#   low_qfq     — 后复权最低价
#   close_qfq   — 后复权收盘价
#   volume      — 成交量（股）
# ==============================================================

# CSV 原始列名 → 项目内部列名
CSV_COL_MAP: Dict[str, str] = {
    "Date":               "date",
    "Open (CNY, qfq)":   "open_qfq",
    "High (CNY, qfq)":   "high_qfq",
    "Low (CNY, qfq)":    "low_qfq",
    "Close (CNY, qfq)":  "close_qfq",
    "Volume (shares)":    "volume",
}

# CSV 加载后必须存在的最小列集合（重命名后）
CSV_REQUIRED_COLS: list[str] = [
    "date",
    "open_qfq",
    "high_qfq",
    "low_qfq",
    "close_qfq",
    "volume",
]

# 从 CSV 衍生的代理字段名（在 load_single_csv 内计算，非 CSV 原始列）
COL_AMOUNT_PROXY: str = "amount_proxy"   # (O+H+L+C)/4 × Volume 估算的成交额代理
COL_LISTING_DAYS: str = "listing_days"   # 上市至今的自然日天数（由文件名推算）


# ==============================================================
# P-basket: Basket 数据目录约定
#
# 目录结构示意：
#   data/
#     basket_1/
#       300299_20120319.csv   ← {股票代码}_{上市日期YYYYMMDD}.csv
#       002555_20110302.csv
#       ...
#     basket_2/
#       ...
#
# 文件命名规则（parse_basket_filename 负责解析）：
#   - 股票代码：文件名下划线前的部分，不限位数（支持 6 位 A 股代码）
#   - 上市日期：下划线后、.csv 前的 8 位数字，格式 YYYYMMDD
#   - 示例：300299_20120319.csv → code="300299", listing_date=date(2012,3,19)
# ==============================================================

# 数据根目录名（相对于项目根，main_cli.py 用于拼接完整路径）
DATA_ROOT_DIR: str = "data"

# Basket 子目录名前缀（实际使用时会拼接序号，如 "basket_1"）
BASKET_DIR_PREFIX: str = "basket"

# 单只股票 CSV 数据长度下限（行数），低于此值时发出警告
MIN_STOCK_ROWS: int = 300

# Basket 内股票数量下限（低于此值时发出警告，不阻止运行）
MIN_BASKET_SIZE: int = 5


# ==============================================================
# P-basket: 组合构建默认参数
# （portfolio_backtester.py 使用；main_cli.py 暴露为 CLI 参数）
#
# 设计原则：
#   所有默认值均面向"中小型股票篮子（20~50 只）+ 周度调仓"场景。
#   用户可通过 CLI 参数或配置文件逐一覆盖。
# ==============================================================

# 每期最大持仓股票数
# 说明：组合中同时持有的最大股票数量。
# 调优建议：
#   - 3-5 只：集中持仓，波动较大，适合风险偏好高的投资者
#   - 5-10 只：适度分散，平衡风险与收益
#   - 10-20 只：高度分散，波动较小，但可能稀释超额收益
# 注意：top_k 不应超过 basket 内股票总数的 1/3，否则分散效果有限。
DEFAULT_TOP_K: int = 5

# 调仓频率：daily | weekly | monthly
# 说明：组合重新平衡的触发频率。
# 各选项含义：
#   - daily：每个交易日都检查并调整持仓（交易成本高，不推荐）
#   - weekly：每周一调仓（推荐，平衡交易成本与信号时效性）
#   - monthly：每月第一个交易日调仓（适合低频策略）
# 调优建议：
#   - 高频信号（如日内动量）：可考虑 weekly
#   - 低频信号（如基本面因子）：建议 monthly
#   - 交易成本敏感：优先 monthly
DEFAULT_REBALANCE_FREQ: str = "weekly"

# 持仓权重方案：equal（等权）| signal（按 dpoint 得分加权）
# 说明：决定每只持仓股票的权重分配方式。
# 各选项含义：
#   - equal：所有持仓股票权重相同（如 top_k=5，则每只 20%）
#   - signal：按 dpoint 预测值比例分配（高分股票权重更高）
# 调优建议：
#   - equal：简单透明，避免模型校准误差影响，推荐用于初期验证
#   - signal：充分利用模型预测强度，但要求模型校准良好
DEFAULT_WEIGHTING_SCHEME: str = "equal"

# 单股最大权重上限（等权时不生效，signal 加权时作为约束）
# 说明：防止单一股票占比过高，控制集中度风险。
# 调优建议：
#   - 0.20-0.30：适度集中，允许前几大持仓占比较高
#   - 0.30-0.40：中等约束，平衡集中与分散
#   - > 0.40：宽松约束，接近等权或自由加权
# 注意：max_weight × top_k 应 >= 1.0，否则无法满仓。
DEFAULT_MAX_WEIGHT: float = 0.3

# 单股最小权重下限（signal 加权时，低于此值的候选不纳入持仓）
# 说明：过滤掉信号强度过低的股票，避免"为了分散而分散"。
# 调优建议：
#   - 0.02-0.05：严格过滤，只持有信号最强的股票
#   - 0.05-0.10：中等过滤，平衡信号强度与分散度
#   - > 0.10：宽松过滤，接近等权分配
# 注意：min_weight × top_k 应 <= 1.0，否则可能无法满仓。
DEFAULT_MIN_WEIGHT: float = 0.05

# 组合回测初始资金（元）；与单股回测的 initial_cash 语义相同
# 说明：模拟组合的起始资金规模，用于计算交易成本和滑点影响。
# 调优建议：
#   - 10 万 -50 万：小资金测试，交易成本影响较大
#   - 50 万 -200 万：中等资金，接近个人投资者实际情况
#   - 200 万 -1000 万：大资金，需考虑市场冲击成本
# 注意：initial_cash 应足够大以支持 top_k 只股票的最小交易单位（100 股/手）。
#       例如：若股票平均价格 50 元，top_k=5，则至少需要 5×100×50=25,000 元。
DEFAULT_PORTFOLIO_INITIAL_CASH: float = 1_000_000.0
