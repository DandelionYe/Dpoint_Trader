# constants.py
"""
全局常量集中管理。
所有模块从此处 import，避免跨模块依赖业务逻辑文件。

合并自：
- Ashare_DpointTrader_deeplearning_Ver2.0/constants.py
- DpointTrader_deeplearning_Ver1.0/constants.py
"""

# ==============================================================
# 数据列定义
# ==============================================================

# 内部标准列名
COL_DATE = "date"
COL_TICKER = "ticker"
COL_OPEN = "open_qfq"
COL_HIGH = "high_qfq"
COL_LOW = "low_qfq"
COL_CLOSE = "close_qfq"
COL_VOLUME = "volume"
COL_AMOUNT = "amount"
COL_TURNOVER = "turnover_rate"

# 必需列（单股模式）
REQUIRED_COLS_SINGLE: list[str] = [
    COL_DATE,
    COL_OPEN,
    COL_HIGH,
    COL_LOW,
    COL_CLOSE,
    COL_VOLUME,
]

# 必需列（面板模式）
REQUIRED_COLS_PANEL: list[str] = [
    COL_DATE,
    COL_TICKER,
    COL_OPEN,
    COL_HIGH,
    COL_LOW,
    COL_CLOSE,
    COL_VOLUME,
]

# 可选列
OPTIONAL_COLS: list[str] = [COL_AMOUNT, COL_TURNOVER]

# CSV 列映射（从外部列名映射到内部标准列名）
DEFAULT_COLUMN_MAP: dict[str, str] = {
    "Date": COL_DATE,
    "Open (CNY, qfq)": COL_OPEN,
    "High (CNY, qfq)": COL_HIGH,
    "Low (CNY, qfq)": COL_LOW,
    "Close (CNY, qfq)": COL_CLOSE,
    "Volume (shares)": COL_VOLUME,
    "Amount (CNY)": COL_AMOUNT,
    "Turnover Rate": COL_TURNOVER,
}

# ==============================================================
# Walk-forward 训练约束 / 惩罚项
# ==============================================================

MIN_CLOSED_TRADES_PER_FOLD: int = 1
TARGET_CLOSED_TRADES_PER_FOLD: int = 4
LAMBDA_TRADE_PENALTY: float = 0.03

# ==============================================================
# 持久化文件名
# ==============================================================

BEST_SO_FAR_FILENAME: str = "best_so_far.json"
BEST_POOL_FILENAME: str = "best_pool.json"

# ==============================================================
# 组合构建常量（篮子模式）
# ==============================================================

DEFAULT_TOP_K: int = 5
DEFAULT_REBALANCE_FREQ: str = "daily"  # daily / weekly / monthly
DEFAULT_MAX_WEIGHT: float = 0.20
DEFAULT_CASH_BUFFER: float = 0.05
DEFAULT_BENCHMARK_MODE: str = "equal_weight"
DEFAULT_WEIGHTING: str = "equal"  # equal / score / vol_inv

# ==============================================================
# 特征工程常量
# ==============================================================

DEFAULT_LABEL_MODE: str = "binary_next_close_up"
DEFAULT_INCLUDE_CROSS_SECTION: bool = True

# ==============================================================
# 回测 / 执行常量
# ==============================================================

DEFAULT_LIMIT_UP_PCT: float = 0.10
DEFAULT_LIMIT_DOWN_PCT: float = 0.10
DEFAULT_LIMIT_UP_PCT_ST: float = 0.05
DEFAULT_LIMIT_DOWN_PCT_ST: float = 0.05
DEFAULT_LIMIT_UP_PCT_CHINEXT_STAR: float = 0.20
DEFAULT_LIMIT_DOWN_PCT_CHINEXT_STAR: float = 0.20
DEFAULT_BOARD_LOT: int = 100
DEFAULT_BUY_COMMISSION_RATE: float = 0.0003
DEFAULT_SELL_COMMISSION_RATE: float = 0.0003
DEFAULT_SELL_STAMP_DUTY_RATE: float = 0.001
DEFAULT_SLIPPAGE_BPS: float = 20.0
DEFAULT_MAX_PARTICIPATION_RATE: float = 0.10
DEFAULT_MIN_TRADE_VALUE: float = 5000.0
DEFAULT_MIN_LISTING_DAYS: int = 60
DEFAULT_FILTER_ST: bool = True

# ==============================================================
# 数据加载常量
# ==============================================================

DEFAULT_DATA_ROOT: str = "./data"
DEFAULT_BASKET_NAME: str = "basket_1"
DEFAULT_FILE_PATTERN: str = "*.csv"
DATA_CONTRACT_VERSION: str = "1.0.0"
