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
        logger.info(
            "Fetching %s (%s ~ %s, %s)",
            stock_code,
            start_date or "all",
            end_date or "latest",
            period,
        )

        # Step 1: 下载到本地缓存
        # 兼容不同版本的 xtquant：部分版本不支持 keyword-only 参数
        # 先尝试关键字参数，失败则回退到位置参数
        if not hasattr(self, "_download_kw_supported"):
            self._download_kw_supported = True  # 乐观假设，首次失败后切换

        if self._download_kw_supported:
            try:
                self._xtdata.download_history_data(
                    stock_code=stock_code,
                    period=period,
                    start_time=start_date,
                    end_time=end_date,
                )
            except TypeError:
                # 关键字参数不支持，切换到位置参数模式
                self._download_kw_supported = False
                self._xtdata.download_history_data(stock_code, period, start_date, end_date)
        else:
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

        # 标准化列名和索引名（xtquant 可能返回混合大小写）
        df.columns = [str(c).lower() for c in df.columns]
        if df.index.name:
            df.index.name = str(df.index.name).lower()

        # 确保 time 列存在
        if "time" not in df.columns and df.index.name in ("time", "timetag"):
            df = df.reset_index()

        logger.info("Fetched %d rows for %s", len(df), stock_code)
        return df

    def fetch_batch(
        self,
        stock_codes: list[str],
        period: str = "1d",
        start_date: str = "",
        end_date: str = "",
        dividend_type: str = "front",
    ) -> dict[str, pd.DataFrame]:
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

        result: dict[str, pd.DataFrame] = {}
        for i, code in enumerate(stock_codes, 1):
            logger.info("[%d/%d] Fetching %s", i, len(stock_codes), code)
            try:
                df = self.fetch_daily_history(code, period, start_date, end_date, dividend_type)
                if not df.empty:
                    result[code] = df
                else:
                    logger.warning("Empty data for %s, skipping", code)
            except (OSError, RuntimeError, KeyError, ValueError) as e:
                logger.error("Failed to fetch %s: %s", code, e)

        logger.info("Batch complete: %d/%d succeeded", len(result), len(stock_codes))
        return result
