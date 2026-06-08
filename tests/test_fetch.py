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


class TestFormatterEdgeCases:
    """测试格式转换器的边界情况。"""

    def test_missing_required_columns(self):
        """缺少必需列应抛出 ValueError。"""
        from dpoint.data.fetch.formatter import qmt_to_dpoint_single

        raw = pd.DataFrame({"time": [1609459200000], "open": [10.0]})
        with pytest.raises(ValueError, match="缺少必需列"):
            qmt_to_dpoint_single(raw)

    def test_generate_csv_filename_edge_cases(self):
        """文件名生成的边界情况。"""
        from dpoint.data.fetch.formatter import generate_csv_filename

        # 代码带多个点
        name = generate_csv_filename("000001.SZ.HK", "20210101")
        assert name == "000001_20210101.csv"
