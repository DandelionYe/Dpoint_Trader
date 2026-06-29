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
