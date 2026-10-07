"""CaseService 单元测试（FakeSession 注入 + 真实 cases/ YAML 校验，不触库）。"""
import contextlib
from pathlib import Path

import pytest
from sqlalchemy import false, true

from app.models import Case
from app.schemas.case import CaseState, assert_no_internal_leak, to_public_case
from app.services.case_service import CaseService, CaseServiceError
from tests.test_case_schema import make_case


class FakeQuery:
    """支持 == / is_ 条件真过滤的假 Query（识别 SQLAlchemy BinaryExpression）。

    注意 col.is_(True) 右侧是 true()/false() 单例而非 bind 值，需单独映射。
    """

    def __init__(self, rows):
        self._rows = list(rows)

    @staticmethod
    def _right_value(right):
        if right is true():
            return True
        if right is false():
            return False
        return getattr(right, "value", None)

    def filter(self, *conditions):
        rows = self._rows
        for cond in conditions:
            column = getattr(cond, "left", None)
            value = self._right_value(getattr(cond, "right", None))
            if column is not None:
                rows = [r for r in rows if getattr(r, column.name, None) == value]
        return FakeQuery(rows)

    def order_by(self, *columns):
        if columns:
            key = lambda r: tuple(getattr(r, c.name, None) for c in columns)
            return FakeQuery(sorted(self._rows, key=key))
        return self

    def one_or_none(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return list(self._rows)


class FakeSessionFactory:
    def __init__(self, rows=None):
        self.rows: list = list(rows or [])
        self.added: list[Case] = []

    def __call__(self):
        factory = self

        class _Session:
            def add(self, obj):
                factory.added.append(obj)

            def query(self, model):
                return FakeQuery(factory.rows)

            def close(self):
                pass

        return contextlib.closing(_Session())


def make_service(tmp_path=None, rows=None) -> tuple[CaseService, FakeSessionFactory]:
    factory = FakeSessionFactory(rows)
    service = CaseService(
        session_factory=factory,
        case_dir=tmp_path if tmp_path else Path(__file__).parents[1] / "cases",
    )
    return service, factory


def make_row(case: CaseState) -> Case:
    return Case(case_id=case.case_id, industry=case.industry, company=case.company,
                client_role=case.client_role, case_json=case.model_dump(), enabled=True)


# =====================================================
# YAML 加载（真实 cases/ 目录）
# =====================================================

class TestLoadYamlCases:

    def test_three_industry_cases_valid(self):
        cases = CaseService().load_yaml_cases()
        ids = [c.case_id for c in cases]
        assert ids == ["ECOM_001", "INFRA_001", "PHARMA_001"]   # sorted
        assert {c.industry for c in cases} == {"基建", "医药", "电商"}
        for case in cases:
            assert case.unknown_info, f"{case.case_id} 应有 unknown_info"
            assert case.behavior_rules, f"{case.case_id} 应有 behavior_rules"
            assert case.hidden_risks, f"{case.case_id} 应有 hidden_risks"

    def test_infra_case_matches_tdd_spec(self):
        cases = {c.case_id: c for c in CaseService().load_yaml_cases()}
        infra = cases["INFRA_001"]
        assert infra.company == "程诚建材贸易有限公司"
        assert infra.transaction_chain.payer == "市城投下属总包、民企总包"
        assert infra.receivable.amount == 28000000
        assert infra.receivable.aging_days is None
        assert any("财政拨付" in item for item in infra.unknown_info)

    def test_missing_dir_raises(self, tmp_path):
        service = CaseService(case_dir=tmp_path / "nope")
        with pytest.raises(CaseServiceError, match="不存在"):
            service.load_yaml_cases()

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(CaseServiceError, match="为空"):
            CaseService(case_dir=tmp_path).load_yaml_cases()

    def test_bad_yaml_raises(self, tmp_path):
        (tmp_path / "broken.yaml").write_text("industry: 123\ncompany: [unclosed",
                                              encoding="utf-8")
        with pytest.raises(CaseServiceError, match="解析失败"):
            CaseService(case_dir=tmp_path).load_yaml_cases()

    def test_non_yaml_files_ignored(self, tmp_path):
        (tmp_path / "readme.txt").write_text("x", encoding="utf-8")
        (tmp_path / "A.yaml").write_text(
            "case_id: A\nindustry: 其他\ncompany: C\nclient_role: R",
            encoding="utf-8")
        cases = CaseService(case_dir=tmp_path).load_yaml_cases()
        assert [c.case_id for c in cases] == ["A"]


# =====================================================
# 同步（幂等 upsert）
# =====================================================

class TestSyncToMysql:

    def test_first_sync_inserts(self):
        service, factory = make_service()
        cases = [make_case()]
        result = service.sync_to_mysql(cases)
        assert result["total"] == 1 and result["inserted"] == 1 and result["updated"] == 0
        assert len(factory.added) == 1
        row = factory.added[0]
        assert row.case_id == "INFRA_001" and row.enabled is True
        assert row.case_json["unknown_info"] == cases[0].unknown_info

    def test_second_sync_updates_idempotent(self):
        rows = [make_row(make_case())]
        service, factory = make_service(rows=rows)
        result = service.sync_to_mysql([make_case()])
        assert result == {"total": 1, "inserted": 0, "updated": 1,
                          "case_ids": ["INFRA_001"]}
        assert factory.added == []                       # 不重复插入
        assert rows[0].company == "华建建设有限公司"        # 已被刷新


# =====================================================
# 查询（list / get / public）
# =====================================================

class TestQueries:

    def make_three_rows(self):
        infra = make_case()
        pharma = CaseState(case_id="PHARMA_001", industry="医药",
                           company="康泽医药", client_role="经理",
                           unknown_info=["x"])
        ecom = CaseState(case_id="ECOM_001", industry="电商", company="优品家居",
                         client_role="副总")
        return [make_row(infra), make_row(pharma), make_row(ecom)]

    def test_list_cases_returns_all_sorted(self):
        service, _ = make_service(rows=self.make_three_rows())
        ids = [c.case_id for c in service.list_cases()]
        assert ids == ["ECOM_001", "INFRA_001", "PHARMA_001"]

    def test_list_filters_industry(self):
        service, _ = make_service(rows=self.make_three_rows())
        assert [c.case_id for c in service.list_cases("基建")] == ["INFRA_001"]

    def test_get_case_full_view(self):
        service, _ = make_service(rows=self.make_three_rows())
        case = service.get_case("PHARMA_001")
        assert case.company == "康泽医药"
        assert case.unknown_info == ["x"]                 # agent 视角保留

    def test_get_case_missing_returns_none(self):
        service, _ = make_service(rows=self.make_three_rows())
        assert service.get_case("NOT_EXIST") is None

    def test_get_public_case_filtered(self):
        service, _ = make_service(rows=self.make_three_rows())
        public = service.get_public_case("INFRA_001")
        data = public.model_dump()
        assert_no_internal_leak(data)
        assert public.company == "华建建设有限公司"

    def test_disabled_case_invisible(self):
        rows = self.make_three_rows()
        rows[1].enabled = False                           # PHARMA 停用
        service, _ = make_service(rows=rows)
        ids = [c.case_id for c in service.list_cases()]
        assert "PHARMA_001" not in ids
        assert service.get_case("PHARMA_001") is None

    def test_public_conversion_safe(self):
        # 兜底：任意 CaseState 转公视角后均无内部键
        for row in self.make_three_rows():
            assert_no_internal_leak(to_public_case(CaseState(**row.case_json)).model_dump())
