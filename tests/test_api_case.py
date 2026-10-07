"""案例 API 单元测试（stub CaseService 注入 + 防泄漏二次校验验证）。"""
import pytest
from fastapi.testclient import TestClient

from app.api.v1.case import get_case_service
from app.main import app
from app.schemas.case import CaseLeakError, to_public_case
from tests.test_case_schema import make_case


class StubCaseService:
    def __init__(self, cases):
        self.cases = cases

    def list_cases(self, industry=None):
        return [c for c in self.cases if not industry or c.industry == industry]

    def get_public_case(self, case_id):
        for c in self.cases:
            if c.case_id == case_id:
                return to_public_case(c)
        return None


def make_client(stub) -> TestClient:
    app.dependency_overrides[get_case_service] = lambda: stub
    return TestClient(app)


def teardown_function():
    app.dependency_overrides.clear()


def make_stub_with_three() -> StubCaseService:
    from app.schemas.case import CaseState
    return StubCaseService([
        make_case(),
        CaseState(case_id="PHARMA_001", industry="医药", company="康泽医药",
                  client_role="经理", unknown_info=["医保占比"]),
        CaseState(case_id="ECOM_001", industry="电商", company="优品家居",
                  client_role="副总"),
    ])


class TestCaseApi:

    def test_list_cases(self):
        client = make_client(make_stub_with_three())
        resp = client.get("/api/v1/cases")
        assert resp.status_code == 200
        body = resp.json()
        assert {c["case_id"] for c in body} == {
            "INFRA_001", "PHARMA_001", "ECOM_001"}
        for c in body:                                   # 销售视角：无内部字段
            assert "unknown_info" not in c and "hidden_risks" not in c

    def test_list_cases_industry_filter(self):
        client = make_client(make_stub_with_three())
        resp = client.get("/api/v1/cases", params={"industry": "基建"})
        assert resp.status_code == 200
        assert [c["case_id"] for c in resp.json()] == ["INFRA_001"]

    def test_case_detail(self):
        client = make_client(make_stub_with_three())
        resp = client.get("/api/v1/cases/INFRA_001")
        assert resp.status_code == 200
        body = resp.json()
        assert body["company"] == "华建建设有限公司"
        assert body["transaction_chain"]["payer"] == "市交通集团"
        assert "unknown_info" not in body

    def test_case_detail_404(self):
        client = make_client(make_stub_with_three())
        resp = client.get("/api/v1/cases/NOT_EXIST")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "案例不存在: NOT_EXIST"

    def test_leaky_payload_blocked_by_second_check(self):
        class LeakyPayload:
            """模拟被污染的 State：model_dump 混入内部键。"""

            def model_dump(self):
                data = to_public_case(make_case()).model_dump()
                data["unknown_info"] = ["不应出现"]
                return data

        class LeakyStub:
            def list_cases(self, industry=None):
                return []

            def get_public_case(self, case_id):
                return LeakyPayload()

        client = make_client(LeakyStub())
        with pytest.raises(CaseLeakError, match="unknown_info"):
            client.get("/api/v1/cases/INFRA_001")       # 二次校验拦截（§19.4）
