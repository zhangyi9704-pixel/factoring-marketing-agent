"""Case 库真实 MySQL 集成测试（YAML → t_case 幂等同步 → 双视角回读）。

依赖本机 factoring MySQL（根目录 .env 配置）。
"""
import pytest

from app.schemas.case import assert_no_internal_leak
from app.services.case_service import CaseService


@pytest.mark.integration
class TestCaseIntegration:

    def test_sync_idempotent_and_dual_view_readback(self):
        service = CaseService()                 # get_settings()：.env → 本地 MySQL
        cases = service.load_yaml_cases()

        first = service.sync_to_mysql(cases)    # 首次：插入（或此前已插入则更新）
        second = service.sync_to_mysql(cases)   # 重跑：全部 update，无新插入
        assert second["total"] == len(cases) >= 3
        assert second["inserted"] == 0           # 幂等
        assert first["total"] == second["total"]

        stored = service.list_cases()
        ids = {c.case_id for c in stored}
        assert {"INFRA_001", "PHARMA_001", "ECOM_001"} <= ids

        # agent 全量视角：内部字段完整
        infra = service.get_case("INFRA_001")
        assert infra.company == next(c.company for c in cases if c.case_id == "INFRA_001")
        assert infra.unknown_info and infra.hidden_risks and len(infra.fact_cards) == 16

        # 销售视角：双视角过滤 + API 层二次校验
        public = service.get_public_case("INFRA_001")
        data = public.model_dump()
        assert_no_internal_leak(data)
        assert public.known_facts == [c.fact for c in infra.fact_cards if c.level == "disclosed"]
        assert "payer" not in public.transaction_chain

        assert service.get_case("NOT_EXIST") is None
