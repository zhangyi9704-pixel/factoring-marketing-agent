"""roleplay 真实链路集成测试（Redis checkpoint + MySQL 落库 + DeepSeek chat）。

验收 TDD §20 Phase 2：checkpoint 中断后可恢复续跑；§10.4 fact_constraint
对话循环；§14 guardrail 每轮检测。依赖 .env 真实凭证。
"""
import pytest

from app.agents.roleplay.agent import RoleplayAgent
from app.models import ComplianceLog
from app.services.session_service import SessionService


@pytest.mark.integration
class TestRoleplayIntegration:

    def test_session_flow_with_checkpoint_and_persistence(self):
        agent = RoleplayAgent()                 # 真实组件：DeepSeek + Redis + MySQL
        session_service = SessionService()

        # 1) 创建会话：开场白 + public_case（销售视角）
        import os
        existing = os.environ.get("FACTORING_TEST_EXISTING_ROLEPLAY_SESSION")
        if existing:
            # 只续跑在旧字段断言处中止且尚无销售轮次的会话，不重放初始化。
            status = agent.get_status(existing)
            assert status["case_id"] == "INFRA_001" and status["turn"] == 0 and status["waiting_input"]
            public = agent.case_service.get_public_case("INFRA_001").model_dump()
            opening = session_service.list_messages(existing)[0]
            created = dict(session_id=existing, public_case=public, opening=opening)
        else:
            created = agent.start_session("INFRA_001", 1001)
        sid = created["session_id"]
        assert created["public_case"]["company"] == agent.case_service.get_case("INFRA_001").company
        assert "unknown_info" not in created["public_case"]
        assert created["opening"]["content"]

        # 2) 第 1 轮：普通寒暄（无白名单释放）
        r1 = agent.send_message(
            sid, "王总您好，听说高架桥项目回款压力比较大，主要卡在哪个环节？")
        assert r1["turn"] == 1 and len(r1["reply"]) > 5
        assert r1["finished"] is False and r1["guardrail_warnings"] == []

        # 3) 第 2 轮：事实卡关键词解锁应收/计量
        r2 = agent.send_message(
            sid, "咱们和业主的工程计量与对账完成了吗？这关系到应收的确认。")
        assert r2["turn"] == 2 and r2["reply"]

        # 4) 第 3 轮：违禁话术（guardrail 检测 + 合规留痕）
        r3 = agent.send_message(sid, "放心，我们保证额度一周内就能批下来放款")
        assert r3["guardrail_warnings"], "违禁话术应被检出"

        # 5) resume：checkpoint 快照（§8.3 中断恢复语义）
        status = agent.get_status(sid)
        assert status["waiting_input"] is True and status["turn"] == 3

        # 6) 结束演练
        final = agent.finish_session(sid)
        assert final["finished"] is True and final["waiting_input"] is False

        # 7) MySQL 回读：对话逐轮落库（含 turn 0 开场）
        messages = session_service.list_messages(sid)
        roles = {(m["turn"], m["role"]) for m in messages}
        assert (0, "customer") in roles and (3, "sales") in roles
        sales_rows = [m for m in messages if m["role"] == "sales" and m["turn"] == 3]
        assert sales_rows and sales_rows[0]["guardrail_hit"]    # 违禁留痕在消息行

        # 8) t_session 状态收敛
        row = session_service.get_session(sid)
        assert row["status"] == "finished" and row["turn_count"] == 3
        assert row["phase"] == "evaluate"

    def test_compliance_log_written(self, tmp_session):
        # 独立短会话：验证 t_compliance_log 落库
        agent, sid = tmp_session
        import os
        if not os.environ.get("FACTORING_TEST_EXISTING_COMPLIANCE_SESSION"):
            agent.send_message(sid, "我保证审批肯定能通过，利率也承诺固定")
        session_service = SessionService()
        with session_service.session_factory() as session:
            rows = session.query(ComplianceLog).filter(
                ComplianceLog.session_id == sid,
            ).all()
        assert rows and {r.rule for r in rows} >= {"CMP-03", "CMP-04"}
        assert all(r.review_status == "pending" for r in rows)


@pytest.fixture
def tmp_session():
    """独立迷你会话（复用真实链路，只跑一轮违禁话术）。"""
    agent = RoleplayAgent()
    import os
    existing = os.environ.get("FACTORING_TEST_EXISTING_COMPLIANCE_SESSION")
    if existing:
        assert agent.get_status(existing)["finished"]
        yield agent, existing
        return
    created = agent.start_session("ECOM_001", 1001)
    yield agent, created["session_id"]
    agent.finish_session(created["session_id"])
