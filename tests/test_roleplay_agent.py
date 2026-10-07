"""RoleplayAgent 门面测试（全替身注入：FakeLLM + InMemorySaver）。"""
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.roleplay.agent import RoleplayAgent, RoleplayAgentError
from app.config import Settings
from app.guardrail.detector import GuardrailService
from tests.test_case_schema import make_case
from tests.test_roleplay_nodes import (
    FakeCaseService,
    FakeLLM,
    FakeRetriever,
    FakeSessionService,
)


def make_agent(llm=None) -> tuple[RoleplayAgent, FakeSessionService]:
    settings = Settings(_env_file=None)
    recorder = FakeSessionService()
    agent = RoleplayAgent(
        settings,
        llm=llm or FakeLLM(),
        case_service=FakeCaseService(make_case()),
        guardrail=GuardrailService(settings),
        session_service=recorder,
        retriever=FakeRetriever(),          # 单测不触 Milvus
        checkpointer=InMemorySaver(),
    )
    return agent, recorder


class TestStartSession:

    def test_returns_public_case_briefing_and_opening(self):
        agent, recorder = make_agent(FakeLLM(openings=["最近资金压力大。"]))
        result = agent.start_session("INFRA_001", 1001)
        assert result["session_id"].startswith("sess_")
        assert result["public_case"]["company"] == "华建建设有限公司"
        assert "unknown_info" not in result["public_case"]        # 双视角过滤
        assert result["briefing"]["case_id"] == "INFRA_001"       # 访前简报（§7）
        assert result["briefing"]["visit_goals"]
        assert recorder.briefings[result["session_id"]] == result["briefing"]
        assert result["opening"]["content"] == "最近资金压力大。"
        assert result["turn"] == 0
        assert recorder.created == [(result["session_id"], 1001, "INFRA_001")]

    def test_case_not_found_raises(self):
        from tests.test_roleplay_nodes import FakeCaseService as F

        settings = Settings(_env_file=None)
        agent = RoleplayAgent(
            settings, llm=FakeLLM(), case_service=F(None),
            guardrail=GuardrailService(settings),
            session_service=FakeSessionService(), checkpointer=InMemorySaver(),
        )
        with pytest.raises(RoleplayAgentError, match="案例不存在"):
            agent.start_session("NOPE", 1)


class TestSendMessage:

    def test_turn_result_shape(self):
        agent, _ = make_agent()
        created = agent.start_session("INFRA_001", 1001)
        sid = created["session_id"]
        result = agent.send_message(sid, "你们回款压力大吗")
        assert result["turn"] == 1 and result["reply"]
        assert result["finished"] is False
        assert result["guardrail_warnings"] == []

    def test_guardrail_warnings_surface_rule_names_only(self):
        agent, _ = make_agent()
        sid = agent.start_session("INFRA_001", 1001)["session_id"]
        result = agent.send_message(sid, "我保证一周内放款")
        assert result["guardrail_warnings"] == ["保证放款时间"]
        assert "snippet" not in str(result["guardrail_warnings"])

    def test_unknown_session_raises(self):
        agent, _ = make_agent()
        with pytest.raises(RoleplayAgentError, match="不存在"):
            agent.send_message("sess_nope", "你好")

    def test_finished_session_rejected(self):
        agent, _ = make_agent()
        sid = agent.start_session("INFRA_001", 1001)["session_id"]
        agent.send_message(sid, "[结束]")
        with pytest.raises(RoleplayAgentError, match="已结束"):
            agent.send_message(sid, "再聊两句")


class TestFinishAndStatus:

    def test_finish_then_status_finished(self):
        agent, _ = make_agent()
        sid = agent.start_session("INFRA_001", 1001)["session_id"]
        agent.send_message(sid, "你好")
        status = agent.finish_session(sid)
        assert status["finished"] is True and status["waiting_input"] is False
        assert agent.get_status(sid)["turn"] == 1

    def test_finish_idempotent(self):
        agent, _ = make_agent()
        sid = agent.start_session("INFRA_001", 1001)["session_id"]
        first = agent.finish_session(sid)
        second = agent.finish_session(sid)                        # 已结束再调不报错
        assert first["finished"] and second["finished"]

    def test_status_waiting_input_while_active(self):
        agent, _ = make_agent()
        sid = agent.start_session("INFRA_001", 1001)["session_id"]
        status = agent.get_status(sid)                            # resume 端点数据源
        assert status["waiting_input"] is True
        assert status["finished"] is False and status["case_id"] == "INFRA_001"

    def test_status_unknown_session_raises(self):
        agent, _ = make_agent()
        with pytest.raises(RoleplayAgentError, match="不存在"):
            agent.get_status("sess_nope")


class TestSendMessageStream:
    """send_message_stream：custom stream 通道 delta → done（InMemorySaver 全链路）。"""

    def test_deltas_then_done(self):
        reply = {"reply": "我们账期大概 180 天，验收还没完全走完。",
                 "revealed": [], "mood": "neutral"}
        agent, _ = make_agent(FakeLLM(replies=[reply]))
        created = agent.start_session("INFRA_001", 1001)
        events = list(agent.send_message_stream(created["session_id"], "请问账期多久？"))

        deltas = [e for e in events if e["type"] == "delta"]
        done = events[-1]
        assert done["type"] == "done"
        assert done["reply"] == reply["reply"]             # 终帧为权威内容
        assert done["turn"] == 1 and done["finished"] is False
        assert "".join(d["text"] for d in deltas) == reply["reply"]   # 增量拼接无损

    def test_plain_send_message_unchanged(self):
        reply = {"reply": "你先说说方案。", "revealed": [], "mood": "neutral"}
        agent, _ = make_agent(FakeLLM(replies=[reply]))
        created = agent.start_session("INFRA_001", 1001)
        result = agent.send_message(created["session_id"], "你好")
        assert result["reply"] == reply["reply"]           # 非流式入口不受影响

    def test_missing_session_raises(self):
        agent, _ = make_agent()
        with pytest.raises(RoleplayAgentError, match="会话不存在"):
            list(agent.send_message_stream("sess_nope", "你好"))
