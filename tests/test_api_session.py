"""sessions API 单元测试（stub RoleplayAgent 注入）。"""
import pytest
from fastapi.testclient import TestClient

from app.agents.roleplay.agent import RoleplayAgentError
from app.api.v1.session import get_roleplay_agent
from app.main import app


class StubAgent:
    """按 TDD §15 示例行为打桩。"""

    def start_session(self, case_id, user_id, *, request_id=None):
        if case_id == "NOPE":
            raise RoleplayAgentError(f"案例不存在: {case_id}")
        return {
            "session_id": "sess_stub", "case_id": case_id, "user_id": user_id,
            "public_case": {"company": "华建建设有限公司", "industry": "基建",
                            "known_facts": ["回款周期长"]},
            "briefing": {"case_id": case_id, "industry": "基建",
                         "visit_goals": ["确认交易背景真实性"]},
            "opening": {"role": "customer", "turn": 0, "content": "最近资金压力大。"},
            "turn": 0,
        }

    def send_message(self, session_id, content, **kwargs):
        if session_id == "sess_missing":
            raise RoleplayAgentError(f"会话不存在: {session_id}")
        if session_id == "sess_done":
            raise RoleplayAgentError("会话已结束，无法继续发送消息")
        return {
            "turn": 1, "reply": "主要是上游供应商的付款……", "finished": False,
            "finish_reason": "", "guardrail_warnings": ["保证放款时间"],
            "needs_human_review": False,
        }

    def send_message_stream(self, session_id, content, **kwargs):
        if session_id == "sess_missing":
            raise RoleplayAgentError(f"会话不存在: {session_id}")
        yield {"type": "delta", "text": "主要"}
        yield {"type": "delta", "text": "是上游供应商……"}
        yield {"type": "done",
               "turn": 1, "reply": "主要是上游供应商的付款……", "finished": False,
               "finish_reason": "", "guardrail_warnings": ["保证放款时间"],
               "needs_human_review": False}

    def finish_session(self, session_id):
        if session_id == "sess_missing":
            raise RoleplayAgentError(f"会话不存在: {session_id}")
        return {"session_id": session_id, "case_id": "INFRA_001", "turn": 3,
                "finished": True, "finish_reason": "sales_end",
                "waiting_input": False, "mood": "neutral",
                "needs_human_review": False}

    def get_status(self, session_id):
        if session_id == "sess_missing":
            raise RoleplayAgentError(f"会话不存在: {session_id}")
        return {"session_id": session_id, "case_id": "INFRA_001", "turn": 1,
                "finished": False, "finish_reason": "", "waiting_input": True,
                "mood": "neutral", "needs_human_review": False}


def make_client() -> TestClient:
    app.dependency_overrides[get_roleplay_agent] = lambda: StubAgent()
    return TestClient(app)


def teardown_function():
    app.dependency_overrides.clear()


class TestCreateSession:

    def test_create_201(self):
        resp = make_client().post("/api/v1/sessions",
                                  json={"case_id": "INFRA_001", "request_id": "create-unit", "user_id": 1001})
        assert resp.status_code == 201
        body = resp.json()
        assert body["session_id"] == "sess_stub"
        assert body["public_case"]["company"] == "华建建设有限公司"
        assert "unknown_info" not in body["public_case"]
        assert body["briefing"]["visit_goals"] == ["确认交易背景真实性"]  # §7 简报随响应返回
        assert body["opening"]["turn"] == 0

    def test_case_not_found_404(self):
        resp = make_client().post("/api/v1/sessions", json={"case_id": "NOPE", "request_id": "create-missing"})
        assert resp.status_code == 404


class TestMessages:

    def test_message_200(self):
        resp = make_client().post("/api/v1/sessions/sess_stub/messages",
                                  json={"request_id": "unit-request", "expected_turn": 0, "content": "张总，咱们项目上目前最大的资金压力在哪个环节？"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["turn"] == 1 and body["finished"] is False
        assert body["guardrail_warnings"] == ["保证放款时间"]

    def test_empty_content_422(self):
        resp = make_client().post("/api/v1/sessions/sess_stub/messages",
                                  json={"request_id": "unit-request", "expected_turn": 0, "content": ""})
        assert resp.status_code == 422

    def test_missing_session_404(self):
        resp = make_client().post("/api/v1/sessions/sess_missing/messages",
                                  json={"request_id": "unit-request", "expected_turn": 0, "content": "你好"})
        assert resp.status_code == 404

    def test_finished_session_409(self):
        resp = make_client().post("/api/v1/sessions/sess_done/messages",
                                  json={"request_id": "unit-request", "expected_turn": 0, "content": "你好"})
        assert resp.status_code == 409


class TestFinishAndResume:

    def test_finish_200(self):
        resp = make_client().post("/api/v1/sessions/sess_stub/finish")
        assert resp.status_code == 200
        assert resp.json()["finished"] is True

    def test_resume_returns_waiting_input(self):
        resp = make_client().post("/api/v1/sessions/sess_stub/resume")
        assert resp.status_code == 200
        assert resp.json()["waiting_input"] is True

    def test_resume_missing_404(self):
        resp = make_client().post("/api/v1/sessions/sess_missing/resume")
        assert resp.status_code == 404


class TestMessageStreamSSE:

    def test_stream_frames(self):
        with make_client().stream(
            "POST", "/api/v1/sessions/sess_stub/messages/stream",
            json={"request_id": "unit-request", "expected_turn": 0, "content": "张总，资金压力主要在哪？"},
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            text = "".join(resp.iter_text())
        assert 'event: delta\ndata: {"text": "主要"}' in text
        assert '"text": "是上游供应商……"' in text
        assert 'event: done' in text
        assert '"turn": 1' in text and '"needs_human_review"' in text

    def test_stream_error_event(self):
        with make_client().stream(
            "POST", "/api/v1/sessions/sess_missing/messages/stream",
            json={"request_id": "unit-request", "expected_turn": 0, "content": "你好"},
        ) as resp:
            assert resp.status_code == 200                  # 流已开，错误走事件
            text = "".join(resp.iter_text())
        assert 'event: error' in text and "会话不存在" in text
