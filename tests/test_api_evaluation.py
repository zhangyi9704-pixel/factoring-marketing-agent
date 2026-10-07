"""evaluation API 单元测试（stub EvaluationService 注入）。"""
from fastapi.testclient import TestClient

from app.api.v1.evaluation import get_evaluation_service
from app.evaluation.service import EvaluationError
from app.main import app

REPORT = {
    "session_id": "sess_stub",
    "opportunity": {"score": 82, "level": "高潜机会", "reasons": ["付款主体明确"],
                    "dimension_scores": {}},
    "sales_score": {"total": 74, "dimension_scores": {}},
    "missing_questions": [{"item": "验收状态", "impact": "n", "suggestion": "s"}],
    "next_actions": [{"action": "尽调", "priority": "high", "note": "n"}],
    "evidence": [{"dimension": "付款来源", "score": 70, "turn": 1,
                  "quote_sales": "付款方是谁？", "quote_customer": "交通集团",
                  "comment": "明确"}],
    "total_score": 82.0,
    "generated_at": "2026-09-06T12:00:00",
}


class StubService:
    def get(self, session_id):
        if session_id == "sess_missing":
            raise EvaluationError(f"会话不存在: {session_id}")
        return REPORT


def make_client() -> TestClient:
    app.dependency_overrides[get_evaluation_service] = lambda: StubService()
    return TestClient(app)


def teardown_function():
    app.dependency_overrides.clear()


class TestGetEvaluation:

    def test_get_200_report_shape(self):
        resp = make_client().get("/api/v1/sessions/sess_stub/evaluation")
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == "sess_stub"
        assert body["opportunity"]["level"] == "高潜机会"
        assert body["total_score"] == 82.0
        assert body["evidence"][0]["quote_sales"] == "付款方是谁？"

    def test_missing_session_404(self):
        resp = make_client().get("/api/v1/sessions/sess_missing/evaluation")
        assert resp.status_code == 404


class StubStreamService:
    def run_stream(self, session_id):
        if session_id == "sess_missing":
            raise EvaluationError(f"会话不存在: {session_id}")
        yield {"type": "stage", "stage": "evidence", "label": "抽取评分证据",
               "status": "done"}
        yield {"type": "section", "key": "opportunity",
               "data": REPORT["opportunity"]}
        yield {"type": "done", "report": REPORT}


class TestEvaluationStreamSSE:

    def test_stream_frames(self):
        app.dependency_overrides[get_evaluation_service] = lambda: StubStreamService()
        with TestClient(app).stream(
            "GET", "/api/v1/sessions/sess_stub/evaluation/stream",
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            text = "".join(resp.iter_text())
        assert 'event: stage\ndata: {"stage": "evidence"' in text
        assert 'event: section\ndata: {"key": "opportunity"' in text
        assert 'event: done' in text and '"total_score": 82.0' in text

    def test_stream_error_event(self):
        app.dependency_overrides[get_evaluation_service] = lambda: StubStreamService()
        with TestClient(app).stream(
            "GET", "/api/v1/sessions/sess_missing/evaluation/stream",
        ) as resp:
            text = "".join(resp.iter_text())
        assert 'event: error' in text and "会话不存在" in text
