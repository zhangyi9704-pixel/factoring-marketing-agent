"""用户成绩画像 API 单元测试（stub UserScoreService 注入）。"""
from fastapi.testclient import TestClient

from app.api.v1.user import get_user_score_service
from app.main import app
from app.services.user_score_service import UserScoreServiceError

REPORT = {
    "user_id": 7,
    "total_sessions": 3,
    "evaluated_sessions": 2,
    "avg_opportunity": 71.0,
    "avg_sales_score": 70.0,
    "level_distribution": {"高潜机会": 1, "观察": 1, "暂缓": 0},
    "dimension_averages": {"payer_confirmation": 70.0},
    "latest_sessions": [{"session_id": "s1", "case_id": "INFRA_001"}],
    "coaching": {
        "opportunity": {"avg_score": 71.0, "level_distribution": {}, "dimension_averages": []},
        "review": {"avg_total": 70.0, "dimension_averages": []},
        "evidence": {"by_dimension": [], "latest": {"session_id": "", "items": []}},
        "missing": [],
        "improvement": {"weakest_dimensions": [], "evidence_gaps": [], "next_actions": []},
    },
}


class StubService:
    def scores(self, user_id):
        if user_id == 999:
            raise UserScoreServiceError("DB 挂")
        return dict(REPORT, user_id=user_id)


def make_client() -> TestClient:
    app.dependency_overrides[get_user_score_service] = lambda: StubService()
    return TestClient(app)


def teardown_function():
    app.dependency_overrides.clear()


class TestUserScores:

    def test_200_report_shape(self):
        resp = make_client().get("/api/v1/users/7/scores")
        assert resp.status_code == 200
        body = resp.json()
        assert body["user_id"] == 7
        assert body["avg_opportunity"] == 71.0
        assert body["avg_sales_score"] == 70.0
        assert body["level_distribution"]["高潜机会"] == 1
        assert body["latest_sessions"][0]["session_id"] == "s1"
        assert body["coaching"]["opportunity"]["avg_score"] == 71.0   # 五段式辅导随响应返回

    def test_service_error_500(self):
        resp = make_client().get("/api/v1/users/999/scores")
        assert resp.status_code == 500


class StubStreamService:
    def scores_stream(self, user_id):
        if user_id == 999:
            raise UserScoreServiceError("DB 挂")
        profile = {k: v for k, v in REPORT.items() if k != "coaching"}
        yield {"type": "profile", "data": profile}
        yield {"type": "section", "key": "opportunity",
               "data": REPORT["coaching"]["opportunity"]}
        yield {"type": "done"}


class TestUserScoresStreamSSE:

    def test_stream_frames(self):
        app.dependency_overrides[get_user_score_service] = lambda: StubStreamService()
        with TestClient(app).stream(
            "GET", "/api/v1/users/7/scores/stream",
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            text = "".join(resp.iter_text())
        assert 'event: profile' in text and '"avg_opportunity": 71.0' in text
        assert 'event: section\ndata: {"key": "opportunity"' in text
        assert 'event: done' in text

    def test_stream_error_event(self):
        app.dependency_overrides[get_user_score_service] = lambda: StubStreamService()
        with TestClient(app).stream(
            "GET", "/api/v1/users/999/scores/stream",
        ) as resp:
            text = "".join(resp.iter_text())
        assert 'event: error' in text and "DB 挂" in text
