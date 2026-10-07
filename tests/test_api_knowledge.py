"""knowledge 注入 API 单元测试（stub ingestor 注入，不触 Milvus/embedder）。"""
import io
import pytest

from fastapi.testclient import TestClient

from app.api.v1.knowledge import get_ingestor
from app.config import Settings
from app.main import app
from app.rag.ingest import STATUS_FAILED, STATUS_INDEXED, DocResult


class StubIngestor:
    """带真 Settings + 覆写 ingest_file 的替身（只验证端点编排，不跑真实入库）。"""

    def __init__(self, result=None, raise_on_call=False):
        self.settings = Settings(_env_file=None)
        self.result = result or DocResult("d.md", "product", STATUS_INDEXED, chunk_cnt=3)
        self.called_with = None
        self.raise_on_call = raise_on_call

    def ingest_file(self, path, *, source, priority):
        self.called_with = {"path": str(path), "source": source, "priority": priority}
        if self.raise_on_call:
            raise RuntimeError("embedder 挂了")
        return self.result


def make_client(stub=None) -> TestClient:
    app.dependency_overrides[get_ingestor] = lambda: stub or StubIngestor()
    return TestClient(app)


def teardown_function():
    app.dependency_overrides.clear()


class TestSearch:
    @pytest.mark.parametrize("mode,reason", [("rag", None), ("conflict", None), ("missing", "out_of_scope")])
    def test_only_three_modes_and_citations(self, mode, reason):
        from app.api.v1.knowledge import get_retriever
        from app.rag.retriever import RetrievalResult, RetrievalItem
        class Retriever:
            def retrieve(self, query, top_k, industry):
                item = RetrievalItem("c", "p", "child", "脱敏来源", "示例.md", 1, "依据摘要", .9,
                    doc_id="KB-TEST", version="v1", retrieval_score=.032, rerank_score=.9)
                return RetrievalResult(query, [] if mode == "missing" else [item], mode=mode,
                    missing=mode == "missing", reason=reason, escalation_id="esc_test" if mode != "rag" else None)
        app.dependency_overrides[get_retriever] = Retriever
        response = TestClient(app).post("/api/v1/knowledge/search", json={"query": "脱敏问题", "industry": "医药"})
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == mode and data["reason"] == reason
        if data["items"]:
            assert data["items"][0]["doc_id"] == "KB-TEST" and data["items"][0]["retrieval_score"] == .032

    def test_industry_whitelist(self):
        from app.api.v1.knowledge import get_retriever
        app.dependency_overrides[get_retriever] = lambda: object()
        response = TestClient(app).post("/api/v1/knowledge/search", json={"query": "脱敏问题", "industry": "其他行业"})
        assert response.status_code == 422


class TestIngest:

    def test_ingest_ok(self):
        stub = StubIngestor()
        resp = make_client(stub).post(
            "/api/v1/knowledge/ingest",
            params={"source": "product"},
            files={"file": ("note.md", io.BytesIO(b"# hello"), "text/markdown")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "indexed"
        assert body["doc_name"] == "d.md" and body["chunk_cnt"] == 3
        assert stub.called_with["source"] == "product"
        assert stub.called_with["priority"] == 2          # config knowledge.sources.product.priority

    def test_unknown_source_400(self):
        resp = make_client().post(
            "/api/v1/knowledge/ingest",
            params={"source": "nope"},
            files={"file": ("a.md", io.BytesIO(b"x"), "text/markdown")},
        )
        assert resp.status_code == 400

    def test_failed_status_500(self):
        stub = StubIngestor(DocResult("f.md", "product", STATUS_FAILED, error="boom"))
        resp = make_client(stub).post(
            "/api/v1/knowledge/ingest",
            params={"source": "product"},
            files={"file": ("f.md", io.BytesIO(b"x"), "text/markdown")},
        )
        assert resp.status_code == 500

    def test_ingestor_raises_500(self):
        resp = make_client(StubIngestor(raise_on_call=True)).post(
            "/api/v1/knowledge/ingest",
            params={"source": "product"},
            files={"file": ("r.md", io.BytesIO(b"x"), "text/markdown")},
        )
        assert resp.status_code == 500
