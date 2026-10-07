"""Failure branches with transport/database substitutes; no external connections."""
import asyncio
import contextlib
import json
import threading
import time
from types import SimpleNamespace

import httpx
import pytest

from app.config import Settings
from app.services.external_calls import ExternalCalls, ExternalCallError, RequestCancelled, cancel_event
from app.services.llm_service import LLMService, LLMServiceError


def settings():
    cfg = Settings(_env_file=None)
    cfg.llm.api.api_key = "unit-key"
    cfg.rag.embedding.api_key = "unit-key"
    cfg.rag.retrieval.rerank.api_key = "unit-key"
    cfg.llm.retry.backoff_initial = cfg.llm.retry.backoff_max = 0
    return cfg


@pytest.mark.parametrize("code,attempts,category", [(401, 1, "authentication"), (403, 1, "permission"),
    (400, 1, "invalid_request"), (429, 3, "rate_limit"), (503, 3, "server_error"), (500, 1, "server_error")])
def test_classification_bounds_and_redaction(code, attempts, category):
    calls = []
    client = httpx.Client(transport=httpx.MockTransport(lambda req: calls.append(req) or httpx.Response(code, text="private-secret")))
    runner = ExternalCalls(settings(), wait=lambda delay: None)
    with pytest.raises(ExternalCallError) as failure:
        runner.post(client, "https://unit.invalid", operation="embedding")
    assert len(calls) == attempts
    assert failure.value.category == category
    assert "private-secret" not in str(failure.value) + str(runner.last_record)
    assert runner.last_record["retries"] == attempts - 1


def test_timeout_retry_success_usage_and_backoff():
    delays, attempts = [], []
    cfg = settings()
    cfg.llm.retry.backoff_initial = .1
    cfg.llm.retry.backoff_max = .15
    runner = ExternalCalls(cfg, wait=delays.append)
    def call():
        attempts.append(1)
        if len(attempts) < 3:
            raise httpx.ReadTimeout("secret")
        return {"usage": {"total_tokens": 7}}
    runner.run("rerank", call)
    assert delays == [.1, .15]
    assert runner.last_record["usage"] == {"total_tokens": 7}
    assert runner.last_record["elapsed_ms"] >= 0


def test_cancel_before_call_and_during_backoff():
    event = threading.Event()
    token = cancel_event.set(event)
    count = []
    try:
        event.set()
        with pytest.raises(RequestCancelled):
            ExternalCalls(settings()).run("llm", lambda: count.append(1))
        assert not count
        event.clear()
        def failed():
            count.append(1)
            raise httpx.ConnectError("private")
        def wait(_):
            event.set()
        with pytest.raises(RequestCancelled):
            ExternalCalls(settings(), wait=wait).run("llm", failed)
        assert len(count) == 1
    finally:
        cancel_event.reset(token)


@pytest.mark.parametrize("kind", ["embedding", "rerank"])
def test_service_transient_retry_and_auth_immediate(kind):
    from app.rag.embedder import EmbeddingService, EmbeddingError
    from app.rag.reranker import Reranker
    cfg = settings()
    cfg.rag.embedding.dimension = 3
    attempts = []
    def handler(req):
        attempts.append(1)
        if len(attempts) == 1:
            raise httpx.ConnectTimeout("private")
        if kind == "embedding":
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0., 1., 0.]}], "usage": {"total_tokens": 1}})
        return httpx.Response(200, json={"output": {"results": [{"index": 0, "relevance_score": .9}]}})
    client = httpx.Client(transport=httpx.MockTransport(handler))
    service = EmbeddingService(cfg, http_client=client) if kind == "embedding" else Reranker(cfg, http_client=client)
    invoke = (lambda: service.embed(["unit"])) if kind == "embedding" else (lambda: service.rerank("unit", [{"content": "unit"}]))
    invoke()
    assert len(attempts) == 2
    attempts.clear()
    service_client = httpx.Client(transport=httpx.MockTransport(lambda req: attempts.append(1) or httpx.Response(401, text="secret")))
    if kind == "embedding":
        service._http = service_client
        with pytest.raises(EmbeddingError, match="authentication"):
            invoke()
    else:
        service._http_client = service_client
        invoke()  # Existing RRF fallback, with no authentication retry.
    assert len(attempts) == 1


def test_llm_sdk_single_retry_owner_and_partial_stream_no_restart():
    svc = LLMService(settings())
    assert svc._client.max_retries == 0
    requests, pieces, closed = [], [], []
    class Stream:
        def __iter__(self):
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="one"))])
            raise httpx.ReadTimeout("private")
        def close(self):
            closed.append(True)
    def create(**kwargs):
        requests.append(kwargs)
        return Stream()
    svc._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with pytest.raises(LLMServiceError, match="timeout"):
        svc.chat("unit", on_delta=pieces.append)
    assert len(requests) == 1 and pieces == ["one"] and closed == [True]


def test_llm_stream_cancellation_stops_and_closes():
    svc = LLMService(settings())
    event, closed = threading.Event(), []
    class Stream:
        def __iter__(self):
            for piece in ("one", "two"):
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=piece))])
        def close(self):
            closed.append(True)
    svc._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: Stream())))
    token = cancel_event.set(event)
    pieces = []
    try:
        def delta(piece):
            pieces.append(piece)
            event.set()
        with pytest.raises(RequestCancelled):
            svc.chat("unit", on_delta=delta)
    finally:
        cancel_event.reset(token)
    assert pieces == ["one"] and closed == [True]


def test_cache_down_and_owner_gate_before_hit():
    from test_session_service import FakeSessionFactory
    from app.models import CoachSession
    from app.services.session_service import SessionService
    factory = FakeSessionFactory()
    factory.rows.append(CoachSession(session_id="unit", user_id=1, case_id="INFRA_001", phase="roleplay", status="active", turn_count=0))
    class Cache:
        calls = 0
        def get_json(self, key):
            self.calls += 1
            assert key.startswith("factoring:session:1:")
            raise ConnectionError("private")
        def set_json(self, *args):
            raise AssertionError("no write on failed read")
    cache = Cache()
    service = SessionService(settings(), session_factory=factory, cache=cache)
    assert service.get_session("unit", user_id=1)["user_id"] == 1
    assert service.cache_status == "degraded"
    with pytest.raises(PermissionError):
        service.get_session("unit", user_id=2)
    assert cache.calls == 1
    from app.models import VisitReport
    factory.rows.append(VisitReport(session_id="unit", report_json={"case_id": "INFRA_001"}))
    service._use_cache = False
    with pytest.raises(PermissionError):
        service.get_briefing("unit", user_id=2)
    assert cache.calls == 1


def test_cache_stale_or_forged_hit_uses_mysql_and_ttl():
    from test_session_service import FakeSessionFactory
    from app.models import CoachSession
    from app.services.session_service import SessionService
    factory = FakeSessionFactory()
    factory.rows.append(CoachSession(session_id="unit", user_id=1, case_id="INFRA_001", phase="roleplay", status="finished", turn_count=1))
    class Cache:
        def get_json(self, key):
            return {"session_id": "unit", "user_id": 2, "status": "active"}
        def set_json(self, key, value, ttl):
            self.saved = key, value, ttl
    cache = Cache()
    service = SessionService(settings(), session_factory=factory, cache=cache)
    assert service.get_session("unit", user_id=1)["status"] == "finished"
    assert cache.saved[2] == 60 and cache.saved[0].startswith("factoring:")


def test_mysql_commit_error_rolls_back_and_does_not_cache():
    from app.db.mysql import MySQLDatabase, MySQLError
    db = object.__new__(MySQLDatabase)
    marks = []
    class Session:
        def commit(self): raise RuntimeError("password=private")
        def rollback(self): marks.append("rollback")
        def close(self): marks.append("close")
    db._session_factory = Session
    with pytest.raises(MySQLError) as failure:
        with db.session_scope():
            pass
    assert marks == ["rollback", "close"]
    assert "private" not in str(failure.value)
    marks.clear()
    with pytest.raises(PermissionError):
        with db.session_scope():
            raise PermissionError("session_owner_mismatch")
    assert marks == ["rollback", "close"]


def test_node_write_failures_are_not_silent_success():
    from app.agents.roleplay.case_loader import make_init_customer_state_node
    from app.agents.roleplay.guardrail_node import make_update_state_node
    from app.services.session_service import SessionServiceError
    writes = []
    class Store:
        def record_message(self, *args, **kwargs):
            writes.append(1)
            raise RuntimeError("private")
    with pytest.raises(SessionServiceError):
        make_init_customer_state_node(SimpleNamespace(chat=lambda *a, **kw: "ok"), Store())({"session_id": "unit"})
    with pytest.raises(SessionServiceError):
        make_update_state_node(session_service=Store())({"session_id": "unit", "turn": 0})
    assert len(writes) == 2  # one failed write per separate node test; no retry wrapper


def test_health_single_probe_deadline_and_alive_state():
    from app.services.health import HealthService
    done = threading.Event()
    service = HealthService(settings(), probe_timeout=.02)
    try:
        started = time.monotonic()
        result = service._probe("unit", lambda: done.wait(.3))
        assert result["detail"] == "timeout" and time.monotonic() - started < .15
    finally:
        done.set()
    from test_health import _service
    result = _service().check()
    assert result["alive"] and result["dependencies_available"]
    assert not result["models_verified"] and result["status"] == "degraded"


def test_async_ingest_moves_sync_work_off_event_loop():
    from app.api.v1.knowledge import ingest_doc
    from app.rag.ingest import DocResult, STATUS_INDEXED
    origin, worker, ticks = threading.get_ident(), [], []
    class File:
        filename = "unit.md"
        async def read(self): return b"unit"
    class Ingestor:
        settings = SimpleNamespace(knowledge=SimpleNamespace(sources={"product": SimpleNamespace(priority=2)}))
        def ingest_file(self, path, **kwargs):
            worker.append(threading.get_ident())
            time.sleep(.04)
            return DocResult("unit.md", "product", STATUS_INDEXED, 1)
    async def run():
        async def ticker():
            for _ in range(3):
                await asyncio.sleep(.005)
                ticks.append(1)
        task = asyncio.create_task(ticker())
        await ingest_doc(file=File(), source="product", ingestor=Ingestor())
        assert len(ticks) == 3
        await task
    asyncio.run(run())
    assert worker and worker[0] != origin


def test_sse_disconnect_marks_worker_cancelled_without_blocking_loop():
    import anyio
    from app.api.sse import RequestStreamingResponse
    attempted = []
    async def run():
        first = anyio.Event()
        def body():
            yield "data: unit\n\n"
            runner = ExternalCalls(settings())
            def failed():
                attempted.append(1)
                raise httpx.ConnectError("private")
            runner.run("unit", failed)
        async def send(message):
            if message["type"] == "http.response.body": first.set()
        async def receive():
            await first.wait()
            return {"type": "http.disconnect"}
        response = RequestStreamingResponse(body())
        await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
    anyio.run(run)
    assert len(attempted) <= 1


@pytest.mark.parametrize("transient", [True, False])
def test_llm_nonstream_uses_same_retry_and_usage_policy(transient):
    svc = LLMService(settings())
    calls = []
    def create(**kwargs):
        calls.append(1)
        if len(calls) == 1:
            if transient:
                raise httpx.ReadTimeout("private")
            from openai import AuthenticationError
            response = httpx.Response(401, request=httpx.Request("POST", "https://unit.invalid"))
            raise AuthenticationError("private", response=response, body=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                               usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1, total_tokens=3))
    svc._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    if transient:
        assert svc.chat("unit") == "ok"
        assert len(calls) == 2 and svc.calls.last_record["usage"]["total_tokens"] == 3
    else:
        with pytest.raises(LLMServiceError, match="authentication"):
            svc.chat("unit")
        assert len(calls) == 1


def test_write_failure_returns_redacted_503_at_http_boundary():
    from fastapi.testclient import TestClient
    from app.main import app
    from app.api.v1.session import get_roleplay_agent
    from app.services.session_service import SessionServiceError
    class Agent:
        def start_session(self, *args, **kwargs):
            raise SessionServiceError("private_database_connection")
    app.dependency_overrides[get_roleplay_agent] = Agent
    try:
        with TestClient(app) as client:
            response = client.post("/api/v1/sessions", json={"case_id": "INFRA_001", "request_id": "write-failure-create"})
            assert response.status_code == 503
            assert response.json() == {"detail": "mysql_write_failed"}
    finally:
        app.dependency_overrides.clear()
