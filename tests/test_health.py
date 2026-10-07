"""健康检查 API 单元测试（stub 注入，不触真实 Milvus/Redis/MySQL/LLM）。"""
import pytest
from fastapi.testclient import TestClient

from app.api.v1.health import get_health_service
from app.api.deps import get_auth_service
from app.services.auth_service import Principal, issue_token
from app.config import Settings
from app.main import app
from app.services.health import HealthService


class StubMysql:
    def __init__(self, version="8.0.42", raise_on_call=False):
        self.version = version
        self.raise_on_call = raise_on_call

    def server_version(self):
        if self.raise_on_call:
            raise RuntimeError("mysql boom")
        return self.version


class StubRedis:
    def __init__(self, raise_on_call=False):
        self.raise_on_call = raise_on_call

    def ping(self):
        if self.raise_on_call:
            raise RuntimeError("redis boom")
        return True


class StubMilvus:
    def __init__(self, version="v2.4.4", raise_on_call=False):
        self.version = version
        self.raise_on_call = raise_on_call

    def ping(self):
        if self.raise_on_call:
            raise RuntimeError("milvus boom")
        return self.version


class StubLLM:
    def __init__(self, text="ok", raise_on_call=False):
        self.text = text
        self.raise_on_call = raise_on_call

    def chat(self, *args, **kwargs):
        if self.raise_on_call:
            raise RuntimeError("llm boom")
        return self.text


def make_client(service: HealthService) -> TestClient:
    app.dependency_overrides[get_health_service] = lambda: service
    user = Principal(3, "offline_admin", "admin")
    app.dependency_overrides[get_auth_service] = lambda: type("FakeAuth", (), {"get_user": lambda self, uid: user if uid == user.id else None})()
    return TestClient(app, headers={"Authorization": "Bearer " + issue_token(user)})


@pytest.fixture(autouse=True)
def fresh_cloud_gate(monkeypatch):
    import app.api.v1.health as module
    monkeypatch.setattr(module, "cloud_probe_gate", module.CloudProbeGate(interval=0))


def teardown_function():
    app.dependency_overrides.clear()


def _service(**overrides) -> HealthService:
    kwargs = dict(
        settings=Settings(_env_file=None),
        mysql=StubMysql(),
        redis_client=StubRedis(),
        vector_store=StubMilvus(),
        llm=StubLLM(),
        checkpoint_probe=lambda: "authenticated_ping/json/search",
    )
    kwargs.update(overrides)
    return HealthService(**kwargs)


class TestHealthCheck:

    def test_all_healthy(self):
        resp = make_client(_service()).get("/api/v1/health?llm=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["errors"] == []
        assert data["components"]["mysql"]["status"] == "ok"
        assert data["components"]["mysql"]["detail"] == "8.0.42"
        assert data["components"]["milvus"]["status"] == "ok"
        assert data["components"]["milvus"]["detail"] == "v2.4.4"
        assert data["components"]["redis"]["status"] == "ok"
        assert data["components"]["llm"]["status"] == "ok"

    def test_component_failure_degrades_without_affecting_others(self):
        svc = _service(mysql=StubMysql(raise_on_call=True))
        data = make_client(svc).get("/api/v1/health?llm=1").json()
        assert data["status"] == "degraded"
        assert data["components"]["mysql"]["status"] == "error"
        assert data["components"]["mysql"]["detail"] == "RuntimeError"
        # 其余组件仍正常探测
        assert data["components"]["redis"]["status"] == "ok"
        assert data["components"]["milvus"]["status"] == "ok"
        assert len(data["errors"]) == 1

    def test_llm_failure_is_isolated_component(self):
        svc = _service(llm=StubLLM(raise_on_call=True))
        data = make_client(svc).get("/api/v1/health?llm=1").json()
        assert data["status"] == "degraded"
        assert data["components"]["llm"]["status"] == "error"
        assert data["components"]["mysql"]["status"] == "ok"

    def test_llm_skipped_when_param_false(self):
        svc = _service()
        data = make_client(svc).get("/api/v1/health", params={"llm": False}).json()
        assert data["components"]["llm"]["verified"] is False
        assert data["components"]["checkpoint"]["status"] == "ok"
        assert data["status"] == "degraded"


def test_checkpoint_failure_is_redacted_and_model_default_unverified():
    def failed():
        raise RuntimeError("redis://private:secret@example/")
    data = make_client(_service(checkpoint_probe=failed)).get("/api/v1/health").json()
    assert data["components"]["checkpoint"]["status"] == "error"
    assert "secret" not in str(data)
    assert all(data["components"][name]["verified"] is False for name in ("llm", "embedding", "rerank"))


def test_anonymous_health_is_free_but_cloud_probe_requires_admin():
    class SpyHealth:
        calls = []
        def check(self, include_llm=False):
            self.calls.append(include_llm)
            return {"status": "degraded"}
    service = SpyHealth()
    client = make_client(service)
    assert client.get("/api/v1/health", headers={"Authorization": ""}).status_code == 200
    assert client.get("/api/v1/health?llm=1", headers={"Authorization": ""}).status_code == 401
    for role, enabled in [("sales", True), ("coach", True), ("admin", False)]:
        user = Principal(3, "offline_user", role, enabled)
        app.dependency_overrides[get_auth_service] = lambda: type("FakeAuth", (), {"get_user": lambda self, uid: user})()
        response = client.get("/api/v1/health?llm=1", headers={"Authorization": "Bearer " + issue_token(user)})
        assert response.status_code == (403 if enabled else 401)
    assert service.calls == [False]


def test_cloud_probe_cooldown_and_concurrency_do_not_start_extra_work(monkeypatch):
    import app.api.v1.health as module
    gate = module.CloudProbeGate(clock=lambda: 100)
    monkeypatch.setattr(module, "cloud_probe_gate", gate)
    service = _service()
    client = make_client(service)
    with gate.enter():
        response = client.get("/api/v1/health?llm=1")
        assert response.status_code == 429
        assert int(response.headers["retry-after"]) == 30
    assert client.get("/api/v1/health?llm=1").status_code == 429
    assert client.get("/api/v1/health").status_code == 200


def test_timed_out_cloud_worker_keeps_slot_until_real_call_returns(monkeypatch):
    import threading
    import app.api.v1.health as route
    import app.services.health as health_module
    now = [100]
    monkeypatch.setattr(route, "cloud_probe_gate", route.CloudProbeGate(clock=lambda: now[0]))
    slot = threading.BoundedSemaphore(1)
    monkeypatch.setattr(health_module, "_CLOUD_SLOT", slot)
    entered, release = threading.Event(), threading.Event()
    calls = []
    class BlockingLLM:
        def chat(self, *args, **kwargs):
            calls.append(True)
            entered.set()
            assert release.wait(3), "offline test did not release worker"
            return "ok"
    service = _service(llm=BlockingLLM(), probe_timeout=0.1)
    client = make_client(service)
    try:
        first = client.get("/api/v1/health?llm=1")
        assert entered.is_set()
        assert first.json()["components"]["llm"]["detail"] == "timeout"
        now[0] += 31  # Cooldown elapsed, original provider call still in flight.
        second = client.get("/api/v1/health?llm=1")
        assert second.json()["components"]["llm"]["detail"] == "CloudProbeBusy"
        assert calls == [True]
    finally:
        release.set()
        assert slot.acquire(timeout=2)
        slot.release()
