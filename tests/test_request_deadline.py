"""Offline boundary tests: a shared budget survives retries, nested work and streaming."""
from types import SimpleNamespace

import httpx
import pytest

from app.config import Settings
from app.services import external_calls as outbound
from app.services.llm_service import LLMService, LLMServiceError


def test_expired_budget_does_not_retry_or_start_followup(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(outbound.time, "monotonic", lambda: now[0])
    calls = outbound.ExternalCalls(Settings(_env_file=None), wait=lambda seconds: now.__setitem__(0, now[0] + seconds))
    attempts = []
    def timeout():
        attempts.append(True)
        now[0] += 2
        raise httpx.ReadTimeout("offline")
    with outbound.request_deadline(1):
        with pytest.raises(outbound.ExternalCallError, match="deadline_exceeded"):
            calls.run("offline", timeout)
        with pytest.raises(outbound.ExternalCallError, match="deadline_exceeded"):
            calls.run("offline", lambda: attempts.append(False))
    assert attempts == [True]
    assert calls.run("offline", lambda: "outside") == "outside"


def test_each_http_retry_uses_remaining_phase_timeouts(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(outbound.time, "monotonic", lambda: now[0])
    seen = []
    class Client:
        timeout = httpx.Timeout(60, connect=5)
        def post(self, url, **kwargs):
            seen.append(kwargs["timeout"])
            if len(seen) == 1:
                now[0] += 2
                raise httpx.ConnectError("offline")
            return httpx.Response(200, json={"ok": True})
    calls = outbound.ExternalCalls(Settings(_env_file=None), wait=lambda seconds: now.__setitem__(0, now[0] + seconds))
    with outbound.request_deadline(10):
        assert calls.post(Client(), "https://offline.invalid", operation="offline") == {"ok": True}
    assert seen[0].read == 10
    assert seen[1].read < 8
    assert seen[1].connect == 5


def test_nested_budget_cannot_extend_parent(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(outbound.time, "monotonic", lambda: now[0])
    with outbound.request_deadline(5):
        with outbound.request_deadline(60):
            assert outbound.remaining_timeout(100) == 5
        assert outbound.remaining_timeout(100) == 5


def test_cached_http_client_does_not_keep_an_earlier_request_budget(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(outbound.time, "monotonic", lambda: now[0])
    seen = []
    def respond(request):
        seen.append(request.extensions["timeout"]["read"])
        return httpx.Response(200, json={"ok": True})
    settings = Settings(_env_file=None)
    calls = outbound.ExternalCalls(settings)
    with outbound.request_deadline(2):
        client = httpx.Client(timeout=outbound.timeout_for(settings, 60), transport=httpx.MockTransport(respond))
        calls.post(client, "https://offline.invalid", operation="offline")
    try:
        calls.post(client, "https://offline.invalid", operation="offline")
    finally:
        client.close()
    assert seen == [2, 60]


def test_stream_stops_and_closes_without_replay_when_budget_expires(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-key")
    now = [100.0]
    monkeypatch.setattr(outbound.time, "monotonic", lambda: now[0])
    stream_closed = []
    attempts = []
    def chunks():
        try:
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="first"))])
            now[0] += 3
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="late"))])
        finally:
            stream_closed.append(True)
    def create(**kwargs):
        attempts.append(kwargs)
        return chunks()
    service = LLMService(Settings(_env_file=None))
    service._client.close()
    service._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    received = []
    with outbound.request_deadline(2):
        with pytest.raises(LLMServiceError, match="deadline_exceeded"):
            service.chat("offline", on_delta=received.append)
    assert received == ["first"]
    assert stream_closed == [True]
    assert len(attempts) == 1
