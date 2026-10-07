"""Priority, credential isolation and SDK boundary checks; no external services."""
import os

import pytest

from app.config import Settings, get_settings
from app.graphs.coach_graph import build_redis_checkpointer, checkpoint_ttl_minutes
from scripts.verify_connections import error_category, model_status, validate_schema
from scripts.verify_connections import verify_model


def test_file_process_and_init_priority(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("APP_DATABASE__MYSQL__PORT=3333\nDEEPSEEK_API_KEY=unit-file-key\n", encoding="utf-8")
    assert Settings(_env_file=env).database.mysql.port == 3333
    assert Settings(_env_file=env).llm.api.api_key == "unit-file-key"
    monkeypatch.setenv("APP_DATABASE__MYSQL__PORT", "4444")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "unit-process-key")
    assert Settings(_env_file=env).database.mysql.port == 4444
    assert Settings(_env_file=env).llm.api.api_key == "unit-process-key"
    assert Settings(_env_file=env, database={"mysql": {"port": 5555}}).database.mysql.port == 5555


def test_dotenv_refs_and_no_process_injection(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("MYSQL_PASSWORD=unit-mysql\nCHECKPOINT_REDIS_PASSWORD=unit-checkpoint\n"
                   "APP_DATABASE__MYSQL__PASSWORD=${MYSQL_PASSWORD}\n"
                   "APP_LANGGRAPH__CHECKPOINT__PASSWORD=${CHECKPOINT_REDIS_PASSWORD}\n"
                   "APP_AUTH__JWT_SECRET=unit-secret\nALIYUN_API_KEY=unit-cloud\n", encoding="utf-8")
    for key in ("MYSQL_PASSWORD", "CHECKPOINT_REDIS_PASSWORD", "ALIYUN_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    before = dict(os.environ)
    configured = Settings(_env_file=env)
    assert configured.database.mysql.password == "unit-mysql"
    assert configured.langgraph.checkpoint.password == "unit-checkpoint"
    assert configured.rag.embedding.api_key == configured.rag.retrieval.rerank.api_key == "unit-cloud"
    assert "unit-secret" not in repr(configured.auth)
    assert dict(os.environ) == before
    isolated = Settings(_env_file=None)
    assert isolated.rag.embedding.api_key is None
    assert isolated.langgraph.checkpoint.password is None


def test_cached_settings_does_not_inject_dotenv(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("DEEPSEEK_API_KEY=unit-cache-key\n", encoding="utf-8")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", env)
    get_settings.cache_clear()
    try:
        assert get_settings().llm.api.api_key == "unit-cache-key"
        assert "DEEPSEEK_API_KEY" not in os.environ
        assert Settings(_env_file=None).llm.api.api_key is None
    finally:
        get_settings.cache_clear()


@pytest.mark.parametrize("seconds", [1, 29, 45, 60, 61, 86400])
def test_seconds_to_sdk_minutes(seconds):
    assert int(checkpoint_ttl_minutes(seconds) * 60) == seconds


def test_checkpoint_uses_independent_password(monkeypatch):
    import redis
    import langgraph.checkpoint.redis as sdk
    seen = {}

    class Client:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    class Saver:
        def __init__(self, **kwargs):
            seen["saver"] = kwargs

        def setup(self):
            seen["setup"] = True

    monkeypatch.setattr(redis, "Redis", Client)
    monkeypatch.setattr(sdk, "RedisSaver", Saver)
    settings = Settings(_env_file=None)
    settings.database.redis.password = "unit-business"
    settings.langgraph.checkpoint.password = "unit-checkpoint"
    settings.langgraph.checkpoint.ttl = 45
    build_redis_checkpointer(settings)
    assert seen["host"] == "127.0.0.1" and seen["port"] == 6380
    assert seen["password"] == "unit-checkpoint" and seen["setup"]
    assert seen["saver"]["ttl"]["default_ttl"] == .75


def test_unprobed_models_are_not_up():
    settings = Settings(_env_file=None)
    settings.llm.api.api_key = "unit-key"
    status = model_status(settings)
    assert status["llm"] == {"status": "configured", "verified": False}
    assert not any(r["status"] == "up" for r in status.values())
    assert error_category(RuntimeError("password=never-output")) == "RuntimeError"


def test_existing_schema_dimension_mismatch_rejected():
    from app.rag.milvus_client import VectorStore
    expected = VectorStore.build_schema(1024)
    description = {"fields": [f.to_dict() for f in expected.fields], "auto_id": False}
    validate_schema(description, expected)
    description["fields"][-2]["params"]["dim"] = 768
    with pytest.raises(AssertionError):
        validate_schema(description, expected)


@pytest.mark.parametrize("kind", ["llm", "embedding", "rerank"])
def test_minimal_cloud_probe_one_request_and_validated_output(kind, monkeypatch):
    import httpx
    settings = Settings(_env_file=None)
    settings.llm.api.api_key = "unit-key"
    settings.rag.embedding.api_key = settings.rag.retrieval.rerank.api_key = "unit-key"
    calls = []

    def handle(request):
        import json
        body = json.loads(request.content)
        calls.append(body)
        if kind == "llm":
            assert body["thinking"] == {"type": "disabled"}
            return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})
        if kind == "embedding":
            assert body["dimensions"] == 1024
            return httpx.Response(200, json={"data": [{"embedding": [1.] * 1024}]})
        return httpx.Response(200, json={"output": {"results": [{"index": 0, "relevance_score": .9}]}})

    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(handle)))
    assert verify_model(settings, kind)["verified"]
    assert len(calls) == 1
