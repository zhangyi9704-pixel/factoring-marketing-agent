"""Explicit, sanitized connection probes; never run coach_graph or initialize business data."""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.graphs.coach_graph import build_redis_checkpointer


class ProbeError(RuntimeError):
    pass


def error_category(exc):
    # Exception messages may contain credentials/connection URLs. Never serialize them.
    code = getattr(exc, "status_code", None)
    if code is None and getattr(exc, "response", None) is not None:
        code = exc.response.status_code
    if code:
        return {400: "request_rejected", 401: "authentication", 403: "permission",
                404: "model_or_endpoint_not_found", 429: "rate_limit"}.get(code, "remote_http_error")
    return type(exc).__name__


def probe(fn):
    try:
        return {"status": "up", **(fn() or {})}
    except Exception as exc:
        return {"status": "error", "error_category": error_category(exc)}


def verify_mappings(settings, env_file=ROOT / ".env"):
    from dotenv import dotenv_values
    values = dotenv_values(env_file)
    pairs = {
        "mysql_database": (settings.database.mysql.database, values.get("MYSQL_DATABASE")),
        "mysql_user": (settings.database.mysql.user, values.get("MYSQL_USER")),
        "mysql_password": (settings.database.mysql.password, values.get("MYSQL_PASSWORD")),
        "redis_password": (settings.database.redis.password, values.get("REDIS_PASSWORD")),
        "checkpoint_password": (settings.langgraph.checkpoint.password, values.get("CHECKPOINT_REDIS_PASSWORD")),
        "llm_key": (settings.llm.api.api_key, values.get("DEEPSEEK_API_KEY")),
        "embedding_key": (settings.rag.embedding.api_key, values.get("ALIYUN_API_KEY")),
        "rerank_key": (settings.rag.retrieval.rerank.api_key, values.get("ALIYUN_API_KEY")),
    }
    matches = {name: bool(actual) and actual == expected for name, (actual, expected) in pairs.items()}
    if not all(matches.values()):
        # Report names and booleans only, never either side's value.
        return {"status": "error", "matches": matches, "error_category": "credential_mapping"}
    return {"status": "up", "matches": matches}


def verify_mysql(settings):
    from sqlalchemy import text
    from app.db.mysql import MySQLDatabase
    db = MySQLDatabase(settings)
    try:
        with db.engine.connect() as connection:
            assert connection.execute(text("SELECT 1")).scalar() == 1
            assert connection.execute(text("SELECT DATABASE()")).scalar() == settings.database.mysql.database
            account = connection.execute(text("SELECT CURRENT_USER()")).scalar()
            assert account.split("@")[0] == settings.database.mysql.user
        return {"select_1": True, "configured_database": True, "application_account": True}
    finally:
        db.engine.dispose()


def verify_redis(settings):
    from app.db.redis import RedisClient
    client = RedisClient(settings)
    try:
        assert client.ping()
        return {"authenticated_ping": True}
    finally:
        client.client.close()
        client.pool.disconnect()


def verify_checkpoint(settings):
    import redis
    from langgraph.checkpoint.base import empty_checkpoint
    from langgraph.checkpoint.redis import RedisSaver
    cfg = settings.langgraph.checkpoint
    base = "factoring_envcheck_" + uuid.uuid4().hex
    prefixes = {"checkpoint_prefix": base + "_checkpoint", "checkpoint_write_prefix": base + "_write"}
    thread_id = base + "_thread"
    client = redis.Redis(host=cfg.host, port=cfg.port, password=cfg.password,
                         decode_responses=True, socket_connect_timeout=5, socket_timeout=15)
    try:
        assert client.ping()
        modules = {m["name"] for m in client.module_list()}
        assert {"ReJSON", "search"} <= modules
        saver = build_redis_checkpointer(settings, **prefixes)
        try:
            snapshot = empty_checkpoint()
            snapshot["channel_values"] = {"marker": thread_id}
            snapshot["channel_versions"] = {"marker": "1"}
            config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
            written = saver.put(config, snapshot, {"source": "input", "step": 0}, {"marker": "1"})
            restored = saver.get_tuple(written)
            assert restored.checkpoint["channel_values"]["marker"] == thread_id
            # A fresh client/saver proves state does not depend on the first object's caches.
            other = RedisSaver(redis_client=client, ttl=saver.ttl_config, **prefixes)
            assert other.get_tuple(config).checkpoint["channel_values"]["marker"] == thread_id
            key = saver._make_redis_checkpoint_key_cached(thread_id, "", snapshot["id"])
            assert 0 < client.ttl(key) <= cfg.ttl
            result = {"authenticated_ping": True, "json_search": True, "setup": True,
                      "write_read": True, "fresh_saver_read": True, "ttl_seconds": cfg.ttl}
        finally:
            saver._redis.close()
            saver._redis.connection_pool.disconnect()
    finally:
        # Only our UUID prefix: no FLUSHDB, business keys, shared indexes or collections.
        try:
            for key in client.scan_iter(match=base + "*"):
                client.delete(key)
            names = set(client.execute_command("FT._LIST"))
            for name in prefixes.values():
                if name in names:
                    client.execute_command("FT.DROPINDEX", name)
            assert not list(client.scan_iter(match=base + "*"))
            assert not set(prefixes.values()) & set(client.execute_command("FT._LIST"))
        finally:
            client.close()
            client.connection_pool.disconnect()
    return {**result, "cleanup": True}


def validate_schema(description, expected):
    actual = {f["name"]: f for f in description["fields"]}
    wanted = {f.name: f.to_dict() for f in expected.fields}
    assert set(actual) == set(wanted), "schema_fields"
    for name, field in wanted.items():
        other = actual[name]
        assert int(other["type"]) == int(field["type"]), "schema_type"
        assert bool(other.get("is_primary", False)) == bool(field.get("is_primary", False)), "schema_primary"
        for param in ("dim", "max_length"):
            if param in field.get("params", {}):
                assert int(other.get("params", {}).get(param, -1)) == int(field["params"][param]), "schema_parameter"
    assert bool(description.get("auto_id")) == bool(wanted["id"].get("auto_id")), "schema_auto_id"


def verify_milvus(settings):
    from app.rag.milvus_client import VectorStore
    store = VectorStore(settings)
    try:
        version = store.ping()
        if not str(version).lstrip("v").startswith("2.5."):
            raise ProbeError("milvus_version")
        initialized = store.client.has_collection(store.collection_name)
        if initialized:
            validate_schema(store.client.describe_collection(store.collection_name), store.build_schema(store.dimension))
        return {"server_version": version, "collection": "schema_valid" if initialized else "not_initialized"}
    finally:
        if store._client is not None:
            store._client.close()


def model_status(settings):
    return {kind: {"status": "configured" if key else "unconfigured", "verified": False}
            for kind, key in (("llm", settings.llm.api.api_key), ("embedding", settings.rag.embedding.api_key),
                              ("rerank", settings.rag.retrieval.rerank.api_key))}


def verify_model(settings, kind):
    import httpx
    llm, embedding, rerank = settings.llm, settings.rag.embedding, settings.rag.retrieval.rerank
    if kind == "llm":
        key, url = llm.api.api_key, llm.api.base_url.rstrip("/") + "/chat/completions"
        body = {"model": llm.models.chat.name, "messages": [{"role": "user", "content": "Reply OK."}],
                "max_tokens": 16, "thinking": {"type": "disabled"}}
    elif kind == "embedding":
        assert embedding.provider == "dashscope"
        key, url = embedding.api_key, embedding.base_url.rstrip("/") + "/embeddings"
        body = {"model": embedding.model, "input": ["保理融资"], "dimensions": embedding.dimension}
    else:
        assert rerank.provider == "dashscope"
        key, url = rerank.api_key, rerank.base_url
        body = {"model": rerank.model, "input": {"query": "保理融资", "documents": ["保理融资依赖真实应收账款。"]},
                "parameters": {"top_n": 1, "return_documents": False}}
    if not key:
        raise ProbeError("missing_key")
    # Exactly one external request per model, no SDK retries or graph execution.
    with httpx.Client(timeout=httpx.Timeout(30, connect=15)) as client:
        response = client.post(url, headers={"Authorization": "Bearer " + key}, json=body)
        response.raise_for_status()
        data = response.json()
    if kind == "llm":
        assert data["choices"] and data["choices"][0]["message"].get("content")
        return {"verified": True, "role": "chat", "reasoning": "unverified"}
    if kind == "embedding":
        vector = data["data"][0]["embedding"]
        assert len(vector) == embedding.dimension == 1024
        assert all(math.isfinite(float(v)) for v in vector)
        return {"verified": True, "dimension": len(vector)}
    scores = data["output"]["results"]
    assert scores and int(scores[0]["index"]) == 0 and math.isfinite(float(scores[0]["relevance_score"]))
    return {"verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", action="store_true", help="Explicit one-call LLM/embedding/rerank cloud probes")
    parser.add_argument("--models-only", action="store_true", help="With --models: do not repeat middleware write/read probes")
    parser.add_argument("--model-kind", choices=("llm", "embedding", "rerank"), nargs="+",
                        help="With --models: select only failed/unverified external probes")
    parser.add_argument("--report", type=Path, help="Save sanitized JSON only")
    args = parser.parse_args()
    if args.models_only and not args.models:
        parser.error("--models-only requires --models")
    if args.model_kind and not args.models:
        parser.error("--model-kind requires --models")
    logging.disable(logging.CRITICAL)
    try:
        settings = Settings()
    except Exception as exc:
        print(json.dumps({"status": "error", "error_category": error_category(exc)}))
        return 1
    results = {"credential_mapping": probe(lambda: verify_mappings(settings))}
    if not args.models_only:
        for name, fn in (("mysql", verify_mysql), ("redis", verify_redis),
                         ("checkpoint", verify_checkpoint), ("milvus", verify_milvus)):
            results[name] = probe(lambda fn=fn: fn(settings))
            print(name + ": " + json.dumps(results[name], ensure_ascii=False), flush=True)
    results["models"] = model_status(settings)
    if args.models:
        for name in args.model_kind or results["models"]:
            results["models"][name] = probe(lambda name=name: verify_model(settings, name))
            print("model_" + name + ": " + json.dumps(results["models"][name], ensure_ascii=False), flush=True)
    else:
        print("models: " + json.dumps(results["models"]), flush=True)
    failed = any(r.get("status") == "error" for r in results.values()) or any(
        r.get("status") == "error" for r in results["models"].values())
    report = {"status": "error" if failed else "passed", "components": results}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("RESULT: " + report["status"], flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
