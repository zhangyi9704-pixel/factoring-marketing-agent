"""健康矩阵：真实探测 MySQL/业务 Redis/Milvus/checkpoint。

默认模型只报告 configured/unverified；显式 llm=1 才发出最小 chat 调用。
模型未验证时整体 degraded。异常仅返回类别，不返回连接串或凭据。
checkpoint 健康探测为只读；setup/写读/TTL 由独立连接验收脚本证明。
"""
from __future__ import annotations

import logging
import contextvars
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as ProbeTimeout
from app.services.external_calls import check_cancelled, cancel_event
from typing import Any, Callable, Optional

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="health")
_SLOTS = threading.BoundedSemaphore(4)
_CLOUD_SLOT = threading.BoundedSemaphore(1)


class CloudProbeBusy(RuntimeError):
    """The previous timed-out cloud worker still holds its execution slot."""

COMPONENTS = ("mysql", "redis", "milvus", "checkpoint", "llm", "embedding", "rerank")


class HealthService:
    """连通性矩阵探测（组件可注入，单测用替身）。"""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        mysql: Any = None,
        redis_client: Any = None,
        vector_store: Any = None,
        llm: Any = None,
        checkpoint_probe: Optional[Callable] = None,
        probe_timeout: float = 2.0,
    ) -> None:
        self.settings = settings or get_settings()
        self.probe_timeout = probe_timeout
        self._mysql = mysql
        self._redis = redis_client
        self._vector_store = vector_store
        self._llm = llm
        self._checkpoint_probe = checkpoint_probe or self._probe_checkpoint

    # -------------------------------------------------
    # 懒加载真实组件（测试可注入替身整体绕开）
    # -------------------------------------------------

    @property
    def mysql(self):
        if self._mysql is None:
            from app.db.mysql import get_mysql_database
            self._mysql = get_mysql_database()
        return self._mysql

    @property
    def redis_client(self):
        if self._redis is None:
            from app.db.redis import get_redis_client
            self._redis = get_redis_client()
        return self._redis

    @property
    def vector_store(self):
        if self._vector_store is None:
            from app.rag.milvus_client import VectorStore
            self._vector_store = VectorStore(self.settings)
        return self._vector_store

    @property
    def llm(self):
        if self._llm is None:
            from app.services.llm_service import LLMService
            self._llm = LLMService(self.settings)
        return self._llm

    # -------------------------------------------------
    # 入口
    # -------------------------------------------------

    def check(self, *, include_llm: bool = False) -> dict:
        """探测各组件，返回连通性矩阵。

        Args:
            include_llm: 是否探测 LLM（默认关闭；?llm=1 会调用云 API）。
        """
        components: dict[str, dict] = {
            "mysql": self._probe("MySQL", lambda: self.mysql.server_version()),
            "redis": self._probe("Redis", lambda: self.redis_client.ping()),
            "milvus": self._probe("Milvus", lambda: self.vector_store.ping()),
        }
        components["checkpoint"] = self._probe("Checkpoint", self._checkpoint_probe)
        for kind, key in (("llm", self.settings.llm.api.api_key),
                          ("embedding", self.settings.rag.embedding.api_key),
                          ("rerank", self.settings.rag.retrieval.rerank.api_key)):
            components[kind] = {"status": "configured" if key else "unconfigured", "verified": False,
                                "detail": "unverified"}
        if include_llm:
            components["llm"] = self._probe("LLM", self._probe_llm)
            components["llm"]["verified"] = components["llm"]["status"] == "ok"
        errors = [v["detail"] for v in components.values() if v["status"] == "error"]
        return {
            "status": "healthy" if not errors and all(v["status"] == "ok" for v in components.values()) else "degraded",
            "alive": True,
            "dependencies_available": all(components[n]["status"] == "ok" for n in ("mysql", "redis", "milvus", "checkpoint")),
            "models_verified": all(components[n].get("verified", False) for n in ("llm", "embedding", "rerank")),
            "components": components,
            "errors": errors,
        }

    # -------------------------------------------------
    # 内部
    # -------------------------------------------------

    def _probe(self, name: str, fn: Callable[[], Any]) -> dict:
        """执行一次探测；异常被捕获归为 error（不中断其它组件）。"""
        started = time.monotonic()
        if not _SLOTS.acquire(blocking=False):
            return {"status": "error", "name": name, "detail": "probe_capacity", "elapsed_ms": 0}
        context = contextvars.copy_context()
        parent = cancel_event.get()
        stopped = threading.Event()
        class ProbeCancellation:
            def is_set(self):
                return stopped.is_set() or (parent is not None and parent.is_set())
            def wait(self, seconds):
                return stopped.wait(seconds)
        def contextual():
            token = cancel_event.set(ProbeCancellation())
            try:
                return bounded()
            finally:
                cancel_event.reset(token)
        def bounded():
            try:
                check_cancelled()
                return fn()
            finally:
                _SLOTS.release()
        future = _POOL.submit(context.run, contextual)
        try:
            value = future.result(timeout=self.probe_timeout)
            if value is False:
                raise RuntimeError("probe_returned_false")
            detail = str(value) if value not in (None, True, False) else ""
            return {"status": "ok", "name": name, "detail": detail,
                    "elapsed_ms": round((time.monotonic() - started) * 1000, 2)}
        except ProbeTimeout:
            stopped.set()
            return {"status": "error", "name": name, "detail": "timeout",
                    "elapsed_ms": round((time.monotonic() - started) * 1000, 2)}
        except Exception as exc:
            return {"status": "error", "name": name, "detail": type(exc).__name__,
                    "elapsed_ms": round((time.monotonic() - started) * 1000, 2)}

    def _probe_checkpoint(self):
        import redis
        cfg = self.settings.langgraph.checkpoint
        client = redis.Redis(host=cfg.host, port=cfg.port, password=cfg.password,
                             decode_responses=True, socket_connect_timeout=self.probe_timeout, socket_timeout=self.probe_timeout)
        try:
            if not client.ping():
                raise RuntimeError("checkpoint_ping")
            modules = {m["name"] for m in client.module_list()}
            if not {"ReJSON", "search"} <= modules:
                raise RuntimeError("checkpoint_modules")
            client.execute_command("FT._LIST")
            return "authenticated_ping/json/search"
        finally:
            client.close()
            client.connection_pool.disconnect()

    def _probe_llm(self) -> str:
        """最小 LLM 调用：证明 key/base_url/模型可达、能返回文本。

        要求最小响应非空，空文本不能标记 "ok"；
        任何异常（key 缺失/鉴权失败/网络/超时）都被 _probe catch 归为 error。
        """
        if not _CLOUD_SLOT.acquire(blocking=False):
            raise CloudProbeBusy()
        try:
            check_cancelled()
            text = self.llm.chat("Reply OK", model_role="chat", max_tokens=16, timeout=self.probe_timeout)
            if not text.strip():
                raise RuntimeError("empty_model_response")
            return "ok"
        finally:
            # A handler timeout is not completion of an underlying cloud call.
            _CLOUD_SLOT.release()


__all__ = ["HealthService", "COMPONENTS"]
