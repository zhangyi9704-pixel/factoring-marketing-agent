"""Redis 客户端（TDD §12.2）。

用途（后续阶段）：
    - LangGraph Checkpoint 后端（langgraph-checkpoint-redis）
    - 会话元数据缓存 / Case 读缓存 / 限流计数

配置来源：config.yaml database.redis（host / port / password），
可通过环境变量覆盖（APP_DATABASE__REDIS__HOST / _PORT / _PASSWORD）。
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Optional

import redis

from app.config import Settings, get_settings

# 连接行为参数（客户端默认，非业务配置）
CONNECT_TIMEOUT = 5  # 建连超时（秒）
SOCKET_TIMEOUT = 5   # 读写超时（秒）
MAX_CONNECTIONS = 50  # 连接池上限


class RedisClientError(RuntimeError):
    """Redis 连接或操作错误。"""


class RedisClient:
    """Redis 连接池封装（惰性连接，线程安全）。"""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        cfg = self.settings.database.redis
        self.host = cfg.host
        self.port = cfg.port
        self._pool = redis.ConnectionPool(
            host=cfg.host,
            port=cfg.port,
            password=cfg.password,
            decode_responses=True,  # 返回 str 而非 bytes
            socket_connect_timeout=CONNECT_TIMEOUT,
            socket_timeout=SOCKET_TIMEOUT,
            max_connections=MAX_CONNECTIONS,
        )
        self._client = redis.Redis(connection_pool=self._pool)

    @property
    def pool(self) -> redis.ConnectionPool:
        return self._pool

    @property
    def client(self) -> redis.Redis:
        """redis-py 客户端（惰性连接，首次命令时才真正建连）。"""
        return self._client

    @property
    def url(self) -> str:
        """连接 URL（供 langgraph RedisSaver 等组件使用）。"""
        auth = ":%s@" % self.settings.database.redis.password if self.settings.database.redis.password else ""
        return f"redis://{auth}{self.host}:{self.port}/0"

    @staticmethod
    def _cache_key(key):
        if not key.startswith("factoring:"):
            raise ValueError("cache_namespace")
        return key

    def get_json(self, key):
        raw = self._client.get(self._cache_key(key))
        return json.loads(raw) if raw else None

    def set_json(self, key, value, ttl):
        if ttl <= 0:
            raise ValueError("cache_ttl")
        self._client.set(self._cache_key(key), json.dumps(value, ensure_ascii=False), ex=ttl)

    def ping(self) -> bool:
        """连通性检查，成功返回 True，失败抛出 RedisClientError。"""
        try:
            return bool(self._client.ping())
        except redis.RedisError as exc:
            raise RedisClientError(
                f"Redis 连接失败 ({self.host}:{self.port}): {exc}"
            ) from exc


@lru_cache(maxsize=1)
def get_redis_client() -> RedisClient:
    """全局 Redis 客户端单例。"""
    return RedisClient()
