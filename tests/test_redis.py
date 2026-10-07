"""Redis 连接测试。

集成测试（test_ping_real / test_roundtrip）需要可达的 Redis 服务：
    - 默认目标：config.yaml 中的 database.redis（localhost:6379，可用
      docker-compose.yml 本地启动）
    - 指向远程服务器：APP_DATABASE__REDIS__HOST=8.133.229.0

test_pool_uses_settings 为纯配置装配测试，不触网。
"""
import pytest
from redis.exceptions import ConnectionError

from app.config import Settings
from app.db.redis import RedisClient, RedisClientError


def make_settings() -> Settings:
    return Settings(_env_file=None)


class TestRedisClient:

    def test_pool_uses_settings(self):
        """连接池参数与 Settings 装配一致（不建立连接，兼容环境变量覆盖）。"""
        settings = make_settings()
        client = RedisClient(settings)
        kwargs = client.pool.connection_kwargs
        assert kwargs["host"] == settings.database.redis.host
        assert kwargs["port"] == settings.database.redis.port
        assert kwargs["decode_responses"] is True

    @pytest.mark.integration
    def test_ping_real(self):
        """真实连通性验证。"""
        client = RedisClient(Settings())
        assert client.ping() is True

    def test_ping_failure_wrapped(self, monkeypatch):
        """连接失败时包装为 RedisClientError 并包含地址信息。"""
        monkeypatch.setenv("APP_DATABASE__REDIS__HOST", "10.255.255.1")  # 不可达地址
        monkeypatch.setenv("APP_DATABASE__REDIS__PORT", "6379")
        client = RedisClient(make_settings())
        def unavailable():
            raise ConnectionError("unit fake: unavailable")

        monkeypatch.setattr(client.client, "ping", unavailable)
        with pytest.raises(RedisClientError, match="10.255.255.1:6379"):
            client.ping()

    @pytest.mark.integration
    def test_set_get_roundtrip(self):
        """简单读写验证（带 TTL 与测试前缀，用后清理）。"""
        from uuid import uuid4
        client = RedisClient(Settings())
        key = "factoring:test:ping:" + uuid4().hex
        client.client.set(key, "pong", ex=60)
        try:
            assert client.client.get(key) == "pong"
        finally:
            client.client.delete(key)
