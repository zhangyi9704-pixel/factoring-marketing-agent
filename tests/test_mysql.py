"""MySQL 连接测试。

集成测试需要可达的 MySQL 服务：
    - 默认目标：config.yaml 中的 database.mysql（localhost:3306，可用
      docker-compose.yml 本地启动，账号见 compose 环境变量）
    - 指向远程服务器：APP_DATABASE__MYSQL__HOST=8.133.229.0
      （并按需覆盖 APP_DATABASE__MYSQL__USER / __PASSWORD）

test_url_from_settings 为纯配置装配测试，不触网。
本阶段只验证连通性（SELECT 1），不创建业务表。
"""
import pytest
from sqlalchemy import text

from app.config import Settings
from app.db.mysql import MySQLDatabase


def make_settings() -> Settings:
    return Settings(_env_file=None)


class TestMySQLDatabase:

    def test_url_from_settings(self):
        """连接 URL 由 config.yaml 装配（不建立连接）。"""
        db = MySQLDatabase(make_settings())
        cfg = make_settings().database.mysql
        assert f"@{cfg.host}:{cfg.port}/{cfg.database}" in db.url
        assert "charset=utf8mb4" in db.url

    @pytest.mark.integration
    def test_select_1(self):
        """连通性验证：SELECT 1。"""
        db = MySQLDatabase(Settings())
        assert db.ping() is True

    @pytest.mark.integration
    def test_session_scope_select_1(self):
        """session 事务作用域内执行 SELECT 1。"""
        db = MySQLDatabase(Settings())
        with db.session_scope() as session:
            assert session.execute(text("SELECT 1")).scalar() == 1

    @pytest.mark.integration
    def test_server_version(self):
        """读取服务端版本（信息性断言）。"""
        db = MySQLDatabase(Settings())
        assert db.server_version()
