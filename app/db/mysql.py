"""MySQL 数据访问层（TDD §12.1）。

基于 SQLAlchemy 2.0：
    - engine：连接池（pool_pre_ping 防止远程断连）
    - session_scope()：事务作用域管理（commit / rollback / close）

配置来源：config.yaml database.mysql（host / port / database / user / password），
可通过环境变量覆盖（APP_DATABASE__MYSQL__HOST / _PORT / _USER / _PASSWORD）。

本阶段只验证连通性（SELECT 1），不创建业务表。
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator, Optional
from urllib.parse import quote_plus

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings, get_settings

# 连接行为参数（客户端默认，非业务配置）
POOL_SIZE = 5
MAX_OVERFLOW = 10
POOL_RECYCLE = 3600  # 连接回收（秒），防 8 小时断连
CONNECT_TIMEOUT = 10  # 建连超时（秒）


class MySQLError(RuntimeError):
    """MySQL 连接或操作错误。"""


class MySQLDatabase:
    """MySQL engine 与 session 管理。"""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        cfg = self.settings.database.mysql
        user = cfg.user or "root"
        password = cfg.password or ""
        self.url = (
            f"mysql+pymysql://{quote_plus(user)}:{quote_plus(password)}"
            f"@{cfg.host}:{cfg.port}/{cfg.database}?charset=utf8mb4"
        )
        self._engine = create_engine(
            self.url,
            pool_size=POOL_SIZE,
            max_overflow=MAX_OVERFLOW,
            pool_recycle=POOL_RECYCLE,
            pool_pre_ping=True,
            connect_args={"connect_timeout": CONNECT_TIMEOUT, "read_timeout": 10, "write_timeout": 10},
        )
        self._session_factory = sessionmaker(
            bind=self._engine, expire_on_commit=False
        )

    @property
    def engine(self) -> Engine:
        return self._engine

    def session(self) -> Session:
        """创建独立 session（调用方负责关闭；推荐用 session_scope）。"""
        return self._session_factory()

    @contextmanager
    def session_scope(self) -> Iterator[Session]:
        """事务作用域：正常提交、异常回滚、结束关闭。"""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except PermissionError:
            session.rollback()
            raise
        except Exception as exc:
            try:
                session.rollback()
            except Exception:
                pass
            if getattr(exc,"domain_error",False):
                raise
            raise MySQLError("mysql_transaction_failed:" + type(exc).__name__) from None
        finally:
            session.close()

    def ping(self) -> bool:
        """连通性检查：SELECT 1，成功返回 True。"""
        try:
            with self._engine.connect() as conn:
                return conn.execute(text("SELECT 1")).scalar() == 1
        except Exception as exc:
            raise MySQLError(
                f"MySQL 连接失败 ({self.settings.database.mysql.host}:"
                f"{self.settings.database.mysql.port}): {exc}"
            ) from exc

    def server_version(self) -> str:
        """返回服务端版本字符串（如 '8.0.36'）。"""
        with self._engine.connect() as conn:
            return str(conn.execute(text("SELECT VERSION()")).scalar())


@lru_cache(maxsize=1)
def get_mysql_database() -> MySQLDatabase:
    """全局 MySQL 单例。"""
    return MySQLDatabase()
