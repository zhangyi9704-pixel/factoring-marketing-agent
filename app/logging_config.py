"""运行期日志与集中错误/异常审计配置（审计增强）。

背景：uvicorn 默认日志配置只配了 uvicorn/uvicorn.error/uvicorn.access 三个 logger，
root 无 handler，导致应用模块 logger 的 INFO 级日志被丢弃（仅 WARNING+ 经 lastResort
落到 stderr）。

本模块：
- setup_logging()：给 root 挂 滚动文件 handler（全级别，修复 INFO 丢失）+ stderr handler
  （WARNING+，保留 server.log 上的错误可见性），并按需挂 DatabaseErrorHandler。
- DatabaseErrorHandler：仅对 ERROR/CRITICAL 或携带异常堆栈（exc_info）的记录，写入
  t_agent_error 表；配合 AUDIT_CONTEXT（请求级 contextvars）关联 session/user/path。
- AUDIT_CONTEXT：由 main.py 的 http 中间件填充，DatabaseErrorHandler 读取。

调用方：app/main.py 在 FastAPI lifespan（startup）里调用 setup_logging()。
对数据库失败静默降级到 stderr，绝不因审计自身异常影响业务。
"""
from __future__ import annotations

import contextvars
import logging
import sys
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable, Optional

from app.config import Settings, get_settings
from app.models import AgentErrorLog

# ---------- 请求级审计上下文（contextvars） ----------
AUDIT_CONTEXT: "contextvars.ContextVar[dict]" = contextvars.ContextVar(
    "audit_context", default={}
)


def set_audit_context(**kwargs) -> None:
    """合并写入本次请求的审计上下文（session_id / user_id / path / method…）。

    忽略 None 值，避免覆盖已有上下文。
    """
    ctx = dict(AUDIT_CONTEXT.get())
    ctx.update({k: v for k, v in kwargs.items() if v is not None})
    AUDIT_CONTEXT.set(ctx)


def clear_audit_context() -> None:
    """清空当前请求的审计上下文（中间件 finally 调用）。"""
    AUDIT_CONTEXT.set({})


def _format_exc(record: logging.LogRecord) -> tuple[Optional[str], Optional[str]]:
    """从 LogRecord 提取 (异常类型名, 完整堆栈)；无异常返回 (None, None)。"""
    if not record.exc_info:
        return None, None
    exc_type = record.exc_info[0].__name__ if record.exc_info[0] else None
    stack = "".join(traceback.format_exception(*record.exc_info))
    return exc_type, stack


class DatabaseErrorHandler(logging.Handler):
    """集中错误/异常审计：命中时写入 t_agent_error 表。

    落库阈值：级别 >= threshold（默认 ERROR）或携带异常堆栈（exc_info）。
    数据库不可用时静默降级到 stderr，不影响业务逻辑。
    """

    def __init__(
        self,
        session_factory: Optional[Callable[[], object]] = None,
        threshold: int = logging.ERROR,
    ) -> None:
        super().__init__()
        self._session_factory = session_factory or self._default_factory
        self._threshold = threshold

    @staticmethod
    def _default_factory():
        """默认：MySQL session_scope 事务（提交/回滚/关闭）。"""
        from app.db.mysql import get_mysql_database

        return get_mysql_database().session_scope()

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno < self._threshold and not record.exc_info:
            return
        exc_type, stack = _format_exc(record)
        ctx = AUDIT_CONTEXT.get()
        row = AgentErrorLog(
            level=record.levelname,
            logger_name=record.name,
            message=record.getMessage(),
            exc_type=exc_type,
            traceback=stack,
            session_id=ctx.get("session_id"),
            user_id=ctx.get("user_id"),
            path=ctx.get("path"),
            method=ctx.get("method"),
        )
        try:
            with self._session_factory() as db:
                db.add(row)
        except Exception:
            self._safe_stderr(
                f"审计落库失败(Logger={record.name}): {record.getMessage()}"
            )

    @staticmethod
    def _safe_stderr(msg: str) -> None:
        try:
            sys.stderr.write(f"[audit-error-handler] {msg}\n")
        except Exception:
            pass


def setup_logging(settings: Optional[Settings] = None) -> None:
    """构造 root logger：滚动文件（全级别）+ stderr（WARNING+）+ DB 错误审计。

    幂等：重复调用不重复添加同名 handler；返回早退。
    """
    settings = settings or get_settings()
    cfg = settings.logging
    root = logging.getLogger()
    level = getattr(logging, cfg.level.upper(), logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s · %(message)s"
    )

    # 1) 滚动文件 handler（全级别，INFO+ 落入 logs/agent.log）
    has_file = any(isinstance(h, RotatingFileHandler) for h in root.handlers)
    if not has_file:
        log_path = Path(cfg.file.path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=cfg.file.max_bytes,
            backupCount=cfg.file.backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(fmt)
        file_handler.setLevel(logging.INFO)  # 全级别（含 INFO），修复丢失
        root.addHandler(file_handler)

    # 2) stderr handler（WARNING+，保留 server.log 上错误/告警可见性）
    has_console = any(
        isinstance(h, logging.StreamHandler)
        and not isinstance(h, RotatingFileHandler)
        for h in root.handlers
    )
    if not has_console:
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(fmt)
        console.setLevel(logging.WARNING)
        root.addHandler(console)

    # 3) 集中错误/异常审计（写 t_agent_error）
    has_audit = any(isinstance(h, DatabaseErrorHandler) for h in root.handlers)
    if cfg.save.audit_error and not has_audit:
        root.addHandler(DatabaseErrorHandler())

    root.setLevel(level)
