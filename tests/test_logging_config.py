"""运行期日志 / 集中错误审计 单元测试（Fake 注入，不触真实 MySQL）。"""
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.config import Settings
from app.logging_config import (
    AUDIT_CONTEXT,
    DatabaseErrorHandler,
    clear_audit_context,
    set_audit_context,
    setup_logging,
)
from app.main import audit_kwargs_for


def _reset_own_handlers():
    """移除本模块添加的 handler（保留 pytest/caplog 等既有 handler），保证测试隔离。"""
    root = logging.getLogger()
    root.handlers = [
        h for h in root.handlers
        if not isinstance(h, (RotatingFileHandler, DatabaseErrorHandler))
    ]


class FakeScope:
    """伪 session_scope：捕获 add 的行，__exit__ 记录 flush。"""

    def __init__(self, sink):
        self._sink = sink

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def add(self, row):
        self._sink.append(row)


class FakeSessionFactory:
    def __init__(self, fail=False):
        self.rows = []
        self.fail = fail

    def __call__(self):
        if self.fail:
            raise RuntimeError("mysql down")
        return FakeScope(self.rows)


def _record(level, msg="boom", logger="audit.test", exc_info=None):
    return logging.LogRecord(
        logger, level, "t.py", 1, msg, (), exc_info=exc_info
    )


class TestAuditContext:
    def teardown_method(self):
        clear_audit_context()

    def test_merge_ignores_none(self):
        set_audit_context(path="/a", method="POST", session_id="s1")
        assert AUDIT_CONTEXT.get() == {"path": "/a", "method": "POST", "session_id": "s1"}
        # None 值应忽略，不覆盖已有字段
        set_audit_context(session_id=None, user_id=7)
        assert AUDIT_CONTEXT.get() == {
            "path": "/a", "method": "POST", "session_id": "s1", "user_id": 7,
        }

    def test_clear_resets(self):
        set_audit_context(path="/a", method="GET")
        clear_audit_context()
        assert AUDIT_CONTEXT.get() == {}

    def test_audit_kwargs_parses_path(self):
        assert audit_kwargs_for("/api/v1/sessions/s1/messages/stream", "POST") == {
            "path": "/api/v1/sessions/s1/messages/stream", "method": "POST",
            "session_id": "s1",
        }
        assert audit_kwargs_for("/api/v1/users/7/scores", "GET") == {
            "path": "/api/v1/users/7/scores", "method": "GET", "user_id": 7,
        }
        assert audit_kwargs_for("/api/v1/health", "GET") == {
            "path": "/api/v1/health", "method": "GET",
        }


class TestDatabaseErrorHandler:

    def test_error_record_emitted(self):
        fake = FakeSessionFactory()
        handler = DatabaseErrorHandler(session_factory=fake)
        handler.emit(_record(logging.ERROR, "ml flow failed"))
        assert len(fake.rows) == 1
        row = fake.rows[0]
        assert row.level == "ERROR"
        assert row.message == "ml flow failed"
        assert row.exc_type is None and row.traceback is None

    def test_info_below_threshold_ignored(self):
        fake = FakeSessionFactory()
        DatabaseErrorHandler(session_factory=fake).emit(_record(logging.INFO, "hi"))
        assert fake.rows == []

    def test_warning_with_exception_emitted(self):
        # 即便 level 是 WARNING，只要携带异常堆栈也应审计
        fake = FakeSessionFactory()
        handler = DatabaseErrorHandler(session_factory=fake)
        handler.emit(_record(logging.WARNING, "degraded", exc_info=(
            ValueError, ValueError("bad"), None,
        )))
        assert len(fake.rows) == 1
        row = fake.rows[0]
        assert row.exc_type == "ValueError"
        assert "ValueError: bad" in row.traceback

    def test_merges_audit_context(self):
        fake = FakeSessionFactory()
        handler = DatabaseErrorHandler(session_factory=fake)
        set_audit_context(path="/api/v1/sessions/s9/finish", method="POST", session_id="s9", user_id=3)
        handler.emit(_record(logging.ERROR, "ctx"))
        row = fake.rows[0]
        assert row.session_id == "s9" and row.user_id == 3
        assert row.path == "/api/v1/sessions/s9/finish" and row.method == "POST"
        clear_audit_context()

    def test_db_failure_swallowed(self):
        # 数据库异常不得向上抛/影响业务
        fake = FakeSessionFactory(fail=True)
        handler = DatabaseErrorHandler(session_factory=fake)
        handler.emit(_record(logging.ERROR, "down"))   # 不应 raise


class TestSetupLogging:

    def test_adds_file_handler_and_persists_info(self, tmp_path):
        _reset_own_handlers()
        s = Settings(_env_file=None)
        s.logging.file.path = str(tmp_path / "agent.log")
        setup_logging(s)
        root = logging.getLogger()
        files = [h for h in root.handlers if isinstance(h, RotatingFileHandler)]
        assert files, "未挂载滚动文件 handler"
        assert Path(files[0].baseFilename) == tmp_path / "agent.log"

        logging.getLogger("audit.filet").info("info-marker-123")
        logging.getLogger("audit.filet").warning("warn-456")
        for h in files:
            h.flush()
        text = (tmp_path / "agent.log").read_text(encoding="utf-8")
        assert "info-marker-123" in text      # INFO 不再被丢弃
        assert "warn-456" in text

    def test_idempotent_no_duplicate_file_handler(self, tmp_path):
        _reset_own_handlers()
        s = Settings(_env_file=None)
        s.logging.file.path = str(tmp_path / "agent.log")
        setup_logging(s)
        setup_logging(s)   # 重复调用
        files = [h for h in logging.getLogger().handlers
                 if isinstance(h, RotatingFileHandler)]
        assert len(files) == 1

    def test_audit_handler_attached_by_default(self, tmp_path):
        # save.audit_error 默认 True：setup_logging 后 root 应挂 DatabaseErrorHandler
        _reset_own_handlers()
        s = Settings(_env_file=None)
        s.logging.file.path = str(tmp_path / "audit.log")
        setup_logging(s)
        assert any(isinstance(h, DatabaseErrorHandler)
                   for h in logging.getLogger().handlers)
