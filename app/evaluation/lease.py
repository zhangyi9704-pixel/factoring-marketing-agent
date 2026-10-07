"""图外评估领取：生产MySQL连接级锁，数据库提交后才释放。"""
from contextlib import contextmanager
import hashlib
import threading
import time
from sqlalchemy import text

_LOCKS = {}
_MUTEX = threading.Lock()


@contextmanager
def local_lease(session_id):
    """仅用于显式注入的测试存储；不替代生产跨进程数据库锁。"""
    with _MUTEX:
        lock = _LOCKS.setdefault(session_id, threading.Lock())
    with lock:
        yield


class DatabaseLease:
    def __init__(self, engine):
        self.engine = engine

    @contextmanager
    def __call__(self, session_id):
        name = 'factoring:eval:' + hashlib.sha256(session_id.encode()).hexdigest()[:40]
        with self.engine.connect() as connection:
            acquired = False
            try:
                deadline = time.monotonic() + 600
                while time.monotonic() < deadline:
                    acquired = connection.execute(text('SELECT GET_LOCK(:name, 5)'), {'name': name}).scalar() == 1
                    if acquired:
                        break
                if not acquired:
                    raise RuntimeError('evaluation_lease_timeout')
                yield
            finally:
                if acquired:
                    connection.execute(text('SELECT RELEASE_LOCK(:name)'), {'name': name})
