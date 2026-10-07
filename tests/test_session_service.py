"""SessionService 单元测试（FakeSession 注入，不触库）。"""
import contextlib

from app.models import CoachSession, ComplianceLog, ConversationMessage
from app.services.session_service import SessionService


class FakeSessionFactory:
    def __init__(self):
        self.rows: list = []
        self.added: list = []

    def __call__(self):
        factory = self

        class _Session:
            def add(self, obj):
                factory.added.append(obj)

            def flush(self):
                for obj in factory.added:
                    if obj not in factory.rows:
                        factory.rows.append(obj)

            def query(self, model):
                return FakeQuery(
                    [r for r in factory.rows if isinstance(r, model)], factory,
                )

            def close(self):
                pass

        return contextlib.closing(_Session())


class FakeQuery:
    """支持 == / is_ 真过滤（BinaryExpression + true()/false() 单例）。"""

    def __init__(self, rows, factory):
        self._rows = rows
        self._factory = factory

    def filter(self, *conditions):
        from sqlalchemy import false, true

        rows = self._rows
        for cond in conditions:
            column = getattr(cond, "left", None)
            right = getattr(cond, "right", None)
            if right is true():
                value = True
            elif right is false():
                value = False
            else:
                value = getattr(right, "value", None)
            if column is not None:
                rows = [r for r in rows if getattr(r, column.name, None) == value]
        return FakeQuery(rows, self._factory)

    def order_by(self, *columns):
        if columns:
            key = lambda r: tuple(getattr(r, c.name, None) for c in columns)
            return FakeQuery(sorted(self._rows, key=key), self._factory)
        return self

    def first(self):
        return self._rows[0] if self._rows else None

    def one_or_none(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return list(self._rows)


def make_service() -> tuple[SessionService, FakeSessionFactory]:
    factory = FakeSessionFactory()
    return SessionService(session_factory=factory), factory


class TestSessionLifecycle:

    def test_create_session(self):
        service, factory = make_service()
        service.create_session("s1", 1001, "INFRA_001")
        assert len(factory.added) == 1
        row = factory.added[0]
        assert row.session_id == "s1" and row.status == "active" and row.turn_count == 0

    def test_create_idempotent(self):
        service, factory = make_service()
        factory.rows.append(CoachSession(
            session_id="s1", user_id=1, case_id="INFRA_001", phase="roleplay",
            status="active", turn_count=0))
        service.create_session("s1", 1, "INFRA_001")
        assert factory.added == []                                # 不重复插入

    def test_touch_turn_and_finish(self):
        service, factory = make_service()
        row = CoachSession(session_id="s1", user_id=1, case_id="C",
                           phase="roleplay", status="active", turn_count=0)
        factory.rows.append(row)
        service.touch_session("s1", 3)
        assert row.turn_count == 3 and row.status == "active"
        service.touch_session("s1", 4, finished=True)
        assert row.status == "finished" and row.phase == "evaluate"
        assert row.finished_at is not None

    def test_touch_missing_row_no_error(self):
        service, _ = make_service()
        service.touch_session("ghost", 1)                         # 仅告警不抛

    def test_get_session(self):
        service, factory = make_service()
        factory.rows.append(CoachSession(
            session_id="s1", user_id=7, case_id="ECOM_001",
            phase="roleplay", status="active", turn_count=2))
        data = service.get_session("s1")
        assert data == {"session_id": "s1", "user_id": 7, "case_id": "ECOM_001",
                        "phase": "roleplay", "status": "active", "turn_count": 2, "needs_human_review": False}
        assert service.get_session("ghost") is None


class TestMessagesAndCompliance:

    def test_record_message(self):
        service, factory = make_service()
        service.record_message("s1", 1, "sales", "你好",
                               guardrail_hit=[{"rule": "保证放款时间"}])
        service.record_message("s1", 1, "customer", "你好。",
                               revealed_info=["透露付款方"])
        kinds = [(m.role, m.turn) for m in factory.added]
        assert kinds == [("sales", 1), ("customer", 1)]
        assert factory.added[0].guardrail_hit == [{"rule": "保证放款时间"}]
        assert factory.added[1].revealed_info == ["透露付款方"]

    def test_record_compliance(self):
        service, factory = make_service()
        service.record_compliance("s1", 2, "保证审批通过", "审批肯定过")
        row = factory.added[0]
        assert isinstance(row, ComplianceLog)
        assert row.review_status == "pending" and row.action == "log"

    def test_list_messages_ordered(self):
        service, factory = make_service()
        factory.rows = [
            ConversationMessage(session_id="s1", turn=2, role="sales", content="t2"),
            ConversationMessage(session_id="s1", turn=1, role="customer", content="t1"),
            ConversationMessage(session_id="s1", turn=1, role="sales", content="t0"),
        ]
        messages = service.list_messages("s1")
        assert [(m["turn"], m["role"]) for m in messages] == [
            (1, "customer"), (1, "sales"), (2, "sales")]           # turn → id 稳定序
