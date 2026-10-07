"""同会话互斥与持久化请求账本；失败记录绝不自动重放图。"""
from contextlib import contextmanager
import copy
import hashlib
import threading
from sqlalchemy import text, update
from app.models import MessageRequestRecord, SessionCreateRequestRecord, CoachSession, BusinessAudit


class MessageConflict(RuntimeError):
    pass


def payload_hash(content, expected_turn):
    return hashlib.sha256(f'{expected_turn}\0{content}'.encode()).hexdigest()


def turn_result(state):
    """One whitelist for the graph boundary and transactional result journal."""
    return dict(turn=state.get('turn', 0), reply=state.get('customer_reply', ''),
        finished=bool(state.get('finished')), finish_reason=state.get('finish_reason', ''),
        l2_degraded=bool((state.get('compliance_status') or {}).get('l2_degraded')),
        escalation_id=state.get('compliance_escalation_id'),
        guardrail_warnings=[h.get('rule') for h in state.get('guardrail_hits') or []],
        needs_human_review=bool(state.get('needs_human_review')), input_blocked=bool(state.get('input_blocked')),
        guardrail_events=[dict(rule=h['rule'], severity=h.get('severity', 'warn'),
            action=h.get('action', 'log'), compliant_response=h.get('compliant_response', ''))
            for h in state.get('guardrail_hits') or []])


def replay(record, fingerprint):
    if record['payload_hash'] != fingerprint:
        raise MessageConflict('request_id_conflict')
    if record['status'] == 'completed':
        return copy.deepcopy(record['result'])
    raise MessageConflict('request_interrupted' if record['status'] == 'running' else 'request_failed')


class DatabaseRequests:
    def __init__(self, database, *, lock_provider=None):
        self.database = database
        self.lock_provider = lock_provider

    @contextmanager
    def lock(self, sid):
        if self.lock_provider is not None:  # Explicit offline test adapter only.
            with self.lock_provider(sid):
                yield
            return
        name = 'factoring:message:' + hashlib.sha256(sid.encode()).hexdigest()[:40]
        with self.database.engine.connect() as conn:
            # 不排队重跑旧轮次；0秒领取超时明确返回session_busy。
            acquired = conn.execute(text('SELECT GET_LOCK(:name, 0)'), {'name': name}).scalar() == 1
            if not acquired:
                raise MessageConflict('session_busy')
            try:
                yield
            finally:
                conn.execute(text('SELECT RELEASE_LOCK(:name)'), {'name': name})

    def get(self, sid, rid):
        with self.database.session_scope() as db:
            row = db.query(MessageRequestRecord).filter_by(session_id=sid, request_id=rid).one_or_none()
            return None if row is None else dict(payload_hash=row.payload_hash, status=row.status,
                result=copy.deepcopy(row.result), expected_turn=row.expected_turn)

    def begin(self, sid, rid, fingerprint, expected_turn):
        with self.database.session_scope() as db:
            db.add(MessageRequestRecord(session_id=sid, request_id=rid, payload_hash=fingerprint,
                expected_turn=expected_turn, status='running'))

    def complete(self, sid, rid, result):
        with self.database.session_scope() as db:
            row = db.query(MessageRequestRecord).filter_by(session_id=sid, request_id=rid).one()
            row.status, row.result = 'completed', copy.deepcopy(result)

    def fail(self, sid, rid):
        with self.database.session_scope() as db:
            # Lost commit acknowledgement must not overwrite a committed result.
            db.execute(update(MessageRequestRecord).where(MessageRequestRecord.session_id == sid,
                MessageRequestRecord.request_id == rid, MessageRequestRecord.status == 'running').values(
                    status='failed', error_code='message_execution_failed'))

    def reconcile(self, sid, rid, result, actor_id):
        with self.database.session_scope() as db:
            row = db.query(MessageRequestRecord).filter_by(session_id=sid, request_id=rid).one()
            if row.status not in {'running', 'failed'} or row.result != result:
                raise MessageConflict('reconciliation_state_changed')
            row.status, row.error_code = 'completed', 'verified_reconciliation'
            db.add(BusinessAudit(actor_id=actor_id, queue='session', action='message_reconciled',
                details=dict(session_id=sid, request_id=rid, turn=result['turn'])))

    def unfinished(self, sid):
        with self.database.session_scope() as db:
            return db.query(MessageRequestRecord).filter(MessageRequestRecord.session_id == sid,
                MessageRequestRecord.status.in_(['running', 'failed'])).first() is not None

    def first_unfinished(self, sid):
        with self.database.session_scope() as db:
            row = db.query(MessageRequestRecord).filter(MessageRequestRecord.session_id == sid,
                MessageRequestRecord.status.in_(['running', 'failed'])).order_by(MessageRequestRecord.id).first()
            return row.request_id if row else None


class MemoryRequests:
    """仅显式替身测试/开发；不能作为生产请求账本。"""
    def __init__(self):
        self.records, self.locks, self.mutex = {}, {}, threading.Lock()

    @contextmanager
    def lock(self, sid):
        with self.mutex:
            lock = self.locks.setdefault(sid, threading.Lock())
        if not lock.acquire(blocking=False):
            raise MessageConflict('session_busy')
        try:
            yield
        finally:
            lock.release()

    def get(self, sid, rid):
        return copy.deepcopy(self.records.get((sid, rid)))

    def begin(self, sid, rid, fingerprint, expected_turn):
        self.records[(sid, rid)] = dict(payload_hash=fingerprint, expected_turn=expected_turn, status='running')

    def complete(self, sid, rid, result):
        self.records[(sid, rid)].update(status='completed', result=copy.deepcopy(result))

    def fail(self, sid, rid):
        if self.records[(sid, rid)]['status'] == 'running':
            self.records[(sid, rid)]['status'] = 'failed'

    def prepare_result(self, sid, rid, result):
        self.records[(sid, rid)]['result'] = copy.deepcopy(result)

    def reconcile(self, sid, rid, result, actor_id):
        row = self.records[(sid, rid)]
        if row['status'] not in {'running', 'failed'} or row.get('result') != result:
            raise MessageConflict('reconciliation_state_changed')
        row.update(status='completed', error_code='verified_reconciliation', reconciled_by=actor_id)

    def abort(self, sid):
        for (session_id, _), row in self.records.items():
            if session_id == sid and row['status'] in {'running', 'failed'}:
                row['status'] = 'aborted'

    def unfinished(self, sid):
        return any(s == sid and r['status'] in {'running', 'failed'} for (s, _), r in self.records.items())

    def first_unfinished(self, sid):
        return next((rid for (s, rid), row in self.records.items() if s == sid and row['status'] in {'running', 'failed'}), None)


class DatabaseCreations(DatabaseRequests):
    """Creation request + initializing session are committed in one SQL transaction."""
    def get(self, uid, rid):
        with self.database.session_scope() as db:
            row = db.query(SessionCreateRequestRecord).filter_by(user_id=uid, request_id=rid).one_or_none()
            return None if row is None else dict(case_id=row.case_id, session_id=row.session_id,
                status=row.status, result=copy.deepcopy(row.result), error_code=row.error_code)

    def for_session(self, sid):
        with self.database.session_scope() as db:
            row = db.query(SessionCreateRequestRecord).filter_by(session_id=sid).one_or_none()
            return None if row is None else dict(user_id=row.user_id, request_id=row.request_id)

    def begin(self, uid, rid, case_id, sid):
        with self.database.session_scope() as db:
            db.add(SessionCreateRequestRecord(user_id=uid, request_id=rid, case_id=case_id,
                session_id=sid, status='initializing'))
            db.add(CoachSession(session_id=sid, user_id=uid, case_id=case_id,
                phase='prepare', status='initializing', turn_count=0))

    def complete(self, uid, rid, result):
        with self.database.session_scope() as db:
            row = db.query(SessionCreateRequestRecord).filter_by(user_id=uid, request_id=rid).with_for_update().one()
            session = db.query(CoachSession).filter_by(session_id=row.session_id).with_for_update().one()
            if row.status != 'initializing' or session.status != 'initializing':
                raise MessageConflict('creation_state_changed')
            row.status, row.result, row.error_code = 'completed', copy.deepcopy(result), None
            session.status, session.phase = 'active', 'roleplay'

    def fail(self, uid, rid):
        with self.database.session_scope() as db:
            row = db.query(SessionCreateRequestRecord).filter_by(user_id=uid, request_id=rid).with_for_update().one()
            if row.status == 'initializing':
                session = db.query(CoachSession).filter_by(session_id=row.session_id).with_for_update().one()
                if session.status == 'aborted':
                    row.status, row.error_code = 'aborted', 'operator_aborted'
                elif session.status == 'initializing':
                    row.status, row.error_code = 'failed', 'session_initialization_failed'
                    session.status, session.phase = 'failed', 'prepare'


class MemoryCreations:
    """Explicit test substitutes only; production always uses the SQL ledger."""
    def __init__(self):
        self.records = {}
        self._locks = MemoryRequests()

    def lock(self, key):
        return self._locks.lock(key)

    def get(self, uid, rid):
        return copy.deepcopy(self.records.get((uid, rid)))

    def for_session(self, sid):
        return next((dict(user_id=uid, request_id=rid) for (uid, rid), row in self.records.items()
                     if row['session_id'] == sid), None)

    def begin(self, uid, rid, case_id, sid):
        self.records[(uid, rid)] = dict(case_id=case_id, session_id=sid,
            status='initializing', result=None, error_code=None)

    def complete(self, uid, rid, result):
        if self.records[(uid, rid)]['status'] != 'initializing':
            raise MessageConflict('creation_state_changed')
        self.records[(uid, rid)].update(status='completed', result=copy.deepcopy(result))

    def abort(self, sid):
        for row in self.records.values():
            if row['session_id'] == sid and row['status'] == 'initializing':
                row.update(status='aborted', error_code='operator_aborted')

    def fail(self, uid, rid):
        if self.records[(uid, rid)]['status'] == 'initializing':
            self.records[(uid, rid)].update(status='failed', error_code='session_initialization_failed')
