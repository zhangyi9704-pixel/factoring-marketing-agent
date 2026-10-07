"""Offline SQL + real coach_graph: replay, failure convergence, and safe reconciliation."""
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import threading
import pytest
from sqlalchemy import create_engine, BigInteger, Integer, select, func
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool
from langgraph.checkpoint.memory import InMemorySaver

from app.models import Base, CoachSession, MessageRequestRecord, SessionCreateRequestRecord, BusinessAudit, User
from app.config import Settings
from app.agents.roleplay.agent import RoleplayAgent, RoleplayAgentError
from app.services.session_service import SessionService
from app.services.message_requests import DatabaseRequests, DatabaseCreations, MemoryRequests, MessageConflict
from app.guardrail.detector import GuardrailService
from app.db.session_migration import migrate_session_requests
from tests.test_roleplay_nodes import FakeLLM, FakeCaseService, FakeRetriever
from tests.test_case_schema import make_case


@pytest.fixture
def env(monkeypatch):
    for table in Base.metadata.tables.values():
        if 'id' in table.c and isinstance(table.c.id.type, BigInteger):
            monkeypatch.setattr(table.c.id, 'type', BigInteger().with_variant(Integer(), 'sqlite'))
    engine = create_engine('sqlite://', connect_args={'check_same_thread': False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    @contextmanager
    def factory():
        with Session(engine, expire_on_commit=False) as db:
            with db.begin():
                yield db
    class Database:
        session_scope = staticmethod(factory)
    database = Database()
    database.engine = engine
    cfg, saver, locks = Settings(_env_file=None), InMemorySaver(), MemoryRequests()
    service = SessionService(cfg, session_factory=factory)
    requests = DatabaseRequests(database, lock_provider=locks.lock)
    creations = DatabaseCreations(database, lock_provider=locks.lock)
    def build(llm=None, cases=None):
        return RoleplayAgent(cfg, llm=llm or FakeLLM(), case_service=cases or FakeCaseService(make_case()),
            guardrail=GuardrailService(cfg), session_service=service, retriever=FakeRetriever(),
            checkpointer=saver, request_store=requests, creation_store=creations)
    yield build, factory, requests, creations, service, engine
    engine.dispose()


def test_creation_replay_survives_fresh_agent_and_isolates_owners(env):
    build, factory, _, creations, _, _ = env
    first = build()
    created = first.start_session('INFRA_001', 1, request_id='create')
    fresh = build()
    assert fresh.start_session('INFRA_001', 1, request_id='create') == created
    assert not fresh.llm.json_prompts and not fresh.llm.chat_prompts
    with pytest.raises(RoleplayAgentError, match='creation_request_id_conflict'):
        fresh.start_session('PHARMA_001', 1, request_id='create')
    second = fresh.start_session('INFRA_001', 2, request_id='create')
    assert second['session_id'] != created['session_id']
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(CoachSession)) == 2
        assert db.scalar(select(func.count()).select_from(SessionCreateRequestRecord)) == 2
    assert creations.get(1, 'create')['status'] == 'completed'


def test_competing_creations_do_not_repeat_model_calls(env):
    build, _, _, _, _, _ = env
    entered, release = threading.Event(), threading.Event()
    llm = FakeLLM()
    original = llm.chat_json
    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    llm.chat_json = slow
    first, second = build(llm), build()
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(first.start_session, 'INFRA_001', 1, request_id='concurrent')
        assert entered.wait(5)
        with pytest.raises(RoleplayAgentError, match='creation_busy'):
            second.start_session('INFRA_001', 1, request_id='concurrent')
        release.set()
        created = pending.result()
    assert second.start_session('INFRA_001', 1, request_id='concurrent') == created
    assert second.llm.json_prompts == []


def test_abort_cannot_interleave_with_live_initialization(env):
    build, _, _, creations, service, _ = env
    entered, release = threading.Event(), threading.Event()
    llm = FakeLLM()
    original = llm.chat_json
    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    llm.chat_json = slow
    agent, operator = build(llm), build()
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(agent.start_session, 'INFRA_001', 1, request_id='live-init')
        assert entered.wait(5)
        sid = creations.get(1, 'live-init')['session_id']
        try:
            with pytest.raises(RoleplayAgentError, match='session_busy'):
                operator.reconcile_session(sid, 'manual', actor_id=1, action='abort', reason='stop')
            assert service.get_session(sid)['status'] == 'initializing'
            assert creations.get(1, 'live-init')['status'] == 'initializing'
        finally:
            release.set()
        assert pending.result()['session_id'] == sid
    assert service.get_session(sid)['status'] == 'active'


def test_abandoned_initialization_can_abort_and_never_revive(env):
    build, _, _, creations, service, _ = env
    sid = 'abandoned-initialization'
    creations.begin(1, 'abandoned', 'INFRA_001', sid)
    result = build().reconcile_session(sid, 'manual', actor_id=1, action='abort', reason='cannot verify')
    assert result['status']['finish_reason'] == 'operator_aborted'
    assert creations.get(1, 'abandoned')['status'] == 'aborted'
    with pytest.raises(MessageConflict, match='creation_state_changed'):
        creations.complete(1, 'abandoned', {'session_id': sid})
    creations.fail(1, 'abandoned')
    assert service.get_session(sid)['status'] == 'aborted'
    assert creations.get(1, 'abandoned')['status'] == 'aborted'
    with pytest.raises(RoleplayAgentError, match='session_initialization_aborted'):
        build().start_session('INFRA_001', 1, request_id='abandoned')


def test_creation_commit_checks_session_terminal_state_before_stale_failure(env):
    _, factory, _, creations, service, _ = env
    sid = 'stale-creation-commit'
    creations.begin(1, 'stale', 'INFRA_001', sid)
    with factory() as db:
        db.query(CoachSession).filter_by(session_id=sid).one().status = 'aborted'
    with pytest.raises(MessageConflict, match='creation_state_changed'):
        creations.complete(1, 'stale', {'session_id': sid})
    creations.fail(1, 'stale')
    assert service.get_session(sid)['status'] == 'aborted'
    assert creations.get(1, 'stale')['status'] == 'aborted'


def test_case_preflight_has_no_persistence_or_model_side_effect(env):
    build, factory, _, _, _, _ = env
    agent = build(cases=FakeCaseService(None))
    with pytest.raises(RoleplayAgentError, match='案例不存在'):
        agent.start_session('NOPE', 1, request_id='invalid')
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(CoachSession)) == 0
        assert db.scalar(select(func.count()).select_from(SessionCreateRequestRecord)) == 0
    assert agent.llm.chat_prompts == agent.llm.json_prompts == []


def test_initialization_failure_is_terminal_and_retry_never_replays(env):
    build, _, _, creations, service, _ = env
    agent = build()
    class Broken:
        def invoke(self, *args):
            raise OSError('private infrastructure failure')
    agent._graph = Broken()
    with pytest.raises(RoleplayAgentError, match='session_initialization_failed'):
        agent.start_session('INFRA_001', 1, request_id='failed-create')
    record = creations.get(1, 'failed-create')
    assert record['status'] == service.get_session(record['session_id'])['status'] == 'failed'
    fresh = build()
    with pytest.raises(RoleplayAgentError, match='session_initialization_failed'):
        fresh.start_session('INFRA_001', 1, request_id='failed-create')
    assert fresh.llm.json_prompts == []


def test_crash_after_initialization_reconciles_creation_without_graph_replay(env):
    build, _, _, creations, service, _ = env
    agent = build()
    sid = 'initialization_ack_lost'
    creations.begin(1, 'crashed-create', 'INFRA_001', sid)
    agent.graph.invoke(dict(session_id=sid, user_id=1, case_id='INFRA_001'), agent._config(sid))
    assert service.get_session(sid)['status'] == 'initializing'
    fresh = build()
    result = fresh.start_session('INFRA_001', 1, request_id='crashed-create')
    assert result['session_id'] == sid and service.get_session(sid)['status'] == 'active'
    assert fresh.llm.json_prompts == fresh.llm.chat_prompts == []


def failed_completion(env, monkeypatch):
    build, _, requests, _, _, _ = env
    agent = build()
    sid = agent.start_session('INFRA_001', 1, request_id='create-for-recovery')['session_id']
    original = requests.complete
    def failed(*args):
        raise OSError('unit commit failure')
    monkeypatch.setattr(requests, 'complete', failed)
    with pytest.raises(RoleplayAgentError, match='message_execution_failed'):
        agent.send_message(sid, '账期多久？', request_id='uncertain', expected_turn=0)
    monkeypatch.setattr(requests, 'complete', original)
    return agent, sid


def test_verified_ledger_failure_can_complete_and_continue_without_model_replay(env, monkeypatch):
    agent, sid = failed_completion(env, monkeypatch)
    before = len(agent.llm.json_prompts)
    assert agent.get_status(sid)['interrupted']
    result = agent.reconcile_session(sid, 'uncertain', actor_id=1, reason='verify result')
    assert result['action'] == 'reconciled' and not result['status']['interrupted']
    assert result['result']['turn'] == 1 and len(agent.llm.json_prompts) == before
    assert agent.send_message(sid, '账期多久？', request_id='uncertain', expected_turn=0) == result['result']
    assert agent.send_message(sid, '继续核实', request_id='next', expected_turn=1)['turn'] == 2
    with env[1]() as db:
        assert db.scalar(select(func.count()).select_from(BusinessAudit).where(BusinessAudit.action == 'message_reconciled')) == 1


def test_checkpoint_mismatch_requires_explicit_abort_and_preserves_records(env, monkeypatch):
    agent, sid = failed_completion(env, monkeypatch)
    real = agent.graph.get_state
    def forged(config):
        snapshot = real(config)
        return snapshot._replace(values=dict(snapshot.values, turn=0))
    monkeypatch.setattr(agent.graph, 'get_state', forged)
    with pytest.raises(RoleplayAgentError, match='persistence_mismatch'):
        agent.reconcile_session(sid, 'uncertain', actor_id=1, reason='verify result')
    result = agent.reconcile_session(sid, 'uncertain', actor_id=1, action='abort', reason='cannot verify')
    assert result['status']['finished'] and result['status']['finish_reason'] == 'operator_aborted'
    assert len(result['status']['messages']) == 3
    assert env[2].get(sid, 'uncertain')['status'] == 'aborted'
    with pytest.raises(RoleplayAgentError, match='session_not_active'):
        agent.send_message(sid, '不能继续', request_id='new', expected_turn=1)


def test_claim_identity_mismatch_cannot_be_reconciled(env, monkeypatch):
    agent, sid = failed_completion(env, monkeypatch)
    agent.graph.update_state(agent._config(sid), {'active_payload_hash': 'different'})
    with pytest.raises(RoleplayAgentError, match='reconciliation_requires_abort'):
        agent.reconcile_session(sid, 'uncertain', actor_id=1, reason='verify')
    assert env[2].get(sid, 'uncertain')['status'] == 'failed'


def test_completed_acknowledgement_loss_is_not_downgraded(env, monkeypatch):
    agent = env[0]()
    requests = env[2]
    sid = agent.start_session('INFRA_001', 1, request_id='lost-message-ack')['session_id']
    original = requests.complete
    def lost(*args):
        original(*args)
        raise OSError('acknowledgement lost')
    monkeypatch.setattr(requests, 'complete', lost)
    with pytest.raises(RoleplayAgentError):
        agent.send_message(sid, '账期多久？', request_id='lost-ack', expected_turn=0)
    assert requests.get(sid, 'lost-ack')['status'] == 'completed'
    assert not agent.get_status(sid)['interrupted']


def test_recovery_endpoint_checks_stored_owner_and_admin(env):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.api.deps import get_auth_service, get_ownership_service
    from app.api.v1.session import get_roleplay_agent
    from app.services.auth_service import AuthService, Principal, issue_token
    build, factory, _, _, service, _ = env
    agent = build()
    sid = agent.start_session('INFRA_001', 1, request_id='auth-recovery')['session_id']
    with factory() as db:
        for uid, role in ((1, 'sales'), (2, 'sales'), (3, 'coach'), (4, 'admin')):
            db.add(User(id=uid, username='unit'+str(uid), role=role, enabled=True, password_hash='unused', salt='unused'))
    saved = app.dependency_overrides.copy()
    app.dependency_overrides[get_auth_service] = lambda: AuthService(factory)
    app.dependency_overrides[get_ownership_service] = lambda: service
    app.dependency_overrides[get_roleplay_agent] = lambda: agent
    try:
        client = TestClient(app)
        path, payload = f'/api/v1/sessions/{sid}/reconcile', dict(action='abort', request_id='manual', reason='explicit abort')
        assert client.post(path, json=payload).status_code == 401
        for uid, role in ((2, 'sales'), (3, 'coach')):
            assert client.post(path, json=payload, headers={'Authorization': 'Bearer '+issue_token(Principal(uid, 'unused', role))}).status_code == 403
        assert client.post(path, json=payload, headers={'Authorization': 'Bearer '+issue_token(Principal(4, 'unused', 'admin'))}).status_code == 200
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(saved)


def test_creation_ledger_migration_is_additive_and_repeatable():
    engine = create_engine('sqlite://')
    try:
        assert migrate_session_requests(engine) == ['t_session_create_request']
        assert migrate_session_requests(engine) == []
    finally:
        engine.dispose()
