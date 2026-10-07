"""恢复及持久请求语义，图始终是既有coach_graph。"""
import threading
from concurrent.futures import ThreadPoolExecutor
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from app.agents.roleplay.agent import RoleplayAgent, RoleplayAgentError
from app.services.message_requests import MemoryRequests
from app.schemas.session import SessionStatusResponse, TurnResponse, SessionCreatedResponse
from tests.test_roleplay_agent import make_agent


def test_duplicate_id_replays_done_without_model_or_counter_increment():
    agent, recorder = make_agent()
    sid = agent.start_session('INFRA_001', 1001)['session_id']
    first = agent.send_message(sid, '账期多久？', request_id='same', expected_turn=0)
    before = len(recorder.messages)
    assert agent.send_message(sid, '账期多久？', request_id='same', expected_turn=0) == first
    assert len(recorder.messages) == before and agent.get_status(sid)['turn'] == 1
    with pytest.raises(RoleplayAgentError, match='request_id_conflict'):
        agent.send_message(sid, '不同正文', request_id='same', expected_turn=0)


def test_different_ids_stale_turn_does_not_add_second_round():
    agent, _ = make_agent()
    sid = agent.start_session('INFRA_001', 1001)['session_id']
    agent.send_message(sid, '账期多久？', request_id='one', expected_turn=0)
    with pytest.raises(RoleplayAgentError, match='stale_turn'):
        agent.send_message(sid, '账期多久？', request_id='two', expected_turn=0)
    assert agent.get_status(sid)['turn'] == 1


def test_concurrent_same_id_and_disconnect_only_one_effective_turn():
    agent, recorder = make_agent()
    sid = agent.start_session('INFRA_001', 1001)['session_id']
    entered, release = threading.Event(), threading.Event()
    original = agent.llm.chat_json
    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    agent.llm.chat_json = slow
    with ThreadPoolExecutor(max_workers=2) as pool:
        one = pool.submit(agent.send_message, sid, '账期多久？', request_id='shared', expected_turn=0)
        assert entered.wait(5)
        two = pool.submit(agent.send_message, sid, '账期多久？', request_id='shared', expected_turn=0)
        with pytest.raises(RoleplayAgentError, match='session_busy'):
            agent.send_message(sid, '另一个输入', request_id='other', expected_turn=0)
        release.set()
        assert one.result() == two.result()
    assert agent.get_status(sid)['turn'] == 1 and len(recorder.messages) == 3


def test_failed_request_is_not_replayed_even_with_new_id():
    agent, _ = make_agent()
    sid = agent.start_session('INFRA_001', 1001)['session_id']
    class Broken:
        def stream(self, *a, **kw):
            raise RuntimeError('failure')
            yield
        def get_state(self, config):
            return saved
    saved = agent.graph.get_state(agent._config(sid))
    agent._graph = Broken()
    with pytest.raises(RoleplayAgentError, match='message_execution_failed'):
        agent.send_message(sid, '账期多久？', request_id='failed', expected_turn=0)
    with pytest.raises(RoleplayAgentError, match='request_failed'):
        agent.send_message(sid, '账期多久？', request_id='failed', expected_turn=0)
    with pytest.raises(RoleplayAgentError, match='session_interrupted'):
        agent.send_message(sid, '账期多久？', request_id='new', expected_turn=0)


def test_two_sessions_keep_checkpoint_state_separate_and_binding_verified():
    agent, _ = make_agent()
    first = agent.start_session('INFRA_001', 1001)['session_id']
    second = agent.start_session('INFRA_001', 1001)['session_id']
    agent.send_message(first, '账期多久？', request_id='q', expected_turn=0)
    before = agent.graph.get_state(agent._config(first))
    assert agent.get_status(second)['turn'] == 0
    assert before.next == ('sales_input',) and before.tasks[0].interrupts
    agent._bindings[first]['user_id'] = 999
    with pytest.raises(RoleplayAgentError, match='binding_mismatch'):
        agent.get_status(first)


def test_fresh_agent_reuses_all_checkpoint_fields_and_interrupt():
    agent, recorder = make_agent()
    sid = agent.start_session('INFRA_001', 1001)['session_id']
    agent.send_message(sid, '账期多久？', request_id='q', expected_turn=0)
    values = dict(unlocked_ids=['FC1'], dim_asked_count={'账期': 1}, pending_directives=['FC2'], guardrail_hit_count=2)
    agent.graph.update_state(agent._config(sid), values)
    # 更新节点仍为同一图，不从聊天文本重建；这里仅测试独立门面读取saver。
    fresh = RoleplayAgent(agent.settings, llm=agent.llm, case_service=agent.case_service,
        guardrail=agent.guardrail, session_service=recorder, retriever=agent._retriever,
        checkpointer=agent.checkpointer, request_store=agent.requests)
    fresh._bindings = dict(agent._bindings)
    snapshot = fresh._snapshot(sid)
    for key, expected in values.items():
        assert snapshot.values[key] == expected
    assert fresh._config(sid)['configurable']['thread_id'] == sid
    assert fresh.get_status(sid)['recovery_degraded'] is True


def test_public_schemas_strip_internal_and_nested_message_fields():
    status = SessionStatusResponse(session_id='x', hidden_risks=['secret'], unlocked_ids=['hidden'],
        messages=[dict(role='customer', turn=0, content='您好', pending_directives=['hidden'])]).model_dump()
    assert 'hidden' not in str(status)
    turn = TurnResponse(turn=1, checkpoint={'secret': 1}, guardrail_events=[dict(rule='CMP-01',severity='warn', action='log', compliant_response='核实',snippet='hidden')]).model_dump()
    assert 'hidden' not in str(turn) and 'checkpoint' not in turn
    created = SessionCreatedResponse(session_id='x',case_id='c',user_id=1,
        opening=dict(role='customer',turn=0,content='您好',released_info=['hidden'])).model_dump()
    assert 'hidden' not in str(created)


def test_memory_checkpoint_requires_explicit_development_backend():
    from app.graphs.coach_graph import build_redis_checkpointer
    from app.config import Settings
    settings = Settings(_env_file=None)
    settings.langgraph.checkpoint.backend = 'memory'
    assert isinstance(build_redis_checkpointer(settings), InMemorySaver)
    settings.langgraph.checkpoint.backend = 'invalid'
    with pytest.raises(RuntimeError, match='persistent_checkpoint_required'):
        build_redis_checkpointer(settings)


def test_output_compliance_rewrite_precedes_visible_delta(monkeypatch):
    from app.agents.roleplay import customer_reply as module
    from tests.test_roleplay_nodes import FakeLLM
    events = []
    monkeypatch.setattr(module, '_stream_writer', lambda: events.append)
    class OutputGuard:
        enabled = True
        def protect_tree(self, value, **kwargs):
            assert events == []
            value['reply'] = '请先核实资料，再提交独立审批。'
    node = module.make_customer_reply_node(FakeLLM(replies=[{'reply': '保证当天放款'}]), guardrail=OutputGuard())
    result = node(dict(session_id='unit', case={}, conversation=[], sales_input='您好', release_whitelist=[]))
    assert events == [{'kind':'customer_delta','text':result['customer_reply']}]
    assert '保证当天' not in str(events)


def test_compliance_safe_fallback_does_not_release_or_consume_pending(monkeypatch):
    from types import SimpleNamespace
    from app.agents.roleplay import customer_reply as module
    from tests.test_roleplay_nodes import FakeLLM
    monkeypatch.setattr(module, '_stream_writer', lambda: None)
    class OutputGuard:
        enabled = True
        def protect_tree(self, value, **kwargs):
            value['reply'] = '销售模板不应作为客户事实'
            return SimpleNamespace(hits=['CMP-01'], l2_degraded=False)
    node = module.make_customer_reply_node(FakeLLM(replies=[dict(reply='保证当天放款',revealed=['FC1'])]), guardrail=OutputGuard())
    result = node(dict(session_id='unit',case={},conversation=[],sales_input='您好',release_whitelist=['FC1']))
    assert result['customer_reply'] == module.SAFE_REPLY and result['revealed_this_turn'] == []
