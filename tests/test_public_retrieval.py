"""Public cache is shared across request agents, with revision and owner boundaries."""
from app.agents.roleplay.agent import RoleplayAgent
from app.agents.coach.briefing import make_briefing_node
from app.config import Settings
from app.services.public_retrieval import get_public_retriever
from tests.test_retrieval_branches import build, row
from tests.test_briefing_node import base_state
from tests.test_roleplay_nodes import FakeLLM, FakeSessionService


def test_request_agents_share_a_lazy_public_retriever_but_not_other_settings(monkeypatch):
    constructed = []
    inner = build([row()])
    def factory(settings):
        constructed.append(settings)
        return inner
    monkeypatch.setattr('app.rag.retriever.HybridRetriever', factory)
    settings = Settings(_env_file=None)
    first, second = RoleplayAgent(settings), RoleplayAgent(settings)
    assert first.retriever is second.retriever
    assert not constructed
    first.retriever.retrieve('付款人确权')
    second.retriever.retrieve('付款人确权')
    assert constructed == [settings]
    assert RoleplayAgent(Settings(_env_file=None)).retriever is not first.retriever


def test_shared_public_cache_rechecks_revision_catalog_and_returns_independent_copies(monkeypatch):
    inner = build([row()])
    inner.settings.rag.retrieval.cache_ttl = 60
    current = {'revision': 'one', 'catalog': 'one'}
    class Catalog:
        def catalog(self): return {}, {}, current['catalog']
        def expression(self, catalog): return 'status == "active"'
        def filter(self, items, catalog): return items
    inner.visibility = Catalog()
    monkeypatch.setattr('app.rag.retriever.revision', lambda: current['revision'])
    monkeypatch.setattr('app.rag.retriever.HybridRetriever', lambda settings: inner)
    shared = get_public_retriever(inner.settings)
    first = shared.retrieve('付款人确权')
    first.items[0].content = 'caller-local edit'
    assert shared.retrieve('付款人确权').items[0].content == '可核实的依据'
    assert len(inner.store.calls) == 1
    current['revision'] = 'two'
    shared.retrieve('付款人确权')
    assert len(inner.store.calls) == 2
    current['catalog'] = 'two'
    shared.retrieve('付款人确权')
    assert len(inner.store.calls) == 3


def test_cached_missing_knowledge_creates_distinct_briefing_pending_per_owner(monkeypatch):
    inner = build([])
    inner.settings.rag.retrieval.cache_ttl = 60
    monkeypatch.setattr('app.rag.retriever.HybridRetriever', lambda settings: inner)
    shared = get_public_retriever(inner.settings)
    class Recorder(FakeSessionService):
        def __init__(self):
            super().__init__()
            self.pending = []
        def register_escalation(self, kind, question, **kwargs):
            identifier = f'esc_{kwargs["user_id"]}_{kwargs["session_id"]}_{kind}'
            self.pending.append(dict(kind=kind, external_id=identifier, **kwargs))
            return identifier
    recorder = Recorder()
    responses = []
    for uid in (1, 2):
        node = make_briefing_node(FakeLLM(), settings=inner.settings, retriever=shared, session_service=recorder)
        responses.append(node(base_state(user_id=uid, session_id=f'session_{uid}'))['briefing'])
    assert len(inner.store.calls) == 1
    missing = [record for record in recorder.pending if record['kind'] == 'missing']
    assert {record['user_id'] for record in missing} == {1, 2}
    assert {record['session_id'] for record in missing} == {'session_1', 'session_2'}
    assert responses[0]['escalation_id'] != responses[1]['escalation_id']
