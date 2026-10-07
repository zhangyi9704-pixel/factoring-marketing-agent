import copy
from concurrent.futures import ThreadPoolExecutor
import threading
import pytest
from app.config import Settings
from app.evaluation.contracts import score_with_validation, verified_evidence
from app.evaluation.common import build_scoring_prompt, build_dim_specs
from app.evaluation.evidence import EvidenceExtractor
from app.evaluation.sales_score import SalesScorer
from app.services.user_score_service import UserScoreService
from tests.test_evaluation_service import make_service, SID, FakeLLM, CONVERSATION

CONV = [dict(role='sales', turn=1, content='需要多少融资资金？'),
        dict(role='customer', turn=1, content='需要100万元。')]
EVIDENCE = {'financing_need': [dict(turn=1, quote_sales=CONV[0]['content'], quote_customer=CONV[1]['content'])]}


class Sequence:
    def __init__(self, values):
        self.values = iter(values)
        self.calls = 0
    def chat_json(self, *args, **kwargs):
        self.calls += 1
        return next(self.values)


def payload(score=80, refs=None):
    return {'dimensions': {'financing_need': dict(score=score, rationale='伪造的隐藏事实', evidence=refs)}}


@pytest.mark.parametrize('bad', ['{broken', payload(101), payload(-1), payload(True),
    {'dimensions': {'unknown': {'score': 80}}}, payload(float('nan'))])
def test_invalid_json_schema_score_has_one_repair_then_fallback(bad):
    llm = Sequence([bad, bad])
    scores, evidence, errors = score_with_validation(llm, '', ['financing_need'], CONV, EVIDENCE)
    assert llm.calls == 2 and scores['financing_need']['score'] == 0
    assert 'deterministic_fallback' in errors


def test_schema_repaired_once_then_uses_valid_output():
    llm = Sequence(['broken', payload()])
    scores, _, errors = score_with_validation(llm, '', ['financing_need'], CONV, EVIDENCE)
    assert llm.calls == 2 and scores['financing_need']['score'] == 80
    assert 'schema_repaired' in errors and 'deterministic_fallback' not in errors


@pytest.mark.parametrize('ref', [dict(turn_id=999, original_text=CONV[0]['content']),
    dict(turn_id=1, original_text='不存在的承诺'),
    dict(turn_id=1, original_text=CONV[1]['content'], speaker='customer')])
def test_fake_or_customer_citation_removed(ref):
    scores, evidence, errors = score_with_validation(Sequence([payload(refs=[ref])]), '',
        ['financing_need'], CONV, EVIDENCE)
    assert evidence['financing_need'] == [] and errors
    assert '伪造' not in scores['financing_need']['rationale']


def test_customer_only_information_is_not_sales_evidence():
    conv = [dict(role='customer', turn=0, content='合同回款融资应收已验收。')]
    result = EvidenceExtractor(Settings(_env_file=None)).extract(conv)
    assert not any(result.values())
    assert verified_evidence(EVIDENCE, []) == {'financing_need': []}


def test_prompt_exposes_only_verified_per_dimension_whitelist():
    captured = []
    class Model:
        def chat_json(self, prompt, **kwargs):
            captured.append(prompt)
            return payload(refs=[dict(turn_id=1, original_text=CONV[0]['content'], speaker='sales')])
    source = copy.deepcopy(EVIDENCE)
    source['financing_need'].append(dict(turn=99, quote_sales='UNVERIFIED-QUOTE', quote_customer=''))
    _, verified, errors = score_with_validation(Model(), 'prompt', ['financing_need'], CONV, source)
    assert '各维度已验证销售证据白名单' in captured[0]
    assert CONV[0]['content'] in captured[0] and 'UNVERIFIED-QUOTE' not in captured[0]
    assert verified == EVIDENCE and not errors


def test_customer_volunteers_risk_and_need_does_not_credit_sales_skills():
    conv = [dict(role='sales', turn=1, content='您好。'),
        dict(role='customer', turn=1, content='融资缺口很大，合同账期长，对账验收逾期。')]
    result = EvidenceExtractor(Settings(_env_file=None)).extract(conv)
    assert result['requirement_discovery'] == []
    assert result['business_understanding'] == [] and result['risk_awareness'] == []
    assert result['financing_need']  # 机会事实可用对话中的客户佐证，能力须看销售表现


def test_hidden_case_never_in_scoring_context_or_rationale():
    prompt = build_scoring_prompt(task_name='评分', task_guidance='依据对话',
        dims=build_dim_specs({'financing_need': 15}), conversation=CONV,
        case=dict(industry='医药', hidden_risks=['SECRET-HOSPITAL-831'], unknown_info={'秘密': '999万'}))
    assert 'SECRET-HOSPITAL' not in prompt and '999万' not in prompt
    scores, _, _ = score_with_validation(Sequence([payload()]), prompt, ['financing_need'], CONV, EVIDENCE)
    assert '伪造的隐藏事实' not in str(scores)


def test_warn_deduplicated_five_points_and_block_zero():
    evidence = EvidenceExtractor(Settings(_env_file=None)).extract(CONVERSATION)
    warn = dict(turn=1, rule='CMP-01', raw_text='warn', action='log', details={'severity':'warn'})
    scorer = SalesScorer(FakeLLM(), Settings(_env_file=None))
    result = scorer.score(CONVERSATION, {'_compliance_events':[warn, copy.deepcopy(warn)]}, evidence)
    assert result['dimension_scores']['compliance']['score'] == 85
    result = scorer.score(CONVERSATION, {'_compliance_events':[dict(action='block')]}, evidence)
    assert result['dimension_scores']['compliance']['score'] == 0


def test_concurrent_services_share_one_evaluation_and_model_calls():
    service, factory = make_service()
    second, _ = make_service(factory=factory, llm=service.llm)
    with ThreadPoolExecutor(max_workers=4) as pool:
        reports = list(pool.map(lambda i: (service if i%2 else second).get(SID), range(8)))
    assert len(factory.evals) == 1 and sum(role == 'reasoning' for role, _ in service.llm.calls) == 2
    assert all(r == reports[0] for r in reports)


def test_sse_disconnect_reconnect_and_fresh_service_reuse_snapshot():
    entered, release = threading.Event(), threading.Event()
    class Slow(FakeLLM):
        def chat_json(self, *args, **kwargs):
            entered.set()
            assert release.wait(5)
            return super().chat_json(*args, **kwargs)
    llm = Slow()
    service, factory = make_service(llm)
    stream = service.run_stream(SID)
    assert next(stream)['type'] == 'stage'
    stream.close()
    assert entered.wait(5)
    release.set()
    reconnected = list(service.run_stream(SID))[-1]['report']
    assert len(factory.evals) == 1 and sum(role == 'reasoning' for role, _ in llm.calls) == 2
    fresh, _ = make_service(factory=factory, llm=Sequence([]))
    assert fresh.get(SID) == reconnected


def test_saved_source_and_training_snapshot_not_rebuilt():
    service, factory = make_service()
    source = dict(doc_id='KB-PR-002', version='v1', source='制度', industry='all', status='active', content='核查交易付款与融资信息。')
    service.knowledge_provider = lambda _: [source]
    first = service.get(SID)
    source['version'] = 'v2'
    service.knowledge_provider = lambda _: []
    assert service.get(SID) == first
    assert first['sources'][0]['version'] == 'v1' and first['coaching']
    assert first['coaching'][0]['original_text'] in [m['content'] for m in CONVERSATION if m['role']=='sales']


def test_dashboard_reads_persisted_results_with_all_model_paths_forbidden(monkeypatch):
    from app.services.llm_service import LLMService
    from tests.test_user_score_service import FakeUserSessionService
    from app.evaluation.service import EvaluationService
    service, factory = make_service()
    service.get(SID)
    def forbidden(*args, **kwargs):
        raise AssertionError('dashboard invoked model/evaluation')
    for name in ('chat', 'chat_json', 'chat_json_once'):
        monkeypatch.setattr(LLMService, name, forbidden)
    monkeypatch.setattr(EvaluationService, 'get', forbidden)
    board = UserScoreService(session_service=FakeUserSessionService(
        [dict(session_id=SID, user_id=1, case_id='INFRA_001', status='finished')],
        {SID: factory.evals[0]}))
    assert board.scores(1)['evaluated_sessions'] == 1
    assert list(board.scores_stream(1))[-1]['type'] == 'done'
