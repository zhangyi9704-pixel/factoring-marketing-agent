"""Offline regression cases for delivered briefing/opening safety, not cloud mocks of quality."""
import copy
import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.coach.briefing import DEFAULT_GOALS, make_briefing_node, ground_briefing_raw
from app.agents.roleplay.case_loader import make_init_customer_state_node
from app.compliance.engine import ComplianceEngine
from app.config import Settings
from app.engine.leak_guard import SAFE_REPLY
from app.graphs.coach_graph import build_coach_graph
from app.rag.retriever import RetrievalItem, RetrievalResult
from app.schemas.case import CaseLeakError, assert_case_text_safe, to_public_case
from app.services.case_service import CaseService
from tests.test_roleplay_nodes import FakeCaseService, FakeSessionService


def case_data(case_id="INFRA_001"):
    return next(c for c in CaseService().load_yaml_cases() if c.case_id == case_id)


class Semantic:
    def __init__(self, fail=False, hits=None):
        self.fail, self.hits, self.calls = fail, hits or {}, []

    def chat_json(self, prompt, **kwargs):
        text = json.loads(prompt)["text"]
        self.calls.append(text)
        if self.fail:
            raise RuntimeError("provider-body-must-not-be-exposed")
        return {"decisions": [dict(rule_id=f"CMP-{i:02}",
            hit=bool(self.hits.get(f"CMP-{i:02}")) and self.hits[f"CMP-{i:02}"] in text,
            evidence=self.hits.get(f"CMP-{i:02}", "") if self.hits.get(f"CMP-{i:02}", "") in text else "")
            for i in range(1, 10)]}


class Recorder(FakeSessionService):
    def __init__(self):
        super().__init__()
        self.review, self.pending = [], []

    def mark_human_review(self, sid):
        self.review.append(sid)

    def register_escalation(self, kind, question, **kwargs):
        self.pending.append((kind, question, kwargs))
        return "review-output-audit"

    def record_compliance(self, sid, turn, rule, raw, *, action="log", details=None):
        self.compliance.append((turn, rule, raw, action, details))


class Retriever:
    def retrieve(self, *args, **kwargs):
        return RetrievalResult("query", [RetrievalItem("c", "p", "child", "product", "doc.md", 2,
            "需要核实回款账户与应收质量。", .99, doc_id="KB-PD-006", version="v1")])


class BriefLLM:
    def __init__(self, output):
        self.output = output

    def chat_json(self, *args, **kwargs):
        return copy.deepcopy(self.output)


def run_briefing(raw, *, semantic=None, retriever=None, settings=None):
    settings = settings or Settings(_env_file=None)
    case, recorder = case_data(), Recorder()
    result = make_briefing_node(BriefLLM(raw), settings=settings, retriever=retriever or Retriever(),
        session_service=recorder, guardrail=ComplianceEngine(settings, llm=semantic or Semantic()))(
        dict(session_id="offline-output", user_id=1, case=case.model_dump(), public_case=to_public_case(case).model_dump()))
    return result, recorder


@pytest.mark.parametrize("amount", ["3000000元", "300.0万元", "0.03亿元"])
def test_private_amount_equivalence_is_blocked_before_persistence(amount):
    case = case_data()
    private = next(c.fact for c in case.fact_cards if c.id == "INF-09")
    converted = private.replace("300万", amount)
    with pytest.raises(CaseLeakError, match="private_case_fingerprint"):
        assert_case_text_safe(case.model_dump(), converted)
    result, recorder = run_briefing({"visit_goals": [converted]})
    assert converted not in json.dumps(result["briefing"], ensure_ascii=False)
    assert result["briefing"]["visit_goals"] == DEFAULT_GOALS
    assert "private_case_blocked" in result["briefing"]["degradation_reasons"]
    assert recorder.briefings["offline-output"] == result["briefing"]


def test_briefing_semantic_outage_discards_advice_and_registers_review():
    result, recorder = run_briefing({"visit_goals": ["单据这事有空间"]}, semantic=Semantic(fail=True))
    assert result["briefing"]["visit_goals"] == DEFAULT_GOALS
    assert result["needs_human_review"] and recorder.review
    assert result["briefing"]["compliance_status"]["l2_degraded"]
    assert result["briefing"]["review_pending_id"] == "review-output-audit"
    assert recorder.pending[-1][2]["context"]["phase"] == "briefing_validation"
    assert "provider-body-must-not-be-exposed" not in json.dumps(result)


def test_unsupported_numeric_claim_and_forged_risk_quote_become_verification_guidance():
    raw = {"visit_goals": ["客户应收余额已确认为999000000元"], "risk_warnings": [
        {"risk": "客户已发生刑事涉诉", "basis": "案例公开资料", "source_id": "case", "evidence": "虚构的原文依据"}]}
    result, _ = run_briefing(raw)
    delivered = json.dumps(result["briefing"], ensure_ascii=False)
    assert "999000000" not in delivered and "客户已发生刑事涉诉" not in delivered
    assert "待核实" in delivered and "grounding_rejected" in result["briefing"]["degradation_reasons"]


def test_valid_risk_quote_is_bound_to_supplied_source_and_keeps_uncertainty():
    public = to_public_case(case_data()).model_dump()
    quote = public["pain_points"][0]
    raw, rejected = ground_briefing_raw(public, {"risk_warnings": [
        {"risk": "融资需求需匹配回款安排", "source_id": "case", "evidence": quote}]}, [])
    assert not rejected and raw["risk_warnings"][0]["basis"] == "case：" + quote
    assert raw["risk_warnings"][0]["risk"].startswith("待核实")
    raw, rejected = ground_briefing_raw(public, {"risk_warnings": [
        {"risk": "回款安排需核实", "source_id": "KB-PD-006", "evidence": "需要核实回款账户"}]},
        [{"doc_id": "KB-PD-006", "content": "需要核实回款账户与应收质量。"}])
    assert not rejected and "KB-PD-006" in raw["risk_warnings"][0]["basis"]


@pytest.mark.parametrize("bad_risks", [42, {"risk": "bad shape"}, "bad shape"])
def test_malformed_risk_field_does_not_break_safe_briefing(bad_risks):
    result, recorder = run_briefing({"risk_warnings": bad_risks})
    assert result["briefing"]["risk_warnings"] == []
    assert result["briefing"]["degraded"] and recorder.briefings


def test_expired_briefing_budget_does_not_start_generation_or_semantic_scan(monkeypatch):
    from app.services import external_calls
    clock = [0.0]
    monkeypatch.setattr(external_calls.time, "monotonic", lambda: clock[0])
    settings = Settings(_env_file=None)
    settings.agents.briefing.timeout_seconds = 1
    class SlowRetriever(Retriever):
        def retrieve(self, *args, **kwargs):
            clock[0] = 2
            return super().retrieve(*args, **kwargs)
    semantic = Semantic()
    result, recorder = run_briefing({}, semantic=semantic, retriever=SlowRetriever(), settings=settings)
    assert result["briefing"]["visit_goals"] == DEFAULT_GOALS
    assert semantic.calls == [] and result["needs_human_review"]
    assert recorder.briefings["offline-output"] == result["briefing"]


@pytest.mark.parametrize("fixed", [False, True])
def test_opening_gate_blocks_private_currency_and_unsupported_numbers(fixed):
    case = case_data()
    private = next(c.fact for c in case.fact_cards if c.id == "INF-09").replace("300万", "3000000元")
    if fixed:
        case.opening_line = private
    else:
        from tests.test_case_schema import make_case
        case = make_case()  # The supported compatibility branch uses legacy cases.
        case.hidden_risks = [private.replace("3000000元", "300万")]
    class OpeningLLM:
        def chat(self, *args, **kwargs):
            return private
    recorder, semantic = Recorder(), Semantic()
    result = make_init_customer_state_node(OpeningLLM(), recorder,
        guardrail=ComplianceEngine(Settings(_env_file=None), llm=semantic))(
        {"session_id": "open-audit", "user_id": 1, "case": case.model_dump()})
    assert result["conversation"][0]["content"] == SAFE_REPLY
    assert semantic.calls == []  # No private rejected data is forwarded to L2.
    assert result["needs_human_review"] and recorder.messages[0][2] == SAFE_REPLY


def test_opening_outage_falls_back_but_a_negative_semantic_customer_question_is_preserved():
    case = case_data("ECOM_001")
    question = "你们能保证额度吗？"
    case.opening_line = question
    node = make_init_customer_state_node(None, guardrail=ComplianceEngine(Settings(_env_file=None), llm=Semantic()))
    state = {"case": case.model_dump()}
    assert node(state)["conversation"][0]["content"] == question
    failed = make_init_customer_state_node(None, guardrail=ComplianceEngine(Settings(_env_file=None), llm=Semantic(fail=True)))(state)
    assert failed["conversation"][0]["content"] == SAFE_REPLY and failed["needs_human_review"]
    case.opening_line = "保证额度。你们能保证额度吗？"
    rejected = node({"case": case.model_dump()})
    assert rejected["conversation"][0]["content"] == SAFE_REPLY


def test_dynamic_opening_rejects_a_new_unsupported_amount():
    from tests.test_case_schema import make_case
    class OpeningLLM:
        def chat(self, *args, **kwargs):
            return "我们已确认应收余额为999000000元。"
    recorder = Recorder()
    result = make_init_customer_state_node(OpeningLLM(), recorder)(
        {"session_id": "new-amount", "user_id": 1, "case": make_case().model_dump()})
    assert result["conversation"][0]["content"] == SAFE_REPLY
    assert recorder.messages[0][2] == SAFE_REPLY and result["needs_human_review"]


def test_same_compliance_replacement_uses_one_template_check_plus_final_batch():
    semantic = Semantic(hits={"CMP-01": "保证当天放款"})
    value = {"reasons": ["保证当天放款"] * 10}
    result = ComplianceEngine(Settings(_env_file=None), llm=semantic).protect_tree(value)
    assert len(semantic.calls) == 3 and result.hits
    assert "保证当天放款" not in json.dumps(value, ensure_ascii=False)
    assert semantic.calls[-1] == "\n".join(value["reasons"])


def test_case_load_failure_routes_to_end_without_opening_or_cloud_calls():
    class NeverLLM:
        def chat(self, *args, **kwargs):
            raise AssertionError("must not generate opening")
        def chat_json(self, *args, **kwargs):
            raise AssertionError("must not generate briefing")
    recorder, settings = Recorder(), Settings(_env_file=None)
    graph = build_coach_graph(NeverLLM(), case_service=FakeCaseService(None), guardrail=ComplianceEngine(settings, llm=Semantic()),
        settings=settings, session_service=recorder, retriever=Retriever(), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "missing-case-audit"}}
    result = graph.invoke({"session_id": "missing-case-audit", "case_id": "MISSING", "user_id": 1}, config)
    assert result["finished"] and result["finish_reason"] == "error"
    assert not recorder.messages and not recorder.briefings and graph.get_state(config).next == ()
