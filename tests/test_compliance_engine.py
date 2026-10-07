"""语义结果独立注入；检验双命中、故障、证据伪造及完整改写。"""
import json
import pytest
from app.compliance.engine import ComplianceEngine
from app.config import Settings
from app.guardrail.rules import load_compliance_rules


class Semantic:
    def __init__(self, hits=None, fail=False):
        self.hits, self.fail = hits or {}, fail

    def chat_json(self, prompt, **kwargs):
        if self.fail:
            raise RuntimeError("test unavailable")
        text = json.loads(prompt)["text"]
        return {"decisions": [dict(rule_id=f"CMP-{i:02}", hit=self.hits.get(f"CMP-{i:02}", "") in text and bool(self.hits.get(f"CMP-{i:02}")),
            evidence=self.hits.get(f"CMP-{i:02}", "") if self.hits.get(f"CMP-{i:02}", "") in text else "") for i in range(1, 10)]}


def engine(hits=None, fail=False):
    return ComplianceEngine(Settings(_env_file=None), llm=Semantic(hits, fail))


@pytest.mark.parametrize("rule", [r for r in load_compliance_rules().rules if r.applies_to == "both"])
def test_hook_examples_match(rule):
    result = engine({rule.rule_id: rule.violation_example}).scan(rule.violation_example)
    hit = next(h for h in result.hits if h.rule == rule.rule_id)
    assert hit.severity == rule.severity
    assert result.l1_elapsed_ms >= 0 and hit.details["l2"]


def test_block_requires_double_hit_or_degraded():
    text = "计量没做完没关系，单据可以想办法"
    assert engine({"CMP-05": text}).scan(text).hits[0].severity == "block"
    assert engine().scan(text).hits[0].severity == "warn"
    degraded = engine(fail=True).scan(text)
    assert degraded.hits[0].severity == "block" and degraded.l2_degraded
    implicit = "单据这事有空间"
    hit = engine({"CMP-05": implicit}).scan(implicit).hits[0]
    assert hit.severity == "warn" and not hit.details["l1"]


@pytest.mark.parametrize("text", ["请先核实单据真实性，再提交审批。", "不能保证当天放款。", "无法承诺融资额度。",
    "资料齐备后提交独立审批，额度根据应收质量核定。", "当天需要核验资料。"])
def test_no_false_positive(text):
    assert not engine().scan(text).hits


def test_disclaimer_does_not_cancel_promise():
    assert engine().scan("保证当天放款，额度没问题，以审批结果为准。").hits


def test_output_removes_original_and_rechecks():
    text = "保证当天放款，先给你500万。"
    safe, result = engine({"CMP-01": "保证当天放款", "CMP-02": "先给你500万"}).protect(text)
    assert result.hits and "保证当天放款" not in safe and "500万" not in safe
    assert "审批" in safe and not engine().scan(safe).hits


@pytest.mark.parametrize("kind", ["unknown_rule", "fake_quote", "incomplete", "wrong_boolean"])
def test_invalid_model_schema_is_degraded_conservative_block(kind):
    class Invalid:
        def chat_json(self, *args, **kwargs):
            payload = Semantic().chat_json(json.dumps({"text": ""}))
            if kind == "unknown_rule":
                payload["decisions"][0]["rule_id"] = "CMP-99"
            elif kind == "fake_quote":
                payload["decisions"][0].update(hit=True, evidence="编造引用")
            elif kind == "wrong_boolean":
                payload["decisions"][0]["hit"] = "false"
            else:
                payload["decisions"].pop()
            return payload
    result = ComplianceEngine(Settings(_env_file=None), llm=Invalid()).scan("单据可以想办法")
    assert result.l2_degraded and result.hits[0].severity == "block"


def test_block_excluded_before_unlock_and_audit_sticky_review():
    from app.agents.roleplay.state_judge import make_state_judge_node
    from app.agents.roleplay.guardrail_node import make_guardrail_node, make_update_state_node
    from tests.test_roleplay_nodes import FakeSessionService
    class Recorder(FakeSessionService):
        def record_compliance(self, session_id, turn, rule, raw_text, *, action="log", details=None):
            self.compliance.append((turn, rule, raw_text, action, details))
        def mark_human_review(self, sid):
            self.marked = sid
    text = "付款方 SECRET84 单据可以想办法"
    detector, recorder = engine({"CMP-05": text}), Recorder()
    state = dict(session_id="unit", turn=1, sales_input=text, dim_asked_count={"付款人": 1},
                 conversation=[dict(role="customer", content="您好", turn=0)], unlocked_ids=["disclosed"],
                 pending_directives=["pending"], released_info=["disclosed"], guardrail_hit_count=0)
    original = json.dumps(state["conversation"], ensure_ascii=False)
    state.update(make_state_judge_node(detector)(state))
    assert state["input_blocked"] and state["sales_input"] == "" and state["dim_asked_count"] == {"付款人": 1}
    state.update(make_guardrail_node(detector, recorder)(state))
    state.update(make_update_state_node(session_service=recorder)(state))
    assert state["turn"] == 1 and json.dumps(state["conversation"], ensure_ascii=False) == original
    assert state["needs_human_review"] and recorder.marked == "unit"
    assert recorder.compliance[0][2] == text and recorder.compliance[0][4]["l2"]
    assert state["audit_input"] == "" and state["pending_directives"] == ["pending"]
    state.update(sales_input="您好", guardrail_checked=True, guardrail_hits=[])
    assert make_guardrail_node(detector)(state)["needs_human_review"]  # 后续正常轮不撤销升级


def test_warn_count_three_without_block():
    from app.agents.roleplay.guardrail_node import make_guardrail_node
    state = dict(guardrail_checked=True, guardrail_hits=[dict(rule="CMP-01", severity="warn")], guardrail_hit_count=2)
    assert make_guardrail_node(engine())(state)["needs_human_review"]


def test_cross_field_semantic_promise_removed_in_both_fields():
    value = {"reasons": ["我们保证", "当天放款"]}
    result = engine({"CMP-01": "我们保证\n当天放款"}).protect_tree(value)
    assert result.hits and all("审批" in text for text in value["reasons"])
    assert "当天放款" not in json.dumps(value, ensure_ascii=False)


def test_same_rule_different_promises_all_removed():
    value = {"reasons": ["保证当天放款", "肯定当天到账"]}
    engine({"CMP-01": "保证当天放款"}).protect_tree(value)
    assert all("审批" in text for text in value["reasons"])
    assert "肯定当天到账" not in json.dumps(value, ensure_ascii=False)


def test_independent_block_audit_overrides_model_compliance_score():
    from app.evaluation.evidence import EvidenceExtractor
    from app.evaluation.sales_score import SalesScorer
    from tests.test_evaluation_service import FakeLLM, CONVERSATION
    from tests.test_case_schema import make_case
    evidence = EvidenceExtractor(Settings(_env_file=None)).extract(CONVERSATION,
        guardrail_hits=[dict(rule="CMP-01", turn=1, snippet="当天放款", severity="warn"),
                        dict(rule="CMP-05", turn=2, snippet="单据可以想办法", severity="block")])
    assert evidence['compliance'] == []  # 未送达的拦截原话不能冒充销售对话证据
    case = make_case().model_dump()
    case['_compliance_events'] = [dict(action='block', turn=2, rule='CMP-05', raw_text='单据可以想办法')]
    report = SalesScorer(FakeLLM(), Settings(_env_file=None)).score(CONVERSATION, case, evidence)
    assert report["dimension_scores"]["compliance"]["gated_score"] == 0


def test_sse_sections_guarded_before_delivery_and_persistence():
    from tests.test_evaluation_service import make_service, SID, FakeEvalSessionService
    from app.evaluation import service as evaluation
    class Recorder(FakeEvalSessionService):
        def record_compliance(self, *args, **kwargs):
            pass
    service, factory = make_service()
    service._guardrail_service = engine({"CMP-01": "保证当天放款"})
    service._session_service = Recorder({"session_id": SID, "case_id": "INFRA_001"})
    from unittest.mock import patch
    with patch.object(evaluation.OpportunityDiagnoser, "analyze", return_value={
        "score": 0, "level": "C", "reasons": ["保证当天放款"], "dimension_scores": {}}):
        for event in service.run_stream(SID):
            if event["type"] in {"section", "done"}:
                assert "保证当天放款" not in json.dumps(event, ensure_ascii=False)
    assert "保证当天放款" not in json.dumps(factory.evals[0].opportunity, ensure_ascii=False)


def test_l2_failure_reason_is_sanitized_without_relaxing_redline():
    class Broken:
        def chat_json(self, *args, **kwargs):
            raise RuntimeError("sensitive-provider-body-sentinel")
    result = ComplianceEngine(Settings(_env_file=None), llm=Broken()).scan("资料不真实也可以包装一下")
    assert result.l2_degraded and result.l2_error == "RuntimeError"
    assert "sensitive-provider-body-sentinel" not in str(result)
    assert any(h.severity == "block" for h in result.hits)


def test_l2_fabricated_evidence_has_controlled_reason():
    class Forged:
        def chat_json(self, *args, **kwargs):
            return {"decisions": [dict(rule_id=f"CMP-{i:02}", hit=i == 1, evidence="invented" if i == 1 else "") for i in range(1, 10)]}
    result = ComplianceEngine(Settings(_env_file=None), llm=Forged()).scan("保证当天放款")
    assert result.l2_error == "evidence_mismatch" and result.l2_degraded
