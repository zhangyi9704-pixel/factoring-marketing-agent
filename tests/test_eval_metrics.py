"""Metric denominator, independence and uncached inference regression tests."""
import copy
from types import SimpleNamespace

import pytest

from scripts.eval_metrics import (
    ROOT, briefing_state, coverage, evaluate_fresh, latency, load_rows, summarize, validate_dataset, variance,
)
from app.config import get_settings


def test_real_case_assets_work_with_node_safety_schema_without_internal_marker_dicts():
    import yaml
    from app.schemas.case import CaseState, assert_no_internal_leak
    from app.agents.coach.briefing import make_briefing_node
    from app.rag.retriever import RetrievalResult
    for path in (ROOT / "cases").glob("*.yaml"):
        case = CaseState(**yaml.safe_load(path.read_text(encoding="utf-8")))
        llm = SimpleNamespace(chat_json=lambda *a, **kw: {})
        retriever = SimpleNamespace(retrieve=lambda query, **kw: RetrievalResult(query, [], mode="missing"))
        output = make_briefing_node(llm, settings=get_settings(), retriever=retriever)(briefing_state(case))
        assert output["briefing"]["case_id"] == case.case_id
        assert_no_internal_leak(output["briefing"])


def test_controlled_model_failure_produces_rule_briefing_with_phase_timings():
    import yaml
    from app.schemas.case import CaseState, assert_no_internal_leak
    from app.agents.coach.briefing import make_briefing_node
    from app.rag.retriever import RetrievalResult
    case = CaseState(**yaml.safe_load((ROOT / 'cases/INFRA_001.yaml').read_text(encoding='utf-8')))
    def unavailable(*args, **kwargs):
        raise RuntimeError('controlled external failure')
    output = make_briefing_node(SimpleNamespace(chat_json=unavailable), settings=get_settings(),
        retriever=SimpleNamespace(retrieve=lambda query, **kw: RetrievalResult(query, [], mode='missing')))(briefing_state(case))
    assert output['errors'] and output['briefing']['priority_questions']
    assert_no_internal_leak(output['briefing'])
    timing = output['briefing_timings']
    assert set(timing) == {'retrieval_s', 'llm_s', 'guardrail_s', 'assembly_s', 'persist_s', 'total_s'}
    assert all(value >= 0 for value in timing.values())
    assert timing['total_s'] < 5  # Only immediate-failure rule assembly; not network-timeout SLO.


def test_dataset_is_independent_and_incomplete_dialogues_are_measured():
    rows = load_rows(ROOT / "data/eval/metrics.jsonl")
    validate_dataset(rows)
    dialogues = [r for r in rows if r["kind"] == "dialogue"]
    assert [len(coverage(r["conversation"])) for r in dialogues] == [6, 5, 5]
    assert all(set(coverage(r["conversation"])) == set(r["expected_elements"]) for r in dialogues)
    duplicate = copy.deepcopy(rows)
    duplicate[0]["query"] = load_rows(ROOT / "data/eval/calibration.jsonl")[0]["query"]
    with pytest.raises(ValueError, match="overlaps"):
        validate_dataset(duplicate)


def test_sample_variance_and_p95_do_not_invent_single_sample_certainty():
    assert variance([80]) is None
    assert variance([80, 90]) == 50
    assert latency(list(range(1, 21)))["p95_s"] == 19


def test_service_failure_keeps_denominator_and_safe_error_is_not_false_positive():
    result = dict(repeats=2, details=[
        dict(task="retrieval", kind="typical", industry="基建", correct=True, available=True, elapsed_s=1),
        dict(task="retrieval", kind="typical", industry="基建", correct=False, available=False, elapsed_s=1),
        dict(task="compliance", kind="safe", correct=False, l2_degraded=True, hits=[]),
    ])
    summarize(result)
    assert result["metrics"]["knowledge_recall_accuracy"] == .5
    assert result["metrics"]["safe_false_positive_rate"] is None
    assert result["metrics"]["retrieval_latency"]["n"] == 1


def test_rejected_evidence_keeps_observed_variance_but_not_quality_pass():
    result = dict(repeats=2, details=[dict(id="sample", task="dialogue", kind="dialogue", covered=[],
        correct=False, samples=[
            dict(valid=False, opportunity=80., sales=70., errors=["evidence_rejected:payer_confirmation"],
                 raw_dimensions=[{"payer_confirmation": 90.}]),
            dict(valid=False, opportunity=90., sales=80., errors=["evidence_rejected:payer_confirmation"],
                 raw_dimensions=[{"payer_confirmation": 100.}]),
        ])])
    summarize(result)
    score = result["metrics"]["score_consistency"][0]
    assert score["observed_opportunity_variance"] == 50
    assert score["raw_dimension_variance"]["payer_confirmation"] == 50
    assert score["opportunity_variance"] is None
    assert score["valid"] is False


def test_conflict_with_expected_items_is_in_recall_denominator():
    result = dict(repeats=2, details=[
        dict(task="retrieval", kind="typical", industry="医药", correct=True, available=True, elapsed_s=1),
        dict(task="retrieval", kind="conflict", industry="医药", correct=False, available=True, elapsed_s=1),
    ])
    summarize(result)
    assert result["metrics"]["knowledge_recall_accuracy"] == .5
    assert result["metrics"]["typical_recall_accuracy"] == 1
    assert result["metrics"]["per_industry_recall"]["医药"] == .5


def test_fresh_entry_calls_model_on_every_repeat_and_fallback_is_invalid():
    settings = get_settings()
    row = next(r for r in load_rows(ROOT / "data/eval/metrics.jsonl") if r["kind"] == "dialogue")
    dimensions = set(settings.evaluation.modules.opportunity_diagnosis.dimensions)
    dimensions |= set(settings.evaluation.modules.sales_score.rubric)
    calls = []
    def inference(*args, **kwargs):
        calls.append(kwargs)
        return {"dimensions": {d: {"score": 70.0 + len(calls), "rationale": ""} for d in dimensions}}
    llm = SimpleNamespace(chat_json_once=inference)
    first = evaluate_fresh(row["conversation"], row["industry"], settings, llm)
    second = evaluate_fresh(row["conversation"], row["industry"], settings, llm)
    assert len(calls) == 4
    assert first["valid"] and second["valid"]
    assert first["opportunity"] != second["opportunity"]
    assert first["raw_dimensions"] != second["raw_dimensions"]
    def unavailable(*args, **kwargs):
        raise RuntimeError("external unavailable")
    failed = evaluate_fresh(row["conversation"], row["industry"], settings, SimpleNamespace(chat_json_once=unavailable))
    assert failed["valid"] is False
    assert "deterministic_fallback" in failed["errors"]
