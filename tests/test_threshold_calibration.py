"""校准口径、分数尺度、留出隔离及标签完整性。"""
from pathlib import Path
import pytest
from scripts.calibrate_threshold import load_labels, validate_sets, validate_source_versions, evaluate, recommend, predict, same_sampling_context


def item(doc="KB-X", version="v1", score=.95, rrf=.02, topic="主题", status="active"):
    return dict(doc_id=doc, version=version, rerank_score=score, retrieval_score=rrf, topic=topic, status=status)


def label(identity="true", kind="true", expected=None):
    return dict(id=identity, query=identity, industry="基建", kind=kind,
        expected_mode="rag" if kind == "true" else "conflict" if kind == "conflict" else "missing",
        expected_items=expected if expected is not None else [dict(doc_id="KB-X", version="v1")])


def sample(identity="true", items=None):
    return dict(id=identity, rerank=items or [item()], retrieval=items or [item()])


def test_real_labels_have_required_coverage_and_disjoint_queries():
    root = Path(__file__).resolve().parents[1] / "data/eval"
    calibration = load_labels(root / "calibration.jsonl", "calibration")
    held = load_labels(root / "holdout.jsonl", "holdout")
    validate_sets(calibration, held)
    validate_source_versions(calibration + held)
    assert len(calibration) == 30 and len(held) == 15


def test_split_overlap_rejected():
    root = Path(__file__).resolve().parents[1] / "data/eval"
    calibration = load_labels(root / "calibration.jsonl", "calibration")
    held = load_labels(root / "holdout.jsonl", "holdout")
    held[0]["query"] = calibration[0]["query"] + "？！"
    with pytest.raises(ValueError, match="split leakage"):
        validate_sets(calibration, held)


def test_recall_requires_exact_version_and_branch():
    result = evaluate([label()], [sample(items=[item(version="v2")])], "rerank", .8, 5)
    assert result["recall"] == 0 and result["branch_accuracy"] == 0


def test_blind_false_recall_counts_conflict_as_well_as_rag():
    rows = [label("blind", "blind", [])]
    items = [item("A", status="conflicted"), item("B", status="conflicted")]
    result = evaluate(rows, [sample("blind", items)], "rerank", .8, 5)
    assert result["false_recall"] == 1 and result["branches"]["conflict"] == 1


def test_retained_conflicted_marker_always_requires_manual_review():
    items = [item("A", status="conflicted"), item("B", score=.7, status="conflicted")]
    s = sample(items=items)
    assert predict(s, "rerank", .7, 5)[0] == "conflict"
    assert predict(s, "rerank", .8, 5)[0] == "conflict"
    assert predict(s, "rerank", .7, 1)[0] == "conflict"


def test_score_sources_do_not_share_threshold_scale():
    s = sample()
    assert predict(s, "rerank", .8, 5)[0] == "rag"
    assert predict(s, "retrieval", .8, 5)[0] == "missing"
    assert predict(s, "retrieval", .019, 5)[0] == "rag"


def test_holdout_scores_cannot_change_suggested_threshold():
    rows = [label(), label("blind", "blind", [])]
    samples = [sample(), sample("blind", [item(score=.2)])]
    first = recommend(rows, samples + [sample("held", [item(score=.99)])], "rerank", 5)[1]
    second = recommend(rows, samples + [sample("held", [item(score=.01)])], "rerank", 5)[1]
    assert first["threshold"] == second["threshold"]
    assert first["recall"] == 1 and first["false_recall"] == 0


def test_inseparable_scores_never_claim_feasible_threshold():
    rows = [label(), label("blind", "blind", [])]
    _, selected = recommend(rows, [sample(), sample("blind")], "rerank", 5)
    assert selected is None


def test_version_typo_cannot_pass_source_validation():
    row = label(expected=[dict(doc_id="KB-PR-003", version="KB-v1.0")])
    with pytest.raises(ValueError, match="source/version"):
        validate_source_versions([row])


def test_score_reuse_requires_same_requests_models_and_knowledge():
    original = dict(labels={"holdout": "old"}, sampling_labels="same", knowledge_revision="a", rerank_model="model1")
    assert same_sampling_context(original, dict(original, labels={"holdout": "corrected"}))
    assert not same_sampling_context(original, dict(original, sampling_labels="other"))
    assert not same_sampling_context(original, dict(original, knowledge_revision="b"))
    assert not same_sampling_context(original, dict(original, rerank_model="model2"))
