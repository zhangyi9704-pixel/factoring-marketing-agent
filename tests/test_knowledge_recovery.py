"""Offline SQL commit ambiguity, durable recovery and published body snapshots."""
from contextlib import contextmanager
from copy import deepcopy
import pytest
from sqlalchemy import select

from app.models import KnowledgeDoc, KnowledgeDraft, KnowledgeIngestionOperation
from app.rag.visibility import KnowledgeVisibility
from app.evaluation.knowledge import active_sources
from app.services.domain_errors import FeedbackConflict, FeedbackUnavailable, FeedbackValidation
from test_ingest import kb, ingest
from test_knowledge_feedback import system, draft, ADMIN, COACH


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_committed_ingestion_acknowledgement_loss_keeps_exact_new_body(kb, version):
    assert ingest(kb).status == "indexed"
    original = kb[0].session_factory
    @contextmanager
    def lost_ack():
        with original() as db:
            yield db
        raise ConnectionError("commit acknowledgement lost after commit")
    kb[0].session_factory = lost_ack
    result = ingest(kb, version=version, body="已提交的新正文。")
    assert result.status == "indexed"
    with original() as db:
        active = db.scalars(select(KnowledgeDoc).where(KnowledgeDoc.status == "active")).one()
        assert active.version == version and active.content_snapshot == "已提交的新正文。"
        operations = db.scalars(select(KnowledgeIngestionOperation)).all()
        assert len(operations) == 2 and operations[-1].input_hash == active.content_hash
    effective = KnowledgeVisibility(original).filter(list(kb[1].rows.values()), KnowledgeVisibility(original).catalog())
    assert effective and all(r["version"] == version and "已提交" in r["content"] for r in effective)


def test_catalog_rejects_uncommitted_same_version_body_and_changes_cache_digest(kb):
    assert ingest(kb, body="原始依据。").status == "indexed"
    visibility = KnowledgeVisibility(kb[2])
    before = visibility.catalog()
    mutated = deepcopy(list(kb[1].rows.values()))
    for row in mutated:
        row["content"] = "未提交的新正文。"
    assert visibility.filter(mutated, before) == []
    assert ingest(kb, body="已提交新依据。").status == "indexed"
    assert visibility.catalog()[2] != before[2]


def test_ambiguous_commit_with_unavailable_read_never_blindly_compensates(kb, monkeypatch):
    assert ingest(kb).status == "indexed"
    original = kb[0].session_factory
    @contextmanager
    def lost_ack():
        with original() as db:
            yield db
        raise ConnectionError("ack lost")
    kb[0].session_factory = lost_ack
    def unreadable(*args, **kwargs):
        raise ConnectionError("reconciliation read unavailable")
    monkeypatch.setattr("app.rag.ingest.Session", unreadable)
    result = ingest(kb, version="v2", body="已提交的新正文。")
    assert result.status == "failed" and result.error == "ingestion_outcome_unknown"
    assert any(r["version"] == "v2" for r in kb[1].rows.values())


def test_failed_publish_recovers_only_then_explicit_retry(system):
    publisher, store = system[1], system[3]
    saved = draft(system)
    publisher.confirm(saved["id"], ADMIN, saved["content_hash"])
    store.fail = "partial"
    with pytest.raises(FeedbackUnavailable):
        publisher.publish(saved["id"], ADMIN)
    calls = []
    original = publisher.ingestor.embedder.embed
    publisher.ingestor.embedder.embed = lambda texts: (calls.append(texts), original(texts))[1]
    assert publisher.recover(saved["id"], ADMIN)["status"] == "confirmed"
    assert calls == [] and system[2].retrieve("保理", top_k=10).mode == "conflict"
    assert publisher.publish(saved["id"], ADMIN)["status"] == "published"
    assert len(calls) == 1
    assert publisher.recover(saved["id"], ADMIN)["status"] == "published"


def test_recovery_restores_durable_snapshot_after_worker_crash(system):
    publisher, store, factory = system[1], system[3], system[4]
    saved = draft(system)
    publisher.confirm(saved["id"], ADMIN, saved["content_hash"])
    # Simulate an uncatchable worker exit after active/archived vector changes.
    original = store.upsert
    def crash(rows):
        if any(r["status"] == "active" for r in rows):
            original(rows)
            raise SystemExit("worker terminated")
        return original(rows)
    store.upsert = crash
    with pytest.raises(SystemExit):
        publisher.publish(saved["id"], ADMIN)
    store.upsert = original
    with factory() as db:
        row = db.get(KnowledgeDraft, saved["id"])
        assert row.status == "publishing" and row.publication_snapshot
    assert publisher.recover(saved["id"], ADMIN)["status"] == "confirmed"
    assert {i.version for i in system[2].retrieve("保理", top_k=10).items} == {"v1", "v2"}


def test_recovery_rejects_other_actor_changed_catalog_and_tampered_draft(system):
    publisher, store, factory = system[1], system[3], system[4]
    saved = draft(system)
    publisher.confirm(saved["id"], ADMIN, saved["content_hash"])
    store.fail = "partial"
    with pytest.raises(FeedbackUnavailable): publisher.publish(saved["id"], ADMIN)
    with pytest.raises(PermissionError): publisher.recover(saved["id"], COACH)
    with factory() as db:
        old = db.scalars(select(KnowledgeDoc).where(KnowledgeDoc.version == "v1")).one()
        old.content_hash = "0" * 64
    with pytest.raises(FeedbackConflict, match="recovery_catalog_changed"):
        publisher.recover(saved["id"], ADMIN)
    with factory() as db:
        db.get(KnowledgeDraft, saved["id"]).content = "unconfirmed changed body"
    with pytest.raises(FeedbackConflict, match="draft_content_changed"):
        publisher.recover(saved["id"], ADMIN)


def test_published_feedback_body_is_evaluation_source_without_local_files(system, monkeypatch, tmp_path):
    publisher = system[1]
    saved = draft(system)
    publisher.confirm(saved["id"], ADMIN, saved["content_hash"])
    publisher.publish(saved["id"], ADMIN)
    monkeypatch.setattr("app.evaluation.knowledge.PROJECT_ROOT", tmp_path / "missing-baseline")
    publisher.path(saved["id"]).unlink()
    sources = active_sources(system[4], "基建")
    assert len(sources) == 1
    assert sources[0]["version"] == saved["metadata"]["version"] and sources[0]["content"] == saved["content"]


def test_single_conflict_cannot_archive_only_one_known_party(system):
    pending, publisher = system[:2]
    row = pending.register("conflict", "单方未完整核实", user_id=1,
        context={"items": [dict(doc_id="KB-TEST-001", version="v1", topic="保理基础")]})
    pending.transition("knowledge", row["id"], COACH, "processing")
    from test_knowledge_feedback import PAYLOAD
    with pytest.raises(FeedbackValidation, match="incomplete_conflict"):
        pending.transition("knowledge", row["id"], COACH, "resolved", "核实", feedback=PAYLOAD)


def test_recovery_verification_failure_keeps_reconciliation_required(system, monkeypatch):
    publisher, store = system[1], system[3]
    saved = draft(system)
    publisher.confirm(saved["id"], ADMIN, saved["content_hash"])
    store.fail = "partial"
    with pytest.raises(FeedbackUnavailable): publisher.publish(saved["id"], ADMIN)
    original = store.restore_doc
    def corrupt_restoration(identity, version, rows):
        original(identity, version, rows)
        next(iter(store.rows.values()))["content"] = "corrupt recovery content"
    monkeypatch.setattr(store, "restore_doc", corrupt_restoration)
    with pytest.raises(FeedbackUnavailable, match="recovery_verification_failed"):
        publisher.recover(saved["id"], ADMIN)
    assert publisher.get(saved["id"], ADMIN)["status"] == "reconcile_required"


def test_committed_but_wrong_vector_is_not_reported_as_indexed_or_undone(kb):
    assert ingest(kb).status == "indexed"
    original = kb[0].session_factory
    @contextmanager
    def bad_ack_and_vector():
        with original() as db:
            yield db
        for row in kb[1].rows.values():
            if row["version"] == "v2": row["content"] = "post-commit corruption"
        raise ConnectionError("commit acknowledgement lost")
    kb[0].session_factory = bad_ack_and_vector
    result = ingest(kb, version="v2", body="committed new body")
    assert result.status == "failed" and result.error == "committed_ingestion_requires_reconciliation"
    with original() as db:
        assert db.scalars(select(KnowledgeDoc).where(KnowledgeDoc.version == "v2")).one().status == "active"
    assert any(r["version"] == "v2" and r["content"] == "post-commit corruption" for r in kb[1].rows.values())
