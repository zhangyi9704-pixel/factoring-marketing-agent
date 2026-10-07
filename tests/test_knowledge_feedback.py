"""Publication commits, ownership, compensation and catalog visibility using real SQL."""
from contextlib import contextmanager
from copy import deepcopy
import pytest
from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import Session
from app.models import Base, User, KnowledgeDoc, KnowledgeDraft, BusinessAudit, AgentErrorLog
from app.config import Settings
from app.rag.ingest import KnowledgeIngestor
from app.rag.retriever import HybridRetriever
from app.rag.visibility import KnowledgeVisibility
from app.services.pending_service import PendingService, PendingConflict
from app.services.knowledge_feedback import KnowledgeFeedback
from app.services.domain_errors import FeedbackConflict, FeedbackValidation, FeedbackUnavailable
from app.services.auth_service import Principal
from test_ingest import Embedder, MemoryStore, HEADER

ADMIN = Principal(4, "admin_demo", "admin")
COACH = Principal(3, "coach_demo", "coach")
SALES = Principal(1, "sales_demo", "sales")
PAYLOAD = dict(action="adopt", content="人工验收演示正文：仅测试，不作为业务口径。", source="人工验收演示来源",
               category="product", industry="基建", topic="保理基础", effective_date="2026-10-06",
               adopted=dict(doc_id="KB-TEST-001", version="v2"))


class SearchStore(MemoryStore):
    def hybrid_search(self, *args, **kwargs):
        # Deliberately return archived/staged too: SQL defensive filter must work.
        return [dict(r, score=.9) for r in self.rows.values() if r["chunk_type"] == "child"]

    def get_parents(self, ids):
        return [deepcopy(r) for r in self.rows.values() if r["chunk_id"] in ids]


class Ranker:
    def rerank(self, query, items):
        return items


@pytest.fixture
def system(tmp_path, monkeypatch):
    engine = create_engine("sqlite:///" + (tmp_path / "feedback.db").as_posix())
    Base.metadata.create_all(engine)
    @contextmanager
    def factory():
        with Session(engine, expire_on_commit=False) as db, db.begin():
            yield db
    with factory() as db:
        for actor in (SALES, COACH, ADMIN):
            db.add(User(id=actor.id, username=actor.username, role=actor.role, enabled=True,
                        password_hash="unused", salt="unused"))
    from app.config import get_settings
    monkeypatch.setattr(get_settings().auth, "coach_sales", {"coach_demo": ["sales_demo"]})
    store = SearchStore()
    ingestor = KnowledgeIngestor(Settings(_env_file=None), embedder=Embedder(), store=store, session_factory=factory)
    path = tmp_path / "seed.md"
    for version in ("v1", "v2"):
        path.write_text(HEADER.format(version=version, status="conflicted") + "冲突验收演示。", encoding="utf-8")
        result = ingestor.ingest_file(path, source="product", priority=2)
        assert result.status == "indexed", result.error
    pending = PendingService(factory, engine)
    publisher = KnowledgeFeedback(factory, engine, ingestor, tmp_path / "drafts")
    pending.feedback_service = publisher
    retriever = HybridRetriever(Settings(_env_file=None), embedder=Embedder(), store=store, reranker=Ranker(),
                                visibility=KnowledgeVisibility(factory))
    yield pending, publisher, retriever, store, factory, engine
    engine.dispose()


def draft(system, payload=None, kind="conflict"):
    pending, publisher, *_ = system
    row = pending.register(kind, "冲突验收演示如何处理", user_id=1,
        context={"items": [dict(doc_id="KB-TEST-001", version=v, topic="保理基础") for v in ("v1", "v2")]})
    pending.transition("knowledge", row["id"], COACH, "processing")
    done = pending.transition("knowledge", row["id"], COACH, "resolved", "已核实来源，保存草稿",
                              feedback=payload or deepcopy(PAYLOAD), feedback_service=publisher)
    return publisher.get(done["knowledge_draft"]["id"], ADMIN)


def test_complete_conflict_adoption_retrieval_and_audit(system):
    pending, publisher, retriever, store, factory, _ = system
    assert retriever.retrieve("保理验收", top_k=10).mode == "conflict"
    saved = draft(system)
    assert saved["status"] == "draft"
    assert retriever.retrieve("保理验收", top_k=10).mode == "conflict"
    with pytest.raises(FeedbackConflict):
        publisher.publish(saved["id"], ADMIN)
    publisher.confirm(saved["id"], ADMIN, saved["content_hash"])
    result = publisher.publish(saved["id"], ADMIN)
    assert result["status"] == "published"
    retrieved = retriever.retrieve("保理验收", top_k=10)
    assert retrieved.mode == "rag"
    assert {i.version for i in retrieved.items} == {saved["metadata"]["version"]}
    assert all(i.status == "active" and "人工验收" in i.parent_content for i in retrieved.items)
    rows_before = deepcopy(store.rows)
    assert publisher.publish(saved["id"], ADMIN) == result
    assert store.rows == rows_before
    with factory() as db:
        assert {r.version:r.status for r in db.scalars(select(KnowledgeDoc))} == {
            "v1":"archived", "v2":"archived", saved["metadata"]["version"]:"active"}
        records = db.scalars(select(BusinessAudit).order_by(BusinessAudit.id)).all()
        assert [r.action for r in records] == ["pending_claimed", "pending_closed", "draft_saved",
            "draft_confirmed", "publish_started", "ingestion_verified", "knowledge_published", "versions_archived"]
        assert all(r.actor_id in (3,4) and r.pending_id == saved["pending_id"] and r.created_at for r in records)
        assert db.scalar(select(func.count()).select_from(AgentErrorLog)) == 0


def test_missing_supplements_new_controlled_entry(system):
    payload = dict(PAYLOAD, action="supplement"); payload.pop("adopted")
    saved = draft(system, payload, "missing")
    assert saved["metadata"]["doc_id"].startswith("KB-FB-") and saved["supersedes"] == []
    publisher = system[1]
    assert publisher.path(saved["id"]).is_relative_to(publisher.root)
    publisher.confirm(saved["id"], ADMIN, saved["content_hash"])
    assert publisher.publish(saved["id"], ADMIN)["status"] == "published"


@pytest.mark.parametrize("phase", ["partial", "cleanup", "activation", "final_commit"])
def test_failure_preserves_old_effective_versions_and_records_error(system, phase, monkeypatch):
    _, publisher, retriever, store, factory, _ = system
    saved = draft(system)
    publisher.confirm(saved["id"], ADMIN, saved["content_hash"])
    if phase in ("partial", "cleanup"):
        store.fail = phase
    elif phase == "activation":
        original = store.upsert
        failed = [False]
        def upsert(rows):
            if any(r["status"] == "active" for r in rows) and not failed[0]:
                failed[0] = True
                original(rows[:1])
                raise RuntimeError("fault")
            return original(rows)
        monkeypatch.setattr(store, "upsert", upsert)
    else:
        @contextmanager
        def fail_commit():
            with factory() as db:
                yield db
                if any(isinstance(r, BusinessAudit) and r.action == "knowledge_published" for r in db.new):
                    raise RuntimeError("commit fault")
        publisher.factory = fail_commit
    with pytest.raises(FeedbackUnavailable):
        publisher.publish(saved["id"], ADMIN)
    assert publisher.get(saved["id"], ADMIN)["status"] == "failed"
    result = retriever.retrieve("保理验收", top_k=10)
    assert result.mode == "conflict" and {i.version for i in result.items} == {"v1", "v2"}
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(AgentErrorLog)) == 1
        assert "publish_failed" in db.scalars(select(BusinessAudit.action)).all()
        assert "knowledge_published" not in db.scalars(select(BusinessAudit.action)).all()


def test_catalog_survives_crash_after_vector_changes_and_invalidation_loss(system, monkeypatch):
    _, publisher, retriever, store, factory, _ = system
    saved = draft(system)
    # Vector-only status writes do not make new facts effective or hide old SQL versions.
    for row in store.rows.values():
        row["status"] = "archived"
    assert {i.version for i in retriever.retrieve("保理验收", top_k=10).items} == {"v1", "v2"}
    publisher.confirm(saved["id"], ADMIN, saved["content_hash"])
    monkeypatch.setattr("app.services.knowledge_feedback.invalidate", lambda: None)
    monkeypatch.setattr("app.rag.revision.invalidate", lambda: None)
    publisher.publish(saved["id"], ADMIN)
    assert {i.version for i in retriever.retrieve("保理验收", top_k=10).items} == {saved["metadata"]["version"]}


def test_confirmation_permissions_hash_and_path_guards(system):
    publisher = system[1]; saved = draft(system)
    for actor in (COACH, SALES):
        with pytest.raises(PermissionError):
            publisher.confirm(saved["id"], actor, saved["content_hash"])
        with pytest.raises(PermissionError):
            publisher.publish(saved["id"], actor)
    with pytest.raises(PermissionError):
        publisher.get(saved["id"], SALES)
    with pytest.raises(FeedbackConflict):
        publisher.confirm(saved["id"], ADMIN, "0"*64)
    with pytest.raises(FeedbackValidation):
        publisher.path("../../outside")
    publisher.path(saved["id"]).write_text("tampered", encoding="utf-8")
    with pytest.raises(FeedbackConflict):
        publisher.confirm(saved["id"], ADMIN, saved["content_hash"])


def test_invalid_adoption_rolls_back_pending_close(system):
    pending, publisher, *_ = system
    row = pending.register("conflict", "问题", user_id=1, context={"items": []})
    pending.transition("knowledge", row["id"], COACH, "processing")
    with pytest.raises(FeedbackValidation):
        pending.transition("knowledge", row["id"], COACH, "resolved", "结论", feedback=PAYLOAD, feedback_service=publisher)
    assert pending.get("knowledge", row["id"], COACH)["status"] == "processing"


def test_mysql_json_key_order_does_not_change_confirmed_facts(system):
    publisher = system[1]; saved = draft(system)
    with system[4]() as db:
        row = db.get(KnowledgeDraft, saved["id"])
        row.metadata_json = dict(sorted(row.metadata_json.items()))
    publisher.confirm(saved["id"], ADMIN, saved["content_hash"])
    assert publisher.publish(saved["id"], ADMIN)["status"] == "published"


def test_competing_publications_ingest_once(system, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    publisher = system[1]; saved = draft(system)
    publisher.confirm(saved["id"],ADMIN,saved["content_hash"])
    calls = []
    original = publisher.ingestor.embedder.embed
    def embed(texts):
        calls.append(len(texts)); return original(texts)
    monkeypatch.setattr(publisher.ingestor.embedder,"embed",embed)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _:publisher.publish(saved["id"],ADMIN),range(2)))
    assert [r["status"] for r in results] == ["published","published"]
    assert len(calls) == 1


def test_failed_compensation_blocks_replay_and_preserves_sql_visibility(system, monkeypatch):
    publisher, retriever, store = system[1:4]; saved = draft(system)
    publisher.confirm(saved["id"],ADMIN,saved["content_hash"])
    store.fail = "partial"
    monkeypatch.setattr(store,"restore_doc",lambda *args: (_ for _ in ()).throw(RuntimeError("restore fault")))
    with pytest.raises(FeedbackUnavailable):
        publisher.publish(saved["id"],ADMIN)
    assert publisher.get(saved["id"],ADMIN)["status"] == "reconcile_required"
    assert {i.version for i in retriever.retrieve("保理验收",top_k=10).items} == {"v1","v2"}
    with pytest.raises(FeedbackConflict):
        publisher.publish(saved["id"],ADMIN)


def test_publication_during_rerank_never_returns_archived_items(system, monkeypatch):
    publisher, retriever = system[1:3]; saved = draft(system)
    publisher.confirm(saved["id"],ADMIN,saved["content_hash"])
    def rerank(query,items):
        publisher.publish(saved["id"],ADMIN)
        return items
    monkeypatch.setattr(retriever.reranker,"rerank",rerank)
    result = retriever.retrieve("保理验收",top_k=10)
    assert not result.items and result.mode == "missing"
    assert result.reason == "knowledge_changed_during_retrieval"


def test_legacy_filename_never_authorizes_staged_version():
    catalog = ({}, {"entry.md"}, "stamp")
    rows = [dict(doc_id="new-unpublished",version="v1",status="active",doc_name="entry.md"),
            dict(doc_id="",version="",status="active",doc_name="entry.md")]
    assert KnowledgeVisibility.filter(rows,catalog) == [rows[1]]
    assert 'doc_id == ""' in KnowledgeVisibility.expression(catalog)


def test_http_permissions_explicit_confirmation_and_arbitrary_path_rejection(system):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.api.deps import get_current_user
    from app.api.v1.pending import get_pending_service
    from app.api.v1.knowledge import get_feedback_service
    pending, publisher, *_ = system
    actor = [COACH]
    overrides = dict(app.dependency_overrides)
    app.dependency_overrides[get_current_user] = lambda: actor[0]
    app.dependency_overrides[get_pending_service] = lambda: pending
    app.dependency_overrides[get_feedback_service] = lambda: publisher
    client = TestClient(app)
    try:
        row = pending.register("conflict", "API验收", user_id=1, context={"items": [
            dict(doc_id="KB-TEST-001", version=v, topic="保理基础") for v in ("v1", "v2")]})
        path = "/api/v1/knowledge/pending/" + str(row["id"])
        assert client.patch(path, json={"status":"processing"}).status_code == 200
        body = dict(status="resolved", resolution="核实", feedback=dict(PAYLOAD, file_path="D:/outside.md"))
        assert client.patch(path, json=body).status_code == 422
        body["feedback"].pop("file_path")
        response = client.patch(path, json=body)
        assert response.status_code == 200, response.text
        ident = response.json()["knowledge_draft"]["id"]
        url = "/api/v1/knowledge/drafts/" + ident
        assert client.get(url).status_code == 200
        assert client.post(url+"/confirm", json={"content_hash":"0"*64,"content_confirmed":True,"source_confirmed":True}).status_code == 403
        assert client.post(url+"/publish").status_code == 403
        actor[0] = SALES
        assert client.get(url).status_code == 403
        actor[0] = Principal(9, "unassigned", "coach")
        assert client.get(url).status_code == 403
        assert client.get(url+"/audit").status_code == 403
        actor[0] = ADMIN
        saved = client.get(url).json()
        assert client.post(url+"/publish").status_code == 409
        assert client.post(url+"/confirm", json={"content_hash":saved["content_hash"]}).status_code == 422
        assert client.post(url+"/confirm", json={"content_hash":saved["content_hash"],"content_confirmed":True,"source_confirmed":True}).status_code == 200
        assert client.post(url+"/publish").status_code == 200
        assert len(client.get(url+"/audit").json()) == 8
    finally:
        app.dependency_overrides.clear(); app.dependency_overrides.update(overrides)
        client.close()
