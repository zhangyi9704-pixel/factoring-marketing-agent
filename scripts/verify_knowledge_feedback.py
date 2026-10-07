"""Owned demonstration knowledge only; real MySQL/Milvus/cloud/API publication.

PowerShell project root: .\.venv\Scripts\python.exe scripts\verify_knowledge_feedback.py
Repeated successful runs inspect saved evidence only. Interrupted runs never replay
writes automatically. No graph, formal knowledge edits, passwords or token output.
"""
from pathlib import Path
import sys
import json
import logging
import secrets
import uuid
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def continue_saved(draft_id):
    """Explicit continuation of the reviewed owned draft; no queue/seed replay."""
    logging.disable(logging.CRITICAL)
    from sqlalchemy import select
    from pydantic import SecretStr
    from fastapi.testclient import TestClient
    from app.config import get_settings
    from app.db.mysql import get_mysql_database
    from app.models import User, KnowledgeDoc, KnowledgeDraft, KnowledgePending, AgentErrorLog
    from app.services.auth_service import Principal, issue_token
    from app.services.knowledge_feedback import KnowledgeFeedback
    from app.rag.ingest import KnowledgeIngestor
    from app.rag.retriever import HybridRetriever
    from app.rag.visibility import KnowledgeVisibility
    from app.main import app
    run = json.loads(Path(".runtime/knowledge-feedback-live/run.json").read_text(encoding="utf-8"))["run"]
    database = get_mysql_database()
    doc_id = "KB-FB-VERIFY-" + run
    publisher = KnowledgeFeedback()
    with database.session_scope() as db:
        row = db.get(KnowledgeDraft,draft_id)
        assert row and row.status == "draft" and row.metadata_json["doc_id"] == doc_id
        pending = db.get(KnowledgePending,row.pending_id)
        assert pending.status == "resolved" and run in pending.question
        assert "非真实业务制度" in row.metadata_json["source"] and "不作为真实业务依据" in row.content
        query, pending_id = pending.question, pending.id
        users = [Principal(u.id,u.username,u.role) for u in db.scalars(select(User).where(User.username.like(f"feedback_{run}_%")))]
        assert len(users) == 3 and {u.role for u in users} == {"sales","coach","admin"}
        assert all(not db.get(User,u.id).enabled for u in users)
    settings = get_settings()
    original_secret,original_grants = settings.auth.jwt_secret,settings.auth.coach_sales
    settings.auth.jwt_secret = SecretStr(secrets.token_urlsafe(48))
    admin = next(u for u in users if u.role == "admin")
    coach = next(u for u in users if u.role == "coach")
    ingestor = KnowledgeIngestor()
    class ScopedStore:
        def hybrid_search(self,*args,filter_expr,**kwargs):
            return ingestor.store.hybrid_search(*args,filter_expr=f'({filter_expr}) and doc_id == "{doc_id}"',**kwargs)
        def get_parents(self,ids):
            return ingestor.store.get_parents(ids)
    retriever = HybridRetriever(store=ScopedStore(),visibility=KnowledgeVisibility(database.session_scope))
    before = retriever.retrieve(query,top_k=10)
    assert before.mode == "conflict" and not before.rerank_degraded
    try:
        with database.session_scope() as db:
            db.query(User).filter(User.id.in_([u.id for u in users])).update({User.enabled:True},synchronize_session=False)
        ah = {"Authorization":"Bearer "+issue_token(admin)}
        ch = {"Authorization":"Bearer "+issue_token(coach)}
        url = "/api/v1/knowledge/drafts/" + draft_id
        with TestClient(app) as client:
            saved = client.get(url,headers=ah).json()
            assert client.post(url+"/publish",headers=ch).status_code == 403
            assert client.post(url+"/publish",headers=ah).status_code == 409
            assert client.post(url+"/confirm",headers=ah,json=dict(content_hash=saved["content_hash"],
                content_confirmed=True,source_confirmed=True)).status_code == 200
            response = client.post(url+"/publish",headers=ah)
            assert response.status_code == 200, "publication_failed_check_audit"
            assert client.post(url+"/publish",headers=ah).json()["status"] == "published"
            records = client.get(url+"/audit",headers=ah).json()
        after = retriever.retrieve(query,top_k=10)
        version = saved["metadata"]["version"]
        assert after.mode == "rag" and not after.rerank_degraded and after.items
        assert {i.version for i in after.items} == {version}
        assert all(i.status == "active" and "演示乙" in i.parent_content for i in after.items)
        with database.session_scope() as db:
            assert {r.version:r.status for r in db.scalars(select(KnowledgeDoc).where(KnowledgeDoc.doc_id == doc_id))} == {
                "v1":"archived","v2":"archived",version:"active"}
            assert not db.scalars(select(AgentErrorLog).where(AgentErrorLog.path == url+"/publish")).all()
        actions = [r["action"] for r in records]
        assert actions == ["pending_claimed","pending_closed","draft_saved","draft_confirmed","publish_started",
                           "ingestion_verified","knowledge_published","versions_archived"]
        report = dict(run=run,doc_id=doc_id,draft_id=draft_id,pending_id=pending_id,version=version,
            backend="real MySQL + Milvus + cloud embedding/rerank",scope="owned demonstration document only",
            before_mode=before.mode,after_mode=after.mode,retrieved_versions=sorted({i.version for i in after.items}),
            old_versions_archived=True,audit_actions=actions,idempotent_publish=True,
            http={"coach_publish":403,"unconfirmed_publish":409,"confirm":200,"publish":200},
            continuation="existing draft only after MySQL JSON ordering fix; no registration/closure replay")
    finally:
        settings.auth.jwt_secret,settings.auth.coach_sales = original_secret,original_grants
        with database.session_scope() as db:
            db.query(User).filter(User.id.in_([u.id for u in users])).update({User.enabled:False},synchronize_session=False)
    report.update(test_user_ids=[u.id for u in users],test_users_disabled=True)
    Path("docs/validation/4.3-feedback-live.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    Path(".runtime/knowledge-feedback-live/run.json").write_text(json.dumps({"run":run,"state":"complete"}),encoding="utf-8")
    print("[OK] existing draft -> confirmation -> publication -> rag; archives verified; eight audits; users disabled")


def main():
    logging.disable(logging.CRITICAL)
    from sqlalchemy import select
    from pydantic import SecretStr
    from fastapi.testclient import TestClient
    from app.config import get_settings
    from app.db.mysql import get_mysql_database
    from app.db.pending_migration import migrate_pending
    from app.models import User, KnowledgeDraft, KnowledgeDoc, BusinessAudit, AgentErrorLog
    from app.services.auth_service import Principal, hash_password, issue_token
    from app.services.knowledge_feedback import KnowledgeFeedback
    from app.services.pending_service import PendingService
    from app.rag.ingest import KnowledgeIngestor
    from app.rag.retriever import HybridRetriever
    from app.rag.visibility import KnowledgeVisibility
    from app.rag.loader import parse_metadata
    from app.main import app
    report_path = Path("docs/validation/4.3-feedback-live.json")
    checkpoint = Path(".runtime/knowledge-feedback-live/run.json")
    database = get_mysql_database()
    if report_path.exists():
        saved = json.loads(report_path.read_text(encoding="utf-8"))
        with database.session_scope() as db:
            draft = db.get(KnowledgeDraft, saved["draft_id"])
            assert draft.status == "published"
            rows = db.scalars(select(KnowledgeDoc).where(KnowledgeDoc.doc_id == saved["doc_id"])).all()
            assert {r.version:r.status for r in rows} == {"v1":"archived", "v2":"archived", saved["version"]:"active"}
            assert all(not db.get(User, i).enabled for i in saved["test_user_ids"])
        print("[OK] saved publication/archives verified; test users disabled; read-only")
        return
    if checkpoint.exists():
        raise RuntimeError("interrupted_run_requires_manual_reconciliation")
    migrate_pending(database.engine)
    assert migrate_pending(database.engine) == []
    run = uuid.uuid4().hex[:12]
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text(json.dumps({"run":run,"state":"started"}), encoding="utf-8")
    doc_id = "KB-FB-VERIFY-" + run
    topic = "知识回流验收演示-" + run
    source = "工程人工编写的隔离验收演示（非真实业务制度）"
    query = f"{topic}演示确认编号是哪个？"
    ingestor = KnowledgeIngestor()
    folder = checkpoint.parent / run
    folder.mkdir()
    for version, marker in (("v1","演示甲"),("v2","演示乙")):
        meta = dict(doc_id=doc_id,version=version,topic=topic,source=source,category="process",
                    industry="all",effective_date="2026-10-07")
        text = KnowledgeFeedback.render(meta, f"{query} 演示确认编号是{marker}。仅为工程验收占位知识，不作为真实业务依据。")
        # Explicit conflict fixtures, confined to this unique owned doc_id.
        text = text.replace("status: archived", "status: conflicted")
        parse_metadata(text, "seed.md")
        path = folder / (version + ".md"); path.write_text(text, encoding="utf-8")
        result = ingestor.ingest_file(path, source="process", priority=get_settings().knowledge.sources["process"].priority)
        assert result.status == "indexed", "fixture_ingestion_failed"
    class ScopedStore:
        def hybrid_search(self, *args, filter_expr, **kwargs):
            return ingestor.store.hybrid_search(*args, filter_expr=f'({filter_expr}) and doc_id == "{doc_id}"', **kwargs)
        def get_parents(self, ids):
            return ingestor.store.get_parents(ids)
    retriever = HybridRetriever(store=ScopedStore(), visibility=KnowledgeVisibility(database.session_scope))
    before = retriever.retrieve(query, top_k=10)
    assert before.mode == "conflict" and not before.rerank_degraded, "cloud_conflict_not_verified"
    settings = get_settings()
    original_secret, original_grants = settings.auth.jwt_secret, settings.auth.coach_sales
    settings.auth.jwt_secret = SecretStr(secrets.token_urlsafe(48))
    users = []
    report = dict(run=run,doc_id=doc_id,before_mode=before.mode,
                  backend="real MySQL + Milvus + cloud embedding/rerank",scope="owned demonstration document only")
    try:
        with database.session_scope() as db:
            for role in ("sales","coach","admin"):
                encoded,salt = hash_password(secrets.token_urlsafe(32))
                row = User(username=f"feedback_{run}_{role}",role=role,enabled=True,password_hash=encoded,salt=salt)
                db.add(row); db.flush(); users.append(Principal(row.id,row.username,role))
        sales,coach,admin = users
        settings.auth.coach_sales = {**original_grants,coach.username:[sales.username]}
        headers = {u.id:{"Authorization":"Bearer "+issue_token(u)} for u in users}
        service = PendingService()
        row = service.register("conflict",query,user_id=sales.id,
            context={"items":[i.to_dict() for i in before.items]},
            versions=[dict(doc_id=i.doc_id,version=i.version) for i in before.items])
        path = "/api/v1/knowledge/pending/" + str(row["id"])
        with TestClient(app) as client:
            assert client.patch(path,headers=headers[coach.id],json={"status":"processing"}).status_code == 200
            response = client.patch(path,headers=headers[coach.id],json={"status":"resolved",
                "resolution":"验收演示：人工核实演示乙，先保存草稿，再由管理员单独确认发布。",
                "feedback":dict(action="adopt",adopted=dict(doc_id=doc_id,version="v2"),category="process",industry="all",
                    topic=topic,effective_date="2026-10-07",source=source,
                    content=f"{query} 演示确认编号是演示乙。人工核实的工程验收占位知识，不作为真实业务依据。")})
            assert response.status_code == 200, "draft_save_failed"
            draft_id = response.json()["knowledge_draft"]["id"]
            report.update(draft_id=draft_id,pending_id=row["id"])
            url = "/api/v1/knowledge/drafts/" + draft_id
            draft = client.get(url,headers=headers[admin.id]).json()
            # The fixture content/source above are reviewed engineering test inputs,
            # never model-generated business claims. Confirmation is a separate HTTP action.
            assert draft["metadata"]["source"] == source and "非真实业务制度" in source
            assert "不作为真实业务依据" in draft["content"]
            assert client.post(url+"/publish",headers=headers[coach.id]).status_code == 403
            assert client.post(url+"/publish",headers=headers[admin.id]).status_code == 409
            confirmation = dict(content_hash=draft["content_hash"],content_confirmed=True,source_confirmed=True)
            assert client.post(url+"/confirm",headers=headers[admin.id],json=confirmation).status_code == 200
            published = client.post(url+"/publish",headers=headers[admin.id])
            assert published.status_code == 200, "publication_failed_check_business_audit"
            assert client.post(url+"/publish",headers=headers[admin.id]).json()["status"] == "published"
            audit = client.get(url+"/audit",headers=headers[admin.id]).json()
        after = retriever.retrieve(query,top_k=10)
        version = draft["metadata"]["version"]
        assert after.mode == "rag" and not after.rerank_degraded
        assert after.items and {i.version for i in after.items} == {version}
        assert all(i.status == "active" and "演示乙" in i.parent_content for i in after.items)
        with database.session_scope() as db:
            records = db.scalars(select(KnowledgeDoc).where(KnowledgeDoc.doc_id == doc_id)).all()
            assert {r.version:r.status for r in records} == {"v1":"archived","v2":"archived",version:"active"}
            assert not db.scalars(select(AgentErrorLog).where(AgentErrorLog.path == url+"/publish")).all()
        assert [r["action"] for r in audit] == ["pending_claimed","pending_closed","draft_saved","draft_confirmed",
            "publish_started","ingestion_verified","knowledge_published","versions_archived"]
        report.update(version=version,after_mode=after.mode,retrieved_versions=sorted({i.version for i in after.items}),
                      old_versions_archived=True,audit_actions=[r["action"] for r in audit],idempotent_publish=True,
                      http={"coach_publish":403,"unconfirmed_publish":409,"confirm":200,"publish":200})
        print("[OK] conflict -> owned draft -> human confirmation -> verified publication -> rag; old versions excluded")
    finally:
        settings.auth.jwt_secret,settings.auth.coach_sales = original_secret,original_grants
        if users:
            with database.session_scope() as db:
                db.query(User).filter(User.id.in_([u.id for u in users])).update({User.enabled:False},synchronize_session=False)
    report.update(test_user_ids=[u.id for u in users],test_users_disabled=True)
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    checkpoint.write_text(json.dumps({"run":run,"state":"complete"}),encoding="utf-8")
    print("[OK] eight business audit actions; no publication error; repeat run is read-only")


if __name__ == "__main__":
    try:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--continue-draft", help="Explicit reviewed owned draft ID; never replays seeds/queue writes")
        args = parser.parse_args()
        if args.continue_draft:
            continue_saved(args.continue_draft)
        else:
            main()
    except Exception as exc:
        print("[FAIL] " + type(exc).__name__ + "; inspect saved audit/state before any retry")
        raise SystemExit(1)
