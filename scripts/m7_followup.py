"""Read-only rechecks and explicit continuation of the already saved owned draft."""
import json
from pathlib import Path
import sys
import logging
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run",required=True);args=p.parse_args()
    from scripts.run_m7_validation import configure
    settings=configure(args.run);logging.disable(logging.CRITICAL)
    from sqlalchemy import select
    from fastapi.testclient import TestClient
    from app.db.mysql import get_mysql_database
    from app.models import User, KnowledgeDoc, KnowledgeDraft
    from app.services.auth_service import Principal,issue_token
    from app.services.pending_service import PendingService
    from app.services.knowledge_feedback import KnowledgeFeedback
    from app.services.session_service import SessionService
    from app.rag.milvus_client import VectorStore
    from app.rag.loader import KnowledgeLoader
    from app.agents.roleplay.agent import RoleplayAgent
    from app.main import app
    from scripts.m7_acceptance import require
    run=args.run;db=get_mysql_database();store=VectorStore()
    source=json.loads((ROOT/"docs/validation"/f"m7-acceptance-{run}.json").read_text(encoding="utf8"))
    require(source["status"]!="running","original_acceptance_still_running")
    path=ROOT/"docs/validation"/f"m7-followup-{run}.json";require(not path.exists(),"followup_already_attempted")
    result=dict(run=run,status="running",stages={})
    def save():path.write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf8")
    def stage(name,fn):
        result["stages"][name]=dict(status="running");save()
        try:result["stages"][name]=dict(status="passed",evidence=fn())
        except Exception as exc:
            result["stages"][name].update(status="failed",error=type(exc).__name__)
            if isinstance(exc,AssertionError):result["stages"][name]["reason"]=str(exc)
        save();print(name+": "+result["stages"][name]["status"],flush=True)
    def inventory():
        items=[]
        with db.session_scope() as sql:
            for category in ("product","process","industry","compliance"):
                for f in sorted((ROOT/"data"/category).glob("*.md")):
                    _,meta=KnowledgeLoader().load_document(f)
                    row=sql.scalars(select(KnowledgeDoc).where(KnowledgeDoc.doc_id==meta["doc_id"],KnowledgeDoc.version==meta["version"])).one()
                    chunks=[c for c in store.snapshot_doc(meta["doc_id"]) if c["version"]==meta["version"]]
                    require(chunks and all(c["source"]==meta["source"] for c in chunks),"source_version_mismatch")
                    items.append(dict(doc_id=row.doc_id,version=row.version,source=row.source,industry=row.industry,status=row.status,chunks=len(chunks)))
        require(len(items)==35,"formal_knowledge_missing")
        from app.services.case_service import CaseService
        require({"INFRA_001","PHARMA_001","ECOM_001"}<={c.case_id for c in CaseService().list_cases()},"cases_missing")
        return dict(formal=items,cases=3)
    stage("three_industry_inventory",inventory)
    with db.session_scope() as sql:
        rows=[sql.scalars(select(User).where(User.username==f"m7_{run}_{i}")).one() for i in range(5)]
        require([r.role for r in rows]==["sales","sales","coach","coach","admin"],"owned_user_mismatch")
        actors=[Principal(r.id,r.username,r.role) for r in rows]
        for row in rows:row.enabled=True
    sales,outsider,coach,unassigned,admin=actors
    settings.auth.coach_sales={**settings.auth.coach_sales,coach.username:[sales.username]}
    headers={a.id:{"Authorization":"Bearer "+issue_token(a)} for a in actors}
    try:
        with TestClient(app) as client:
            def matrix():
                evidence=[]
                require(client.get("/api/v1/cases").status_code==401,"anonymous_allowed")
                for a in actors:
                    require(client.get("/api/v1/cases",headers=headers[a.id]).status_code==200,"case_role_failed")
                    score=client.get(f"/api/v1/users/{sales.id}/scores",headers=headers[a.id]).status_code
                    require(score==(200 if a in (sales,coach,admin) else 403),"score_scope_failed")
                    queue=client.get("/api/v1/knowledge/pending",headers=headers[a.id]).status_code
                    require(queue==(403 if a.role=="sales" else 200),"queue_role_failed")
                    if a.role!="admin":require(client.post("/api/v1/knowledge/ingest?source=product",headers=headers[a.id]).status_code==403,"ingest_role_failed")
                    evidence.append(dict(role=a.role,assigned=a==coach,case_read=200,dashboard=score,queue=queue))
                return dict(anonymous=401,roles=evidence)
            stage("permission_matrix",matrix)
        def continuation():
            draft_id=source["stages"]["three_queues_and_publication"].get("draft_id")
            require(draft_id,"no_owned_draft_do_not_replay_queue_registration")
            publisher=KnowledgeFeedback(root=ROOT/".runtime"/f"m7-{run}"/"feedback")
            draft=publisher.get(draft_id,admin);require(draft["status"]=="draft","draft_state_requires_manual_reconciliation")
            # Only continue the saved draft; no new retrieval, claim or close replay.
            confirmed=publisher.confirm(draft_id,admin,draft["content_hash"])
            require(confirmed["status"]=="confirmed","confirm_failed")
            published=publisher.publish(draft_id,admin);require(published["status"]=="published","publish_failed")
            meta=published["metadata"]
            with db.session_scope() as sql:
                new=sql.scalars(select(KnowledgeDoc).where(KnowledgeDoc.doc_id==meta["doc_id"],KnowledgeDoc.version==meta["version"])).one()
                require(new.status=="active","new_version_not_active")
                old=[]
                for ref in published["supersedes"]:
                    row=sql.scalars(select(KnowledgeDoc).where(KnowledgeDoc.doc_id==ref["doc_id"],KnowledgeDoc.version==ref["version"])).one()
                    require(row.status=="archived","old_version_not_archived")
                    old.append(dict(doc_id=row.doc_id,version=row.version,status=row.status))
            require(any(c["version"]==meta["version"] for c in store.snapshot_doc(meta["doc_id"])),"new_vectors_missing")
            pending=PendingService();sid="__m7_"+run+"_full"
            comp=[r for r in pending.list("compliance",coach) if r["session_id"]==sid]
            require(comp,"compliance_queue_missing")
            for row in comp:
                require(row["status"]=="pending","unexpected_compliance_state")
                pending.transition("compliance",row["id"],coach,"processing")
                closed=pending.transition("compliance",row["id"],coach,"resolved","M7隔离演练纠正诱导表述，保留标准合规应答与审计。")
                require(closed["status"]=="resolved" and closed["resolved_at"],"compliance_not_closed")
            with db.session_scope() as sql:
                from app.models import KnowledgePending
                knowledge=list(sql.scalars(select(KnowledgePending).where(KnowledgePending.user_id==sales.id,KnowledgePending.type.in_(["missing","conflict"]))))
                require(any(r.type=="missing" and r.status=="resolved" for r in knowledge),"missing_not_closed")
                require(any(r.type=="conflict" and r.status=="resolved" for r in knowledge),"conflict_not_closed")
            return dict(draft_id=draft_id,new_version=meta,archived=old,compliance_closed=[r["id"] for r in comp],no_graph_replay=True,only_engineering_test_policy=True)
        stage("three_queues_and_publication",continuation)
        def quality():
            snapshot=RoleplayAgent()._snapshot("__m7_"+run+"_full").values
            errors=snapshot.get("errors",[])
            evidence=dict(finished=snapshot.get("finished"),errors=errors,released_count=len(snapshot.get("released_info",[])),blocked_turns=snapshot.get("blocked_turns"))
            result["stages"]["full_loop_quality"]["evidence"]=evidence;save()
            require(not errors and evidence["released_count"]>0,"full_loop_model_safety_fallback")
            return evidence
        stage("full_loop_quality",quality)
    finally:
        with db.session_scope() as sql:sql.query(User).filter(User.id.in_([a.id for a in actors])).update({User.enabled:False},synchronize_session=False)
        result["owned_users_disabled"]=[a.id for a in actors]
        result["status"]="passed" if all(s["status"]=="passed" for s in result["stages"].values()) else "not_passed";save()
    print("[OK] followup archived; no graph replay")
    return 0 if result["status"]=="passed" else 1

if __name__=="__main__":
    try:sys.exit(main())
    except Exception as exc:print("[FAIL] "+type(exc).__name__+"; details withheld, inspect journal");sys.exit(1)
