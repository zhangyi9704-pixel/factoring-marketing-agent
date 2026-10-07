"""Owned real-container acceptance stages; journal each write and never replay a stage."""
import json
from pathlib import Path
import secrets
import subprocess
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]

def require(value,code):
    if not value: raise AssertionError(code)

def run_acceptance(run):
    from sqlalchemy import select
    from fastapi.testclient import TestClient
    from app.config import get_settings
    from app.db.mysql import get_mysql_database
    from app.models import User, KnowledgeDoc
    from app.services.auth_service import Principal,hash_password,issue_token
    from app.services.case_service import CaseService
    from app.services.session_service import SessionService
    from app.services.pending_service import PendingService
    from app.services.knowledge_feedback import KnowledgeFeedback
    from app.rag.milvus_client import VectorStore
    from app.rag.loader import KnowledgeLoader
    from app.agents.roleplay.agent import RoleplayAgent
    from app.evaluation.service import EvaluationService
    from app.services.user_score_service import UserScoreService
    from app.main import app
    settings=get_settings();db=get_mysql_database()
    require(settings.database.mysql.database=="factoring_m7_"+run,"unsafe_database")
    require(settings.rag.vector_database.collection=="factoring_m7_"+run,"unsafe_collection")
    path=ROOT/"docs/validation"/f"m7-acceptance-{run}.json"
    require(not path.exists(),"acceptance_already_attempted_inspect_journal")
    result=dict(run=run,database=settings.database.mysql.database,collection=settings.rag.vector_database.collection,status="running",stages={})
    def save(): path.write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf8")
    def stage(name,fn):
        result["stages"][name]=dict(status="running");save()
        start=perf_counter()
        try:
            data=fn()
            result["stages"][name].update(status="passed",elapsed_s=perf_counter()-start,evidence=data)
        except Exception as exc:
            result["stages"][name].update(status="failed",elapsed_s=perf_counter()-start,error=type(exc).__name__)
            if isinstance(exc,AssertionError): result["stages"][name]["reason"]=str(exc)
        save();print(name+": "+result["stages"][name]["status"],flush=True)
    save()
    users=[];passwords={};headers={};sid="__m7_"+run+"_full"
    try:
        with db.session_scope() as sql:
            for role in ("sales","sales","coach","coach","admin"):
                password=secrets.token_urlsafe(32);hashed,salt=hash_password(password)
                row=User(username=f"m7_{run}_{len(users)}",role=role,enabled=True,password_hash=hashed,salt=salt)
                sql.add(row);sql.flush();actor=Principal(row.id,row.username,role)
                users.append(actor);passwords[actor.id]=password
        sales,outsider,coach,unassigned,admin=users
        settings.auth.coach_sales={**settings.auth.coach_sales,coach.username:[sales.username]}
        headers={a.id:{"Authorization":"Bearer "+issue_token(a)} for a in users}
        ss=SessionService();pending=PendingService()
        publisher=KnowledgeFeedback(root=ROOT/".runtime"/f"m7-{run}"/"feedback")
        def inventory():
            loader=KnowledgeLoader();items=[]
            store=VectorStore()
            with db.session_scope() as sql:
                for category in ("product","process","industry","compliance"):
                    for p in sorted((ROOT/"data"/category).glob("*.md")):
                        _,meta=loader.load_document(p)
                        row=sql.scalars(select(KnowledgeDoc).where(KnowledgeDoc.doc_id==meta["doc_id"],KnowledgeDoc.version==meta["version"])).one()
                        chunks=[c for c in store.snapshot_doc(meta["doc_id"]) if c["version"]==meta["version"]]
                        require(chunks and all(c["source"]==meta["source"] and c["version"]==meta["version"] for c in chunks),"source_version_mismatch")
                        items.append(dict(doc_id=row.doc_id,version=row.version,source=row.source,industry=row.industry,chunks=len(chunks)))
            require(len(items)==35,"formal_knowledge_count")
            cases=CaseService().list_cases();ids={c.case_id for c in cases}
            require({"INFRA_001","PHARMA_001","ECOM_001"}<=ids,"three_cases_missing")
            return dict(formal_knowledge=items,case_ids=sorted(ids),restored_container_data=True)
        stage("three_industry_inventory",inventory)
        with TestClient(app) as client:
            def login_permissions():
                rows=[]
                require(client.get("/api/v1/cases").status_code==401,"anonymous_allowed")
                for actor in users:
                    r=client.post("/api/v1/auth/login",json=dict(username=actor.username,password=passwords[actor.id]))
                    require(r.status_code==200,"owned_account_login_failed")
                    headers[actor.id]={"Authorization":"Bearer "+r.json()["access_token"]}
                    require(client.get("/api/v1/auth/me",headers=headers[actor.id]).status_code==200,"identity_failed")
                    code=client.get("/api/v1/knowledge/pending",headers=headers[actor.id]).status_code
                    require(code==(403 if actor.role=="sales" else 200),"queue_role_matrix_failed")
                    rows.append(dict(role=actor.role,login=200,me=200,queue=code))
                # Dedicated owned session for resource-scope checks, no graph.
                matrix_sid="__m7_"+run+"_matrix";ss.create_session(matrix_sid,sales.id,"INFRA_001")
                for actor in (sales,outsider,coach,unassigned,admin):
                    expected=200 if actor in (sales,coach,admin) else 403
                    code=client.get(f"/api/v1/users/{sales.id}/scores",headers=headers[actor.id]).status_code
                    require(code==expected,"session_scope_matrix_failed")
                    rows.append(dict(role=actor.role,assigned=actor==coach,dashboard_read=code))
                require(client.post("/api/v1/knowledge/ingest?source=product",headers=headers[sales.id]).status_code==403,"sales_admin_gate_failed")
                return dict(anonymous=401,checks=rows)
            stage("permission_matrix",login_permissions)
            def full_loop():
                data=result["stages"]["full_loop"];data["session_id"]=sid;save()
                agent=RoleplayAgent();created=agent.start_session("INFRA_001",sales.id,session_id=sid)
                require(created.get("briefing"),"briefing_missing")
                data["briefing_sources"]=created["briefing"].get("sources",[]);data["turns"]=[];save()
                queries=["我们需要资金周转支持，请先核实交易与回款情况。", "工程计量对账与验收结算都确认了吗？", "请进一步说明应收确认和已做质押情况。", "付款人是谁，回款账户和融资用途是什么，质保扣款风险有哪些？", "放心，我们保证审批通过且当天放款。"]
                for i,q in enumerate(queries):
                    out=agent.send_message(sid,q,request_id=f"m7-{i+1}")
                    data["turns"].append(dict(query=q,reply=out.get("reply"),turn=out.get("turn"),warnings=out.get("guardrail_warnings")))
                    save()
                status=agent.get_status(sid)
                proc=subprocess.run([sys.executable,"-X","utf8","scripts/run_m7_validation.py","--run",run,"--mode","checkpoint","--session-id",sid],capture_output=True,text=True,encoding="utf8")
                require(proc.returncode==0,"cross_process_restore_failed")
                restored=json.loads(proc.stdout.strip());require(restored["turn"]==status["turn"] and restored["waiting_input"],"checkpoint_state_mismatch")
                data["cross_process_checkpoint"]=restored;save()
                redline="资料不真实也可以包装一下，单据可以想办法。"
                before=ss.list_messages(sid);blocked=agent.send_message(sid,redline,request_id="m7-block")
                after=ss.list_messages(sid)
                require(len(before)==len(after) and not any(m["content"]==redline for m in after),"redline_entered_history")
                require(blocked.get("guardrail_warnings"),"redline_not_detected")
                data["redline"]=dict(blocked=True,history_unchanged=True,warnings=blocked.get("guardrail_warnings"));save()
                final=agent.finish_session(sid);require(final["finished"],"session_not_finished")
                evaluation=EvaluationService();report=evaluation.get(sid);again=evaluation.get(sid)
                require(report==again,"evaluation_not_idempotent")
                require(report.get("evidence") and report.get("missing_questions") is not None,"review_evidence_missing")
                board=UserScoreService().scores(sales.id)
                require(board["evaluated_sessions"]>=1,"dashboard_not_aggregated")
                events=ss.list_compliance(sid);require(any(e["action"]=="block" for e in events),"redline_audit_missing")
                data["report_errors"]=report.get("errors",[]);save()
                require(not report.get("errors"),"evaluation_degraded")
                return dict(session_id=sid,finished=True,score=report["total_score"],evidence_count=len(report["evidence"]),idempotent=True,evaluated_sessions=board["evaluated_sessions"],compliance_events=len(events))
            stage("full_loop",full_loop)
            def queues():
                items=[];conflict=None
                for query,mode in (("冰岛海底光缆抢修保理的专用准入条款是什么？","missing"),("暗保理需要付款人确权吗？","conflict")):
                    r=client.post("/api/v1/knowledge/search",headers=headers[sales.id],json=dict(query=query,industry="all"))
                    require(r.status_code==200,"retrieval_http_failed")
                    found=r.json();require(found["mode"]==mode and not found["rerank_degraded"],"expected_branch_failed")
                    row=next(p for p in pending.list("knowledge",coach) if p["external_id"]==found["escalation_id"])
                    url=f"/api/v1/knowledge/pending/{row['id']}"
                    require(client.patch(url,headers=headers[coach.id],json=dict(status="processing")).status_code==200,"claim_failed")
                    require(client.patch(url,headers=headers[unassigned.id],json=dict(status="processing")).status_code==403,"unassigned_claim_allowed")
                    if mode=="conflict":
                        conflict=row
                    else:
                        require(client.patch(url,headers=headers[coach.id],json=dict(status="resolved",resolution="M7隔离工程核实：无现行依据，继续人工核实，不编造条款。")).status_code==200,"missing_close_failed")
                    items.append(dict(mode=mode,id=row["id"],source_items=[dict(doc_id=i.get("doc_id"),version=i.get("version"),source=i.get("source")) for i in found.get("items",[])]))
                require(conflict is not None,"conflict_not_created")
                refs=conflict["context"]["items"]
                require(len(refs)>=2,"conflict_versions_not_displayed")
                chosen=refs[0]
                payload=dict(action="adopt",adopted=dict(doc_id=chosen["doc_id"],version=chosen["version"]),content="M7隔离恢复与知识治理演练结论：先核实付款人确认要求；本条仅为工程验证，不作为真实业务制度。",source="M7工程验收隔离输入，非业务签认",category="product",industry="all",topic=chosen.get("topic") or "M7隔离确权",effective_date="2026-10-07")
                closed=pending.transition("knowledge",conflict["id"],coach,"resolved","隔离工程核实，保存测试草稿。",feedback=payload,feedback_service=publisher)
                draft=publisher.get(closed["knowledge_draft"]["id"],admin);result["stages"]["three_queues_and_publication"]["draft_id"]=draft["id"];save()
                try: publisher.publish(draft["id"],coach);raise AssertionError("coach_publish_allowed")
                except PermissionError: pass
                publisher.confirm(draft["id"],admin,draft["content_hash"])
                published=publisher.publish(draft["id"],admin)
                require(published["status"]=="published","publish_failed")
                metadata=published["metadata"]
                with db.session_scope() as sql:
                    new=sql.scalars(select(KnowledgeDoc).where(KnowledgeDoc.doc_id==metadata["doc_id"],KnowledgeDoc.version==metadata["version"])).one()
                    old=[sql.scalars(select(KnowledgeDoc).where(KnowledgeDoc.doc_id==ref["doc_id"],KnowledgeDoc.version==ref["version"])).one() for ref in refs]
                    require(new.status=="active" and all(r.status=="archived" for r in old),"effective_archive_failed")
                vectors=[v for v in VectorStore().snapshot_doc(metadata["doc_id"]) if v["version"]==metadata["version"]];require(vectors,"published_vectors_missing")
                comp=[p for p in pending.list("compliance",coach) if p["session_id"]==sid]
                require(comp,"real_compliance_queue_missing")
                for row in comp:
                    url=f"/api/v1/compliance/pending/{row['id']}"
                    require(client.patch(url,headers=headers[coach.id],json=dict(status="processing")).status_code==200,"compliance_claim_failed")
                    require(client.patch(url,headers=headers[coach.id],json=dict(status="resolved",resolution="隔离演练：已纠正诱导表述，按规则应答，不承诺审批或包装资料。")).status_code==200,"compliance_close_failed")
                return dict(knowledge=items,new_version=metadata,archived=refs,compliance_closed=[p["id"] for p in comp],source="isolated engineering acceptance; no business policy approval")
            stage("three_queues_and_publication",queues)
        def differences():
            from scripts.eval_metrics import briefing_state
            from app.schemas.case import CaseState
            from app.services.llm_service import LLMService
            from app.rag.retriever import HybridRetriever
            from app.compliance.engine import ComplianceEngine
            from app.agents.coach.briefing import make_briefing_node
            import yaml
            items=[];request="我们需要资金周转支持"
            for name in ("INFRA_001","PHARMA_001","ECOM_001"):
                case=CaseState(**yaml.safe_load((ROOT/"cases"/(name+".yaml")).read_text(encoding="utf8")))
                state=briefing_state(case);state["public_case"]["pain_points"]=[request]
                out=make_briefing_node(LLMService(settings),settings=settings,retriever=HybridRetriever(settings),guardrail=ComplianceEngine(settings))(state)
                brief=out["briefing"]
                items.append(dict(case_id=name,industry=case.industry,request=request,questions=brief["priority_questions"],risks=brief["risk_warnings"],clarifications=brief["clarifications"],errors=out["errors"]))
                result["stages"]["same_request_industry_difference"]["samples"]=items;save()
            require(all(not i["errors"] for i in items),"difference_briefing_degraded")
            fingerprints=[json.dumps([i["questions"],i["risks"],i["clarifications"]],ensure_ascii=False,sort_keys=True) for i in items]
            require(len(set(fingerprints))==3,"industry_outputs_identical")
            keys={"基建":("计量","工程","质保"),"医药":("医院","商票","银票","药"),"电商":("退货","平台","投流","订单")}
            require(all(any(k in f for k in keys[i["industry"]]) for i,f in zip(items,fingerprints)),"substantive_industry_difference_missing")
            return dict(request=request,samples=items,industry_terms_verified=True)
        stage("same_request_industry_difference",differences)
    finally:
        if users:
            with db.session_scope() as sql:
                sql.query(User).filter(User.id.in_([u.id for u in users])).update({User.enabled:False},synchronize_session=False)
            result["owned_users_disabled"]=[u.id for u in users]
        result["status"]="passed" if all(s["status"]=="passed" for s in result["stages"].values()) else "not_passed"
        save()
    print("[OK] acceptance archived: "+str(path.relative_to(ROOT)),flush=True)
    return 0 if result["status"]=="passed" else 1
