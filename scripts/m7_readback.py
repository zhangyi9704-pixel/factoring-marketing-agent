"""Final read-only integrity checks plus scoped authenticated search on restored data."""
import argparse,json,logging,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser();p.add_argument("--run",required=True);args=p.parse_args()
    from scripts.run_m7_validation import configure
    settings=configure(args.run);logging.disable(logging.CRITICAL)
    from app.config import Settings
    from app.db.mysql import MySQLDatabase,get_mysql_database
    from app.rag.milvus_client import VectorStore
    from scripts.verify_handover import sql_fingerprints,canonical
    from app.models import User
    from app.services.auth_service import Principal,issue_token
    from fastapi.testclient import TestClient
    from app.main import app
    from sqlalchemy import select
    report=json.loads((ROOT/"docs/validation"/f"m7-handover-{args.run}.json").read_text(encoding="utf8"))
    primary=settings.model_copy(deep=True);primary.database.mysql.database="factoring_coach";primary.rag.vector_database.collection="factoring_knowledge_v2"
    main_db_same=sql_fingerprints(MySQLDatabase(primary).engine)==report["stages"]["mysql_restore"]["tables"]
    source=VectorStore(primary);fields=[f.name for f in source.build_schema(source.dimension).fields]
    old=json.loads((ROOT/report["backup"]/"milvus-logical.json").read_text(encoding="utf8"))
    actual=source.client.query(collection_name=source.collection_name,filter='id != ""',output_fields=fields,limit=len(old)+1,consistency_level="Strong")
    actual.sort(key=lambda r:r["id"]);old.sort(key=lambda r:r["id"])
    for row in old:row["sparse_vector"]={int(k):v for k,v in row["sparse_vector"].items()}
    main_vector_same=canonical(actual)==canonical(old)
    backup=ROOT/report["backup"]
    (backup/"milvus-schema.json").write_text(canonical(dict(version=source.ping(),dimension=source.dimension,schema=source.client.describe_collection(collection_name=source.collection_name))),encoding="utf8")
    db=get_mysql_database()
    with db.session_scope() as sql:
        row=sql.scalars(select(User).where(User.username==f"m7_{args.run}_0")).one()
        if row.enabled:raise ValueError("test_user_unexpectedly_enabled")
        row.enabled=True;actor=Principal(row.id,row.username,row.role)
    output=dict(run=args.run,source_mysql_unchanged=main_db_same,source_vectors_unchanged=main_vector_same)
    try:
        with TestClient(app) as client:
            health=client.get("/api/v1/health");obj=health.json()
            output["health"]=dict(http=health.status_code,dependencies_available=obj["dependencies_available"],models_verified=obj["models_verified"],components={k:v["status"] for k,v in obj["components"].items()})
            headers={"Authorization":"Bearer "+issue_token(actor)}
            found=client.post("/api/v1/knowledge/search",headers=headers,json=dict(query="保理费率由哪些部分构成，哪些因素影响费率？",industry="all"))
            answer=found.json();output["source_echo"]=dict(http=found.status_code,mode=answer["mode"],items=[{k:i.get(k) for k in ("doc_id","source","version","category","industry")} for i in answer["items"]])
            if found.status_code!=200 or answer["mode"]!="rag":raise ValueError("source_echo_failed")
            if not obj["dependencies_available"]:raise ValueError("restored_health_degraded")
    finally:
        with db.session_scope() as sql:sql.get(User,actor.id).enabled=False
    output["test_user_disabled"]=True
    path=ROOT/"docs/validation"/f"m7-readback-{args.run}.json";path.write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps(output,ensure_ascii=False))
    return 0 if main_db_same and main_vector_same else 1

if __name__=="__main__":
    try:sys.exit(main())
    except Exception as exc:print("[FAIL] "+type(exc).__name__+"; details withheld");sys.exit(1)
