"""Run M7 probes only against a successfully restored owned database/collection."""
import argparse
import json
import logging
import os
from pathlib import Path
import re
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def configure(run):
    if not re.fullmatch(r"[a-f0-9]{12}",run): raise ValueError("invalid_run")
    manifest=json.loads((ROOT/"docs/validation"/f"m7-handover-{run}.json").read_text(encoding="utf8"))
    if manifest["status"] != "passed_logical_restore": raise ValueError("restore_not_verified")
    expected="factoring_m7_"+run
    if manifest["database"] != expected or manifest["collection"] != expected: raise ValueError("target_mismatch")
    os.environ["APP_DATABASE__MYSQL__DATABASE"]=expected
    os.environ["APP_RAG__VECTOR_DATABASE__COLLECTION"]=expected
    from app.rag import revision
    revision.PATH=ROOT/".runtime"/f"m7-{run}"/"knowledge_revision"
    revision.PATH.parent.mkdir(parents=True,exist_ok=True)
    from app.config import get_settings
    get_settings.cache_clear()
    settings=get_settings()
    assert settings.database.mysql.database==expected and settings.rag.vector_database.collection==expected
    return settings

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",required=True)
    p.add_argument("--mode",choices=("integration","metrics","acceptance","checkpoint"),required=True)
    p.add_argument("--session-id")
    p.add_argument("--retry-failed-integration",action="store_true")
    args=p.parse_args()
    configure(args.run)
    logging.disable(logging.CRITICAL)
    if args.mode=="integration":
        import pytest
        nodes=[]
        suffix=""
        if args.retry_failed_integration:
            saved=json.loads((ROOT/"docs/validation/m7-compliance-existing-session.json").read_text(encoding="utf8"))
            if not saved["finished"]: raise ValueError("existing_session_not_finished")
            os.environ["FACTORING_TEST_EXISTING_COMPLIANCE_SESSION"]=saved["session_id"]
            nodes=["tests/test_retriever_integration.py::TestRetrieveIntegration::test_queries_collapse_children_per_parent", "tests/test_roleplay_integration.py::TestRoleplayIntegration::test_compliance_log_written"]
            suffix="-retry"
        return pytest.main(["-q","-m","integration","--tb=no", "--basetemp",f".pytest_tmp_docs_m7_integration_{args.run}{suffix}","--junitxml",f"docs/validation/m7-integration-{args.run}{suffix}.xml",*nodes])
    if args.mode=="metrics":
        from scripts import eval_metrics
        sys.argv=["eval_metrics.py","--repeats","2","--briefing-samples","2"]
        eval_metrics.main()
        return 0
    if args.mode=="checkpoint":
        from app.agents.roleplay.agent import RoleplayAgent
        state=RoleplayAgent().get_status(args.session_id)
        print(json.dumps({k:state.get(k) for k in ("session_id","turn","finished","waiting_input")},ensure_ascii=False))
        return 0
    from scripts.m7_acceptance import run_acceptance
    return run_acceptance(args.run)

if __name__=="__main__":
    try: sys.exit(main())
    except Exception as exc:
        print("[FAIL] M7 probe: "+type(exc).__name__+"; raw details withheld; do not replay graph writes")
        sys.exit(1)
