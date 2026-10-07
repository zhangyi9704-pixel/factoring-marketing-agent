"""One local API closure per explicit mode; sanitized summaries, no graph retries."""
from __future__ import annotations
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def execute(client, mode, report_path, resume_readonly=False):
    from app.services.case_service import CaseService
    from app.schemas.case import assert_no_internal_leak
    from app.graphs.coach_graph import build_redis_checkpointer
    from app.config import get_settings
    from sqlalchemy import select, func
    from app.db.mysql import get_mysql_database
    from app.models import Evaluation, KnowledgeDoc, Case
    report = json.loads(report_path.read_text(encoding="utf-8")) if resume_readonly else {"mode": mode, "status": "running", "steps": []}
    def save():
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    case = CaseService().get_case("INFRA_001")
    secret_facts = [str(x) for x in case.unknown_info + case.hidden_risks if isinstance(x, str)]
    def request(method, path, **kwargs):
        started = time.monotonic()
        response = client.request(method, "/api/v1" + path, **kwargs)
        item = {"method": method, "path": path, "http_status": response.status_code,
                "seconds": round(time.monotonic() - started, 2)}
        report["steps"].append(item)
        save()
        print(json.dumps(item), flush=True)
        if response.status_code >= 400:
            raise RuntimeError("api_http_" + str(response.status_code))
        data = response.json()
        assert_no_internal_leak(data)
        # Exact text values supplement key checks; not a proof of semantic non-leakage.
        serialized = json.dumps(data, ensure_ascii=False)
        forbidden = [v for v in secret_facts if len(v) >= 8 and v in serialized]
        item["internal_keys_absent"] = True
        item["unreleased_exact_fact_matches"] = len(forbidden)
        if forbidden:
            raise RuntimeError("unreleased_fact_leak")
        return data
    try:
        if resume_readonly:
            session_id = report["session_id"]
            user_id = 914001 if mode == "real" else 914002
            with get_mysql_database().session_scope() as db:
                assert db.scalar(select(func.count()).select_from(Evaluation).where(Evaluation.session_id == session_id)) == 1
            report["resumed_readonly"] = True
        else:
            health = request("GET", "/health")
            report["health"] = health
            cases = request("GET", "/cases")
            assert len(cases) == 3
            report["case_count"] = len(cases)
            user_id = 914001 if mode == "real" else 914002
            report['creation_request_id'] = uuid4().hex
            save()  # Preserve the original ID before any request; retries must use it.
            created = request("POST", "/sessions", json={"case_id": case.case_id,
                "request_id": report['creation_request_id'], "user_id": user_id})
            session_id = created["session_id"]
            report["session_id"] = session_id
            report["created"] = {"briefing_present": bool(created.get("briefing")),
                                 "opening_present": bool(created.get("opening", {}).get("content")),
                                 "turn": created.get("turn")}
            save()
            assert report["created"]["briefing_present"] and report["created"]["opening_present"]
            turn = request("POST", f"/sessions/{session_id}/messages", json={"request_id": "baseline-first", "expected_turn": 0, "content": "您好，今天先了解贵公司的业务情况，不涉及额度或审批承诺。"})
            report["turn"] = {"turn": turn["turn"], "reply_present": bool(turn.get("reply")),
                              "finished": turn["finished"]}
            assert turn["turn"] == 1 and report["turn"]["reply_present"]
            finished = request("POST", f"/sessions/{session_id}/finish")
            assert finished["finished"]
        evaluation = request("GET", f"/sessions/{session_id}/evaluation")
        report["evaluation"] = {"keys": sorted(evaluation), "total_score": evaluation.get("total_score"),
                                "error_count": len(evaluation.get("opportunity", {}).get("errors") or []) + len(evaluation.get("sales_score", {}).get("errors") or [])}
        # Repeat the read only; existing evaluation short-circuits without model calls.
        repeated = request("GET", f"/sessions/{session_id}/evaluation")
        differences = [k for k in evaluation if evaluation[k] != repeated.get(k)]
        report["evaluation_read_changes"] = differences
        assert all(k in {"generated_at", "markdown"} for k in differences)
        board = request("GET", f"/users/{user_id}/scores")
        report["dashboard"] = {"keys": sorted(board), "user_id": user_id}
        saver = build_redis_checkpointer(get_settings())
        try:
            checkpoint = saver.get_tuple({"configurable": {"thread_id": session_id}})
            errors = checkpoint.checkpoint["channel_values"].get("errors") or []
            report["checkpoint"] = {"persistent_read": True, "graph_error_count": len(errors),
                                    "graph_error_stages": sorted({e.split(":",1)[0] for e in errors})}
        finally:
            saver._redis.close()
            saver._redis.connection_pool.disconnect()
        with get_mysql_database().session_scope() as db:
            report["initialization_counts"] = {"knowledge_documents": db.scalar(select(func.count()).select_from(KnowledgeDoc)),
                                               "cases": db.scalar(select(func.count()).select_from(Case)),
                                               "session_evaluations": db.scalar(select(func.count()).select_from(Evaluation).where(Evaluation.session_id == session_id))}
        assert report["initialization_counts"]["session_evaluations"] == 1
        report.pop("error_category", None)
        report["status"] = "passed"
        report["model_path"] = "real" if mode == "real" and not errors and not report["evaluation"]["error_count"] else "degraded"
    except Exception as exc:
        report["status"] = "failed"
        report["error_category"] = type(exc).__name__
        # Only controlled smoke assertion messages, never cloud/SQL exception details.
        if str(exc) in {"unreleased_fact_leak"} or str(exc).startswith("api_http_"):
            report["failure"] = str(exc)
    finally:
        save()
    print("RESULT: " + report["status"] + " mode=" + mode, flush=True)
    return int(report["status"] != "passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("real", "degraded"), required=True)
    parser.add_argument("--resume-readonly", action="store_true", help="Continue only reads for an already evaluated session; never create/message/finish again")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    if args.mode == "real":
        import httpx
        with httpx.Client(base_url="http://127.0.0.1:8000", timeout=1200, trust_env=False) as client:
            return execute(client, args.mode, args.report, args.resume_readonly)
    # Isolated in-process test client. Live uvicorn dependencies/config never change.
    from fastapi.testclient import TestClient
    from app.main import app
    from app.api.v1.session import get_roleplay_agent
    from app.api.v1.evaluation import get_evaluation_service
    from app.agents.roleplay.agent import RoleplayAgent
    from app.evaluation.service import EvaluationService
    class UnavailableLLM:
        def chat_json(self, *args, **kwargs):
            raise RuntimeError("intentional_smoke_model_unavailable")
        def chat(self, *args, **kwargs):
            raise RuntimeError("intentional_smoke_model_unavailable")
    class MissingRetriever:
        def retrieve(self, query):
            return SimpleNamespace(missing=True, items=[], message="smoke forced missing context")
    llm = UnavailableLLM()
    agent = RoleplayAgent(llm=llm, retriever=MissingRetriever())
    evaluation = EvaluationService(llm=llm)
    app.dependency_overrides[get_roleplay_agent] = lambda: agent
    app.dependency_overrides[get_evaluation_service] = lambda: evaluation
    try:
        with TestClient(app) as client:
            return execute(client, args.mode, args.report, args.resume_readonly)
    finally:
        app.dependency_overrides.clear()

if __name__ == "__main__":
    raise SystemExit(main())
