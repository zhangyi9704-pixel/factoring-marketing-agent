"""Real MySQL/cloud/API acceptance; creates owned test users and disables them.

PowerShell project root: .\.venv\Scripts\python.exe scripts\verify_pending.py
No graph execution, password/token output, demo-account change or data deletion.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import logging
import secrets
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from pydantic import SecretStr
from app.config import get_settings
from app.db.mysql import get_mysql_database
from app.db.pending_migration import migrate_pending
from app.models import User
from app.services.auth_service import Principal, hash_password, issue_token
from app.services.pending_service import PendingService
from app.main import app


def main():
    logging.disable(logging.CRITICAL)  # This probe prints only its sanitized results.
    database = get_mysql_database()
    output = Path("docs/validation/4.1-pending-live.json")
    if output.exists():
        # Repeating the beginner's command only checks saved evidence. No new
        # users, cloud request, registration, claim or close is performed.
        saved = json.loads(output.read_text(encoding="utf-8"))
        service = PendingService()
        for item in saved["items"]:
            row = service.get("knowledge",item["id"],Principal(1,"verification","admin"))
            assert row and row["external_id"] == item["external_id"]
            assert row["status"] == item["status"] and row["resolved_at"] and row["resolution"]
        ids = {i["user_id"] for i in saved["items"]} | {i["assignee"] for i in saved["items"]}
        with database.session_scope() as db:
            assert all(not u.enabled for u in db.query(User).filter(User.id.in_(ids)))
        print("[OK] saved missing/conflict closed; test users disabled; read-only, no new records")
        return
    migrate_pending(database.engine)
    assert migrate_pending(database.engine) == []
    settings = get_settings()
    original_secret, original_grants = settings.auth.jwt_secret, settings.auth.coach_sales
    # Test-process-only JWT issuer; does not change .env or the running application.
    settings.auth.jwt_secret = SecretStr(secrets.token_urlsafe(48))
    run = uuid.uuid4().hex[:12]
    users, report = [], {"run":run, "backend":"MySQL + cloud embedding/rerank", "items":[]}
    try:
        with database.session_scope() as db:
            for role in ("sales", "coach", "coach"):
                password_hash, salt = hash_password(secrets.token_urlsafe(32))
                row = User(username=f"pending_{run}_{len(users)}", role=role, enabled=True,
                           password_hash=password_hash, salt=salt)
                db.add(row)
                db.flush()
                users.append(Principal(row.id, row.username, role))
        sales, coach, other_coach = users
        settings.auth.coach_sales = {**original_grants,
            coach.username:[sales.username], other_coach.username:[sales.username]}
        headers = {actor.id:{"Authorization":"Bearer "+issue_token(actor)} for actor in users}
        service = PendingService()
        with TestClient(app) as client:
            assert client.get("/api/v1/knowledge/pending").status_code == 401
            assert client.get("/api/v1/knowledge/pending",headers=headers[sales.id]).status_code == 403
            for question, expected, closed in (
                ("冰岛海底光缆抢修保理的专用准入条款是什么？","missing","resolved"),
                ("暗保理需要付款人确权吗？","conflict","dismissed"),
            ):
                response = client.post("/api/v1/knowledge/search",headers=headers[sales.id],
                                       json={"query":question,"industry":"all"})
                assert response.status_code == 200, "retrieval_http_failed"
                result = response.json()
                assert result["mode"] == expected, "unexpected_retrieval_branch"
                assert not result["rerank_degraded"], "cloud_rerank_degraded"
                listed = client.get("/api/v1/knowledge/pending?status=pending",headers=headers[coach.id])
                assert listed.status_code == 200
                row = next(r for r in listed.json() if r["external_id"] == result["escalation_id"])
                assert row["user_id"] == sales.id and row["type"] == expected
                # Exercise MySQL named-lock registration, without another cloud call.
                def duplicate(_):
                    return service.register(expected,question,user_id=sales.id,
                        versions=row["context"]["versions"],context=row["context"])["id"]
                with ThreadPoolExecutor(max_workers=2) as pool:
                    assert list(pool.map(duplicate,range(2))) == [row["id"],row["id"]]
                path = f'/api/v1/knowledge/pending/{row["id"]}'
                def claim(actor):
                    reply = client.patch(path,headers=headers[actor.id],json={"status":"processing"})
                    return actor, reply.status_code
                with ThreadPoolExecutor(max_workers=2) as pool:
                    attempts = list(pool.map(claim,[coach,other_coach]))
                assert sorted(code for _,code in attempts) == [200,409]
                winner = next(actor for actor,code in attempts if code == 200)
                done = client.patch(path,headers=headers[winner.id],json={"status":closed,
                    "resolution":"本地验收确认登记、范围过滤、原子领取与关闭；知识内容处理留待正式审核。"})
                assert done.status_code == 200
                final = done.json()
                assert final["assignee"] == winner.id and final["resolved_at"] and final["resolution"]
                report["items"].append({key:final[key] for key in
                    ("id","external_id","type","status","assignee","created_at","resolved_at","user_id")})
                report["items"][-1].update(claim_http=[code for _,code in attempts],dedup=True)
                print(f'[OK] {expected}: id={row["id"]}; visible; dedup; claim=200/409; close={closed}')
        report["http"] = {"anonymous":401,"sales_pending":403,"authorized_coach":200}
    finally:
        settings.auth.jwt_secret, settings.auth.coach_sales = original_secret, original_grants
        if users:
            with database.session_scope() as db:
                db.query(User).filter(User.id.in_([u.id for u in users])).update(
                    {User.enabled:False},synchronize_session=False)
    report["test_users_disabled"] = True
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"[OK] users disabled; artifact={output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("[FAIL] " + type(exc).__name__)
        raise SystemExit(1)
