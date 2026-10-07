"""Real SQL queue state machine, scopes and competing registrations/claims."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta
import threading
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text, select
from sqlalchemy.orm import Session

from app.models import User, CoachSession, ComplianceLog, KnowledgePending, BusinessAudit
from app.services.auth_service import AuthService, Principal, issue_token
from app.services.pending_service import PendingService, PendingConflict
from app.db.pending_migration import migrate_pending
from app.api.deps import get_auth_service
from app.api.v1.pending import get_pending_service
from app.main import app

SALES = Principal(1,"sales_demo","sales")
COACH = Principal(3,"coach_demo","coach")
OTHER_COACH = Principal(5,"other_coach","coach")
ADMIN = Principal(4,"admin_demo","admin")


@pytest.fixture
def env(tmp_path, monkeypatch):
    engine = create_engine("sqlite:///" + (tmp_path / "pending.db").as_posix(),
                           connect_args={"check_same_thread":False,"timeout":10})
    for model in (User, CoachSession, ComplianceLog):
        model.__table__.create(engine)
    assert "t_knowledge_pending" in migrate_pending(engine)
    assert migrate_pending(engine) == []
    from app.config import get_settings
    monkeypatch.setattr(get_settings().auth,"coach_sales", {"coach_demo":["sales_demo"],"other_coach":["sales_demo"]})

    @contextmanager
    def factory():
        with Session(engine, expire_on_commit=False) as db:
            with db.begin():
                yield db

    with factory() as db:
        for uid,name,role in ((1,"sales_demo","sales"),(2,"sales_b","sales"),(3,"coach_demo","coach"),
                              (4,"admin_demo","admin"),(5,"other_coach","coach"),(6,"unassigned","coach")):
            db.add(User(id=uid, username=name, role=role, enabled=True, password_hash="unused", salt="unused"))
        db.add(CoachSession(id=1,session_id="session_a",user_id=1,case_id="C",status="active",turn_count=0))
        db.add(CoachSession(id=2,session_id="session_b",user_id=2,case_id="C",status="active",turn_count=0))
    now = [datetime(2026,10,6,8)]
    service = PendingService(factory, engine, clock=lambda: now[0])
    yield service, factory, engine, now
    engine.dispose()


def register(service, kind="missing", **kwargs):
    values = dict(user_id=1,session_id="session_a",versions=[{"doc_id":"K","version":"v1"}])
    values.update(kwargs)
    return service.register(kind, values.pop("question","付款方如何确权"), **values)


@pytest.mark.parametrize("kind",["missing","conflict","out_of_scope","compliance"])
def test_dedup_normalized_and_24h_window(env, kind):
    service,_,_,now = env
    first = register(service,kind,question=" Ａ  QUESTION  ")
    assert register(service,kind,question="a question")["id"] == first["id"]
    now[0] += timedelta(hours=24)
    assert register(service,kind,question="a question")["id"] != first["id"]


def test_dedup_distinguishes_owner_session_type_and_version(env):
    service = env[0]
    first = register(service)
    assert register(service,versions=[{"version":"v2"}])["id"] != first["id"]
    assert register(service,"conflict")["id"] != first["id"]
    assert register(service,user_id=2,session_id="session_b")["id"] != first["id"]
    assert register(service,session_id=None)["id"] != first["id"]
    with pytest.raises(PermissionError):
        register(service,user_id=2)


@pytest.mark.parametrize("queue,kind",[("knowledge","missing"),("compliance","compliance")])
@pytest.mark.parametrize("closed",["resolved","dismissed"])
def test_state_machine_and_audit_fields(env,queue,kind,closed):
    service = env[0]
    row = register(service,kind)
    with pytest.raises(PendingConflict):
        service.transition(queue,row["id"],COACH,closed,"conclusion")
    with pytest.raises(ValueError):
        service.transition(queue,row["id"],COACH,"pending")
    claimed = service.transition(queue,row["id"],COACH,"processing")
    assert claimed["assignee"] == COACH.id
    with pytest.raises(PendingConflict):
        service.transition(queue,row["id"],ADMIN,closed,"other reviewer")
    with pytest.raises(ValueError):
        service.transition(queue,row["id"],COACH,closed,"  ")
    done = service.transition(queue,row["id"],COACH,closed," verified conclusion ")
    assert done["status"] == closed and done["resolution"] == "verified conclusion"
    assert done["resolved_at"] and done["assignee"] == COACH.id
    with pytest.raises(PendingConflict):
        service.transition(queue,row["id"],COACH,"processing")
    assert register(service,kind)["id"] == row["id"]  # Closing does not bypass 24h dedup.


@pytest.mark.parametrize("queue,kind",[("knowledge","missing"),("compliance","compliance")])
def test_competing_claims_one_winner(env,queue,kind):
    service = env[0]
    row = register(service,kind)
    barrier = threading.Barrier(2)
    def claim(actor):
        barrier.wait(timeout=5)
        try:
            service.transition(queue,row["id"],actor,"processing")
            return "success"
        except PendingConflict:
            return "conflict"
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim,[COACH,OTHER_COACH]))
    assert sorted(results) == ["conflict","success"]


def test_competing_registration_one_row(env):
    service = env[0]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _:register(service),range(2)))
    assert results[0]["id"] == results[1]["id"]
    assert len(service.list("knowledge",ADMIN)) == 1


def test_list_filters_scopes_and_audit_exclusion(env):
    service,factory,_,now = env
    a = register(service)
    register(service,"conflict",user_id=2,session_id="session_b")
    register(service,"compliance")
    with factory() as db:
        db.add(ComplianceLog(session_id="session_a",rule="CMP-01",raw_text="audit",action="log",created_at=now[0]))
    assert len(service.list("knowledge",ADMIN)) == 2
    assert [r["id"] for r in service.list("knowledge",COACH)] == [a["id"]]
    assert len(service.list("compliance",ADMIN)) == 1
    assert service.list("knowledge",ADMIN,kind="missing",status="processing") == []
    assert service.list("knowledge",ADMIN,since=now[0]+timedelta(seconds=1)) == []
    with pytest.raises(PermissionError):
        service.list("knowledge",SALES)


@pytest.fixture
def client(env):
    service,factory,_,_ = env
    saved = app.dependency_overrides.copy()
    app.dependency_overrides[get_auth_service] = lambda:AuthService(factory)
    app.dependency_overrides[get_pending_service] = lambda:service
    yield TestClient(app)
    app.dependency_overrides.clear()
    app.dependency_overrides.update(saved)


def header(actor):
    return {"Authorization":"Bearer " + issue_token(actor)}


@pytest.mark.parametrize("queue,kind",[("knowledge","missing"),("compliance","compliance")])
def test_api_permissions_and_claim_close(client,env,queue,kind):
    row = register(env[0],kind)
    path = f"/api/v1/{queue}/pending"
    assert client.get(path).status_code == 401
    assert client.get(path,headers=header(SALES)).status_code == 403
    assert client.patch(f'{path}/{row["id"]}',headers=header(SALES),json={"status":"processing"}).status_code == 403
    assert client.get(path,headers=header(COACH)).status_code == 200
    assert client.patch(f'{path}/{row["id"]}',headers=header(COACH),json={"status":"processing"}).status_code == 200
    assert client.patch(f'{path}/{row["id"]}',headers=header(OTHER_COACH),json={"status":"processing"}).status_code == 409
    result = client.patch(f'{path}/{row["id"]}',headers=header(COACH),json={"status":"resolved","resolution":"核实完成"})
    assert result.status_code == 200 and result.json()["resolved_at"]
    assert client.get(f'{path}/{row["id"]}',headers=header(ADMIN)).json()["resolution"] == "核实完成"


def test_api_unauthorized_coach_scope(client,env):
    row = register(env[0],user_id=2,session_id="session_b")
    path = f'/api/v1/knowledge/pending/{row["id"]}'
    assert client.get(path,headers=header(COACH)).status_code == 403
    assert client.patch(path,headers=header(COACH),json={"status":"processing"}).status_code == 403
    assert client.get("/api/v1/knowledge/pending",headers=header(COACH)).json() == []
    assert client.get(path,headers=header(ADMIN)).status_code == 200


def test_api_filters_invalid_range_and_missing(client,env):
    register(env[0])
    assert len(client.get("/api/v1/knowledge/pending?type=missing&status=pending",headers=header(ADMIN)).json()) == 1
    assert client.get("/api/v1/knowledge/pending?status=unknown",headers=header(ADMIN)).status_code == 422
    assert client.get("/api/v1/knowledge/pending?since=2026-10-07&until=2026-10-06",headers=header(ADMIN)).status_code == 422
    assert client.get("/api/v1/knowledge/pending/9999",headers=header(ADMIN)).status_code == 404


def test_retrieval_auto_registration_and_cache_owner_isolation(client,env):
    from app.api.v1.knowledge import get_retriever
    from tests.test_retrieval_branches import build, conflicted
    app.dependency_overrides[get_retriever] = lambda: build([])
    response = client.post("/api/v1/knowledge/search",headers=header(SALES),json={"query":"付款人确权"})
    assert response.status_code == 200 and response.json()["escalation_id"]
    app.dependency_overrides[get_retriever] = lambda: build([conflicted("a","v1"),conflicted("b","v2")])
    response = client.post("/api/v1/knowledge/search",headers=header(SALES),json={"query":"付款人确权"})
    assert response.status_code == 200 and response.json()["mode"] == "conflict"
    rows = env[0].list("knowledge",COACH)
    assert {r["type"] for r in rows} == {"missing","conflict"}
    assert len(rows[0]["context"]["items"]) == 2


def test_compliance_upgrade_durable_without_polluting_audit(env):
    from app.services.session_service import SessionService
    service = SessionService(session_factory=env[1])
    assert service.record_compliance("session_a",1,"CMP-01","warn one",details={"severity":"warn"}) is None
    assert service.record_compliance("session_a",2,"CMP-01","warn two",details={"severity":"warn"}) is None
    identifier = service.record_compliance("session_a",3,"CMP-01","warn three",details={"severity":"warn"})
    assert identifier.startswith("esc_")
    assert len(env[0].list("compliance",ADMIN)) == 1
    assert len(service.list_compliance("session_a")) == 3
    service.record_compliance("session_b",1,"CMP-05","block input",action="block",details={"severity":"block"})
    assert len(env[0].list("compliance",ADMIN)) == 2
    assert len(env[0].list("compliance",COACH)) == 1


def test_shared_retrieval_result_registers_per_jwt_owner(client,env):
    from app.api.v1.knowledge import get_retriever
    from tests.test_retrieval_branches import build
    shared = build([])
    app.dependency_overrides[get_retriever] = lambda:shared
    payload = {"query":"资料缺失的同一问题"}
    assert client.post("/api/v1/knowledge/search",headers=header(SALES),
                       json=dict(payload,user_id=999)).status_code == 422
    response_a = client.post("/api/v1/knowledge/search",headers=header(SALES),json=payload)
    response_b = client.post("/api/v1/knowledge/search",headers=header(Principal(2,"sales_b","sales")),json=payload)
    assert response_a.status_code == response_b.status_code == 200
    first, second = response_a.json(), response_b.json()
    assert first["escalation_id"] != second["escalation_id"]
    assert {r["user_id"] for r in env[0].list("knowledge",ADMIN)} == {1,2}
    assert {r["user_id"] for r in env[0].list("knowledge",COACH)} == {1}


def test_registration_busy_is_redacted_503(client,env):
    from app.api.v1.knowledge import get_retriever
    from tests.test_retrieval_branches import build
    class Busy:
        def register(self,*args,**kwargs):
            raise PendingConflict("registration_busy")
    app.dependency_overrides[get_retriever] = lambda:build([])
    app.dependency_overrides[get_pending_service] = lambda:Busy()
    reply = client.post("/api/v1/knowledge/search",headers=header(SALES),json={"query":"缺失问题"})
    assert reply.status_code == 503
    assert reply.json() == {"detail":"pending_registration_busy"}


def test_migration_preserves_old_compliance_records(tmp_path):
    engine = create_engine("sqlite:///"+(tmp_path/"migration.db").as_posix())
    with engine.begin() as db:
        db.execute(text("CREATE TABLE t_compliance_log (id INTEGER PRIMARY KEY, raw_text TEXT, review_status VARCHAR(16))"))
        db.execute(text("INSERT INTO t_compliance_log VALUES (1, 'existing audit', 'approved')"))
    assert "t_compliance_log.status" in migrate_pending(engine)
    assert migrate_pending(engine) == []
    with engine.connect() as db:
        row = db.execute(text("SELECT raw_text, review_status, status FROM t_compliance_log WHERE id=1")).one()
        assert tuple(row) == ("existing audit","approved",None)
    engine.dispose()


def test_startup_checks_schema_without_migrating(env,monkeypatch):
    from types import SimpleNamespace
    import app.db.mysql as mysql
    import app.db.pending_migration as migration
    import app.main as main
    calls = []
    import app.db.runtime_schema as schema
    def tracked(engine):
        calls.append(engine)
        return []
    monkeypatch.setattr(mysql,"get_mysql_database",lambda:SimpleNamespace(engine=env[2]))
    monkeypatch.setattr(schema,"verify_runtime_schema",tracked)
    monkeypatch.setattr(migration,"migrate_pending",lambda engine: (_ for _ in ()).throw(AssertionError("startup must not migrate")))
    monkeypatch.setattr(main,"setup_logging",lambda:None)
    with TestClient(app):
        pass
    with TestClient(app):
        pass
    assert calls == [env[2],env[2]]


def test_compliance_answer_snapshot_and_legacy_current_reference(client,env):
    row = register(env[0],"compliance",context={"rule":"CMP-01","details":{"compliant_response":"historical answer"}})
    reply = client.get(f'/api/v1/compliance/pending/{row["id"]}',headers=header(COACH))
    assert reply.json()["compliant_response"] == "historical answer"
    assert reply.json()["compliant_response_source"] == "登记时应答快照"
    legacy = register(env[0],"compliance",question="legacy",context={"rule":"CMP-01"})
    reply = client.get(f'/api/v1/compliance/pending/{legacy["id"]}',headers=header(COACH))
    assert reply.json()["compliant_response"]
    assert "原记录未保存应答快照" in reply.json()["compliant_response_source"]
    unknown = register(env[0],"compliance",question="unknown",context={"rule":"UNKNOWN"})
    assert client.get(f'/api/v1/compliance/pending/{unknown["id"]}',headers=header(COACH)).json()["compliant_response"] == ""


@pytest.mark.parametrize("queue,kind", [("knowledge", "missing"), ("compliance", "compliance")])
def test_admin_release_disabled_assignee_and_audited_transfer(client, env, queue, kind):
    service, factory = env[:2]
    row = register(service, kind)
    service.transition(queue, row["id"], COACH, "processing")
    with factory() as db:
        db.get(User, COACH.id).enabled = False
    url = f'/api/v1/{queue}/pending/{row["id"]}/reassign'
    body = dict(expected_assignee=COACH.id, assignee_id=None, reason="领取账号停用，回收待办")
    assert client.post(url, headers=header(OTHER_COACH), json=body).status_code == 403
    result = client.post(url, headers=header(ADMIN), json=body)
    assert result.status_code == 200 and result.json()["status"] == "pending"
    assert result.json()["assignee"] is None
    # A stale admin action cannot overwrite a new claim.
    service.transition(queue, row["id"], OTHER_COACH, "processing")
    assert client.post(url, headers=header(ADMIN), json=body).status_code == 409
    body.update(expected_assignee=OTHER_COACH.id, assignee_id=ADMIN.id, reason="管理员接管处理")
    assert client.post(url, headers=header(ADMIN), json=body).status_code == 200
    service.transition(queue, row["id"], ADMIN, "resolved", "接管后核实完成")
    with factory() as db:
        records = db.scalars(select(BusinessAudit).where(BusinessAudit.pending_id == row["id"],
            BusinessAudit.queue == queue).order_by(BusinessAudit.id)).all()
        assert [r.action for r in records][-3:] == ["pending_claimed", "pending_reassigned", "pending_closed"]
        release = next(r for r in records if r.action == "pending_released")
        assert release.actor_id == ADMIN.id and release.details["previous_assignee"] == COACH.id


def test_transfer_checks_enabled_reviewer_and_current_sales_scope(env):
    service, factory = env[:2]
    row = register(service)
    service.transition("knowledge", row["id"], COACH, "processing")
    with pytest.raises(ValueError, match="invalid_reviewer"):
        service.reassign("knowledge", row["id"], ADMIN, expected_assignee=COACH.id,
                         assignee_id=SALES.id, reason="wrong role")
    with pytest.raises(PermissionError):
        service.reassign("knowledge", row["id"], ADMIN, expected_assignee=COACH.id,
                         assignee_id=6, reason="out of scope")
    with factory() as db:
        db.get(User, OTHER_COACH.id).enabled = False
    with pytest.raises(ValueError, match="invalid_reviewer"):
        service.reassign("knowledge", row["id"], ADMIN, expected_assignee=COACH.id,
                         assignee_id=OTHER_COACH.id, reason="disabled reviewer")
