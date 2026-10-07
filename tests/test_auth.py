"""Exercise actual auth/SQL ownership with a local database; no security overrides."""
from contextlib import contextmanager
import time

import jwt
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.main import app
from app.api.deps import (get_auth_service, get_ownership_service, require_review_scope, require_reviewer)
from app.api.v1.case import get_case_service
from app.api.v1.session import get_roleplay_agent
from app.api.v1.knowledge import get_ingestor
from app.config import get_settings
from app.db.auth_migration import migrate_auth
from app.models import User, CoachSession
from app.services.auth_service import (AuthService, Principal, hash_password, verify_password,
                                     issue_token, decode_token, jwt_secret, TOKEN_SECONDS)
from app.services.session_service import SessionService
from scripts.init_demo_users import provision


@pytest.fixture(scope="module")
def password_hash():
    return hash_password("unit-test-password")


@pytest.fixture
def database(password_hash):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE t_user (id INTEGER PRIMARY KEY AUTOINCREMENT, username VARCHAR(64) UNIQUE NOT NULL, password_hash VARCHAR(128) NOT NULL, role VARCHAR(16) NOT NULL, team VARCHAR(64), created_at DATETIME DEFAULT CURRENT_TIMESTAMP)"))
    assert migrate_auth(engine) == ["t_user.salt", "t_user.enabled"]
    assert migrate_auth(engine) == []
    CoachSession.__table__.create(engine)

    @contextmanager
    def factory():
        with Session(engine, expire_on_commit=False) as db:
            with db.begin():
                yield db

    encoded, salt = password_hash
    with factory() as db:
        for uid, name, role, enabled in ((1, "sales_demo", "sales", True),
                (2, "sales_b", "sales", True), (3, "coach_demo", "coach", True),
                (4, "admin_demo", "admin", True), (5, "disabled", "sales", False),
                (6, "unassigned_coach", "coach", True)):
            db.add(User(id=uid, username=name, role=role, enabled=enabled, password_hash=encoded, salt=salt))
        db.add(CoachSession(id=1, session_id="owned_a", user_id=1, case_id="C", phase="roleplay", status="active", turn_count=0))
        db.add(CoachSession(id=2, session_id="owned_b", user_id=2, case_id="C", phase="roleplay", status="active", turn_count=0))
    yield engine, factory
    engine.dispose()


@pytest.fixture
def client(database):
    _, factory = database
    overrides = app.dependency_overrides.copy()
    app.dependency_overrides[get_auth_service] = lambda: AuthService(factory)
    app.dependency_overrides[get_ownership_service] = lambda: SessionService(session_factory=factory)
    class Cases:
        def list_cases(self, industry):
            return []
    app.dependency_overrides[get_case_service] = Cases
    yield TestClient(app)
    app.dependency_overrides.clear()
    app.dependency_overrides.update(overrides)


def headers(uid=1, role="sales", **claims):
    user = Principal(uid, "unused", role)
    token = issue_token(user)
    if claims:
        payload = decode_token(token)
        payload.update(claims)
        token = jwt.encode(payload, jwt_secret(), algorithm="HS256")
    return {"Authorization": "Bearer " + token}


def test_login_success_and_claims(client):
    result = client.post("/api/v1/auth/login", json={"username": "sales_demo", "password": "unit-test-password"})
    assert result.status_code == 200
    data = result.json()
    claims = decode_token(data["access_token"])
    assert claims["sub"] == "1" and claims["role"] == "sales"
    assert claims["exp"] - claims["iat"] == TOKEN_SECONDS == data["expires_in"]
    assert jwt.get_unverified_header(data["access_token"])["alg"] == "HS256"


@pytest.mark.parametrize("username,password", [("sales_demo", "wrong"), ("unknown", "wrong"), ("disabled", "unit-test-password")])
def test_login_failure(client, username, password):
    response = client.post("/api/v1/auth/login", json={"username": username, "password": password})
    assert response.status_code == 401 and response.json() == {"detail": "invalid_credentials"}


@pytest.mark.parametrize("path,method", [("/cases", "get"), ("/sessions", "post"),
    ("/sessions/owned_a/resume", "post"), ("/sessions/owned_a/messages/stream", "post"),
    ("/sessions/owned_a/evaluation/stream", "get"), ("/knowledge/ingest", "post"),
    ("/users/1/scores/stream", "get"), ("/knowledge/search", "post")])
def test_no_token_401(client, path, method):
    response = getattr(client, method)("/api/v1" + path)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_valid_token_and_static(client):
    assert client.get("/api/v1/cases", headers=headers()).status_code == 200
    assert client.get("/").status_code == 200
    assert client.get("/static/auth.js").status_code == 200


@pytest.mark.parametrize("claims", [{"exp": int(time.time()) - 1}, {"user_id": 2}, {"sub": "x"}, {"role": "manager"}, {"exp": None}, {"iat": True}])
def test_bad_claims(client, claims):
    assert client.get("/api/v1/cases", headers=headers(**claims)).status_code == 401


@pytest.mark.parametrize("token", ["garbage", jwt.encode({"sub": "1", "role": "sales", "iat": 1, "exp": 2}, "wrong-key" * 4, algorithm="HS256"), jwt.encode({"sub": "1"}, "", algorithm="none")])
def test_invalid_signature_algorithm_missing_claims(client, token):
    assert client.get("/api/v1/cases", headers={"Authorization": "Bearer " + token}).status_code == 401


def test_disabled_or_changed_role_token(client, database):
    token = headers()
    with database[1]() as db:
        db.get(User, 1).enabled = False
    assert client.get("/api/v1/cases", headers=token).status_code == 401
    assert client.get("/api/v1/cases", headers=headers(99)).status_code == 401
    assert client.get("/api/v1/cases", headers=headers(1, "admin")).status_code == 401


def test_role_change_invalidates_existing_token(client, database):
    token = headers()
    with database[1]() as db:
        db.get(User, 1).role = "coach"
    assert client.get("/api/v1/cases", headers=token).status_code == 401


def test_login_validation_does_not_echo_password(client):
    password = "sensitive" * 150
    response = client.post("/api/v1/auth/login", json={"username": "sales_demo", "password": password})
    assert response.status_code == 422
    assert password not in response.text and response.json() == {"detail": "invalid_login_payload"}


@pytest.mark.parametrize("path,method", [("messages", "post"), ("messages/stream", "post"),
    ("resume", "post"), ("finish", "post"), ("evaluation", "get"), ("evaluation/stream", "get")])
def test_sales_a_cannot_access_b_before_business_work(client, path, method):
    response = getattr(client, method)("/api/v1/sessions/owned_b/" + path, headers=headers(),
        **({"json": {"content": "hello", "request_id": "auth-test", "expected_turn": 0}} if method == "post" else {}))
    assert response.status_code == 403
    assert response.json()["detail"] == "session_owner_forbidden"


def test_creation_ignores_forged_user_id(client):
    from tests.test_api_session import StubAgent
    app.dependency_overrides[get_roleplay_agent] = StubAgent
    response = client.post("/api/v1/sessions", headers=headers(), json={"case_id": "C", "request_id": "auth-create", "user_id": 2})
    assert response.status_code == 201 and response.json()["user_id"] == 1


def test_owner_resume_allowed_and_missing_404(client):
    from tests.test_api_session import StubAgent
    app.dependency_overrides[get_roleplay_agent] = StubAgent
    assert client.post("/api/v1/sessions/owned_a/resume", headers=headers()).status_code == 200
    assert client.post("/api/v1/sessions/missing/resume", headers=headers()).status_code == 404


@pytest.mark.parametrize("uid,role", [(1, "sales"), (3, "coach")])
def test_knowledge_publication_only_admin(client, uid, role):
    assert client.post("/api/v1/knowledge/ingest?source=product", headers=headers(uid, role),
                       files={"file": ("test.md", b"test")}).status_code == 403


def test_admin_passes_publication_gate(client):
    # Invalid source proves auth succeeded without invoking model/vector writes.
    app.dependency_overrides[get_ingestor] = lambda: object()
    assert client.post("/api/v1/knowledge/ingest?source=invalid", headers=headers(4, "admin"),
                       files={"file": ("test.md", b"test")}).status_code == 400


@pytest.mark.parametrize("uid,role,target,allowed", [(1,"sales",1,True),(1,"sales",2,False),
    (3,"coach",1,True),(3,"coach",2,False),(6,"coach",1,False),(4,"admin",2,True)])
def test_dashboard_scope(client, uid, role, target, allowed):
    from app.api.v1.user import get_user_score_service
    from tests.test_api_user_scores import StubService
    app.dependency_overrides[get_user_score_service] = StubService
    response = client.get(f"/api/v1/users/{target}/scores", headers=headers(uid,role))
    assert response.status_code == (200 if allowed else 403)


@pytest.mark.parametrize("uid,role,target,status", [(1,"sales",1,403),(3,"coach",1,200),
    (3,"coach",2,403),(6,"coach",1,403),(4,"admin",2,200)])
def test_review_claim_dependency(database, uid, role, target, status):
    # M4 endpoint contract: the owner is loaded from the record server-side.
    review_app = FastAPI()
    review_app.dependency_overrides[get_auth_service] = lambda: AuthService(database[1])
    @review_app.post("/claim/{record_id}")
    def claim(record_id: int, user=Depends(authorize_record)):
        return {"claimed": True}
    response = TestClient(review_app).post(f"/claim/{target}", headers=headers(uid, role))
    assert response.status_code == status


def authorize_record(record_id: int, user=Depends(require_reviewer),
                     service=Depends(get_auth_service)):
    owners = {1: 1, 2: 2}  # Stub pending repository, not caller-provided user_id.
    return require_review_scope(owners[record_id], user, service)


def test_password_salts_and_no_plaintext(password_hash):
    other = hash_password("unit-test-password")
    assert other != password_hash
    assert verify_password("unit-test-password", *other)
    assert not verify_password("wrong", *other)
    assert not verify_password("any", "legacy", None)
    assert "unit-test-password" not in other[0]


def test_secret_missing_refuses_startup(monkeypatch):
    monkeypatch.setattr(get_settings().auth, "jwt_secret", None)
    with pytest.raises(RuntimeError, match="APP_AUTH__JWT_SECRET"):
        with TestClient(app):
            pass


def test_demo_script_idempotent_preserves_existing(database):
    _, factory = database
    assert provision(factory, {"sales": "unused", "coach": "unused", "admin": "unused"}) == 0
    with factory() as db:
        db.execute(text("DELETE FROM t_user WHERE username = 'admin_demo'"))
    assert provision(factory, {"sales": "new", "coach": "new", "admin": "new"}) == 1
    with factory() as db:
        admin = db.execute(select(User).where(User.username == "admin_demo")).scalar_one()
        before = (admin.password_hash, admin.salt)
        assert verify_password("new", *before)
    assert provision(factory, {"sales": "changed", "coach": "changed", "admin": "changed"}) == 0
    with factory() as db:
        admin = db.execute(select(User).where(User.username == "admin_demo")).scalar_one()
        assert (admin.password_hash, admin.salt) == before


def test_explicit_demo_reset_preserves_identity_status_and_other_accounts(database):
    _, factory = database
    with factory() as db:
        demo_users = list(db.scalars(select(User).where(User.username.in_(["sales_demo", "coach_demo", "admin_demo"]))))
        admin = next(user for user in demo_users if user.username == "admin_demo")
        admin.enabled = False
        identities = {user.username: (user.id, user.role, user.enabled) for user in demo_users}
        other = db.execute(select(User).where(User.username == "sales_b")).scalar_one()
        other_password = (other.password_hash, other.salt)
    assert provision(factory, {role: "12345" for role in ("sales", "coach", "admin")}, reset_existing_passwords=True) == 0
    with factory() as db:
        for user in db.scalars(select(User).where(User.username.in_(identities))):
            assert verify_password("12345", user.password_hash, user.salt)
            assert (user.id, user.role, user.enabled) == identities[user.username]
            assert "12345" not in user.password_hash
        other = db.execute(select(User).where(User.username == "sales_b")).scalar_one()
        assert (other.password_hash, other.salt) == other_password
