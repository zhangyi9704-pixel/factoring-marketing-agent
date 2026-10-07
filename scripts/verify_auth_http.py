"""Real curl.exe HTTP acceptance, isolated in-memory SQL; no business writes.

Passwords/JWT exist only in this process and curl stdin. Use --live instead to
test running MySQL-backed app with APP_DEMO_SALES_PASSWORD, without provisioning.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import threading
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def curl(url, *, payload=None, token=None):
    # Curl config goes over stdin so tokens are absent from process arguments.
    config = f'url = "{url}"\nsilent\nshow-error\nmax-time = 10\nwrite-out = "\\n%{{http_code}}"\n'
    if payload is not None:
        data = json.dumps(json.dumps(payload))
        config += f'request = "POST"\nheader = "Content-Type: application/json"\ndata = {data}\n'
    if token:
        config += f'header = "Authorization: Bearer {token}"\n'
    result = subprocess.run(["curl.exe", "--config", "-"], input=config,
                            text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError("curl_transport_failed")
    body, code = result.stdout.rsplit("\n", 1)
    return int(code), json.loads(body)


def verify(base, password):
    status, _ = curl(base + "/auth/me")
    assert status == 401, "missing_token"
    print("[OK] curl no token: 401")
    status, result = curl(base + "/auth/login", payload={"username": "sales_demo", "password": password})
    assert status == 200, "login_failed"
    token = result["access_token"]
    status, result = curl(base + "/auth/me", token=token)
    assert status == 200 and result["username"] == "sales_demo", "valid_token"
    print("[OK] curl valid token: 200")
    status, _ = curl(base + "/knowledge/ingest?source=product", payload={}, token=token)
    assert status == 403, "publication_role_gate"
    print("[OK] curl sales -> knowledge administration: 403")


def isolated():
    import uvicorn
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from sqlalchemy.pool import StaticPool
    from app.config import get_settings
    from app.models import User
    from app.main import app
    from app.api.deps import get_auth_service
    from app.services.auth_service import AuthService, hash_password
    from pydantic import SecretStr

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    User.__table__.create(engine)
    password = secrets.token_urlsafe(24)
    encoded, salt = hash_password(password)
    with Session(engine) as db:
        db.add(User(id=1, username="sales_demo", role="sales", password_hash=encoded, salt=salt, enabled=True))
        db.commit()

    @contextmanager
    def factory():
        with Session(engine) as db:
            with db.begin():
                yield db

    get_settings().auth.jwt_secret = SecretStr(secrets.token_urlsafe(32))
    app.dependency_overrides[get_auth_service] = lambda: AuthService(factory)
    # Avoid production file/error audit logging in this disposable verifier.
    from contextlib import asynccontextmanager
    @asynccontextmanager
    async def lifespan(_):
        yield
    app.router.lifespan_context = lifespan
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="critical", access_log=False))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("server_start_timeout")
            time.sleep(.05)
        verify(f"http://127.0.0.1:{port}/api/v1", password)
        print("[OK] isolated SQLite HTTP acceptance; no MySQL/demo account writes")
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        listener.close()
        engine.dispose()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/api/v1")
    args = parser.parse_args()
    try:
        if args.live:
            password = os.environ.get("APP_DEMO_SALES_PASSWORD")
            if not password:
                raise RuntimeError("APP_DEMO_SALES_PASSWORD_missing")
            verify(args.base_url.rstrip("/"), password)
        else:
            isolated()
        return 0
    except Exception as exc:
        print("[FAIL] " + type(exc).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
