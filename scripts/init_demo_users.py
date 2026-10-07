"""Initialize demo accounts; explicit reset affects their passwords only."""
import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select
from app.db.auth_migration import migrate_auth
from app.db.mysql import get_mysql_database
from app.models import User
from app.services.auth_service import hash_password

DEMO_USERS = (("sales_demo", "sales"), ("coach_demo", "coach"), ("admin_demo", "admin"))
DEFAULT_DEMO_PASSWORD = "12345"


def provision(session_factory, passwords, *, reset_existing_passwords=False):
    created = 0
    with session_factory() as db:
        for username, role in DEMO_USERS:
            existing = db.execute(select(User).where(User.username == username)).scalar_one_or_none()
            if existing:
                if reset_existing_passwords:
                    existing.password_hash, existing.salt = hash_password(passwords[role])
                # Account identity, role and enabled state are always preserved.
                continue
            encoded, salt = hash_password(passwords[role])
            db.add(User(username=username, password_hash=encoded, salt=salt, role=role, enabled=True))
            created += 1
    return created


def main():
    parser = argparse.ArgumentParser()
    for _, role in DEMO_USERS:
        parser.add_argument(f"--{role}-password", default=os.environ.get(f"APP_DEMO_{role.upper()}_PASSWORD", DEFAULT_DEMO_PASSWORD))
    parser.add_argument("--reset-passwords", action="store_true", help="Reset only the three existing demo account passwords; preserve roles and enabled state")
    args = parser.parse_args()
    passwords = {role: getattr(args, f"{role}_password") for _, role in DEMO_USERS}
    if not all(passwords.values()):
        parser.error("set APP_DEMO_SALES_PASSWORD / APP_DEMO_COACH_PASSWORD / APP_DEMO_ADMIN_PASSWORD or corresponding arguments")
    try:
        database = get_mysql_database()
        changes = migrate_auth(database.engine)
        count = provision(database.session_scope, passwords, reset_existing_passwords=args.reset_passwords)
        existing_status = "demo passwords reset; roles/status preserved" if args.reset_passwords else "existing preserved"
        print(f"[OK] auth migration: {', '.join(changes) or 'no changes'}; created={count}; {existing_status}")
    except Exception as exc:
        print("[FAIL] " + type(exc).__name__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
