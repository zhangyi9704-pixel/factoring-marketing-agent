"""Add missing t_user auth columns without changing existing credentials."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.db.auth_migration import migrate_auth
from app.db.mysql import get_mysql_database

if __name__ == "__main__":
    try:
        print("[OK] auth migration: " + (", ".join(migrate_auth(get_mysql_database().engine)) or "no changes"))
    except Exception as exc:
        print("[FAIL] " + type(exc).__name__)
        raise SystemExit(1)
