"""Explicit additive creation ledger migration; no model/vector calls."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.db.mysql import get_mysql_database
from app.db.session_migration import migrate_session_requests


if __name__ == '__main__':
    try:
        changes = migrate_session_requests(get_mysql_database().engine)
        print('[OK] session request migration: ' + (', '.join(changes) or 'no changes'))
    except Exception as exc:
        print('[FAIL] session request migration: ' + type(exc).__name__)
        raise SystemExit(1)
