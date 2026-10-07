"""PowerShell project root: .\.venv\Scripts\python.exe scripts\migrate_pending.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.db.mysql import get_mysql_database
from app.db.pending_migration import migrate_pending

if __name__ == "__main__":
    try:
        print("[OK] pending migration: " + (", ".join(migrate_pending(get_mysql_database().engine)) or "no changes"))
    except Exception as exc:
        print("[FAIL] " + type(exc).__name__)
        raise SystemExit(1)
