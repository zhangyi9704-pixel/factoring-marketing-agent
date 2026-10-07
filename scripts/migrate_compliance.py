"""PowerShell 项目根目录：.\.venv\Scripts\python.exe scripts\migrate_compliance.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.db.mysql import get_mysql_database
from app.db.compliance_migration import migrate_compliance

if __name__ == "__main__":
    try:
        print("[OK] additive migration: " + (", ".join(migrate_compliance(get_mysql_database().engine)) or "no changes"))
    except Exception as exc:
        print("[FAIL] " + type(exc).__name__)
        raise SystemExit(1)
