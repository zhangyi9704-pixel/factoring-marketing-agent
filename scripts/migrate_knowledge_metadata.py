"""项目根目录 PowerShell：.\.venv\Scripts\python.exe scripts\migrate_knowledge_metadata.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.db.mysql import get_mysql_database
from app.db.knowledge_migration import migrate_knowledge_metadata

if __name__ == "__main__":
    try:
        changes = migrate_knowledge_metadata(get_mysql_database().engine)
        print("[OK] knowledge metadata migration: " + (", ".join(changes) or "no changes"))
    except Exception as exc:
        print("[FAIL] migration:" + type(exc).__name__)
        raise SystemExit(1)
