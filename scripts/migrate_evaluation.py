from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.db.mysql import get_mysql_database
from app.db.evaluation_migration import migrate_evaluation

if __name__ == '__main__':
    changes = migrate_evaluation(get_mysql_database().engine)
    print('[OK] evaluation migration: ' + (', '.join(changes) or 'no changes'))
