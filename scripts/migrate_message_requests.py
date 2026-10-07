"""只新增请求账本；不改写历史对话、检查点或评估。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.models import MessageRequestRecord
from app.db.mysql import get_mysql_database
from sqlalchemy import inspect

if __name__ == '__main__':
    try:
        engine = get_mysql_database().engine
        exists = inspect(engine).has_table('t_message_request')
        MessageRequestRecord.__table__.create(engine, checkfirst=True)
        print('[OK] message request migration: ' + ('no changes' if exists else 't_message_request created'))
    except Exception as exc:
        print('[FAIL] message request migration: ' + type(exc).__name__)
        sys.exit(1)
