"""补报告快照、核对唯一约束并扩容满分字段；不删除历史记录。"""
from sqlalchemy import inspect, text


def migrate_evaluation(engine):
    changes = []
    with engine.begin() as conn:
        columns = {c['name']: c for c in inspect(conn).get_columns('t_evaluation')}
        if 'report_snapshot' not in columns:
            conn.execute(text('ALTER TABLE t_evaluation ADD COLUMN report_snapshot JSON NULL'))
            changes.append('t_evaluation.report_snapshot')
        unique = inspect(conn).get_unique_constraints('t_evaluation')
        if not any(u['column_names'] == ['session_id'] for u in unique):
            duplicates = conn.execute(text('SELECT COUNT(*) FROM (SELECT session_id FROM t_evaluation GROUP BY session_id HAVING COUNT(*)>1) duplicates')).scalar()
            if duplicates:
                raise RuntimeError('duplicate_evaluations_require_review; no rows deleted')
            conn.execute(text('ALTER TABLE t_evaluation ADD CONSTRAINT uq_evaluation_session UNIQUE(session_id)'))
            changes.append('t_evaluation.unique_session')
        score = next(c for c in inspect(conn).get_columns('t_evaluation_evidence') if c['name'] == 'score')
        if getattr(score['type'], 'precision', None) == 4:
            conn.execute(text('ALTER TABLE t_evaluation_evidence MODIFY COLUMN score DECIMAL(5,2) NULL'))
            changes.append('t_evaluation_evidence.score_decimal5_2')
    return changes
