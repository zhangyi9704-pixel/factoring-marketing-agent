"""M3 合规审计扩展，只新增列，重复执行无修改。"""
from sqlalchemy import inspect, text


def migrate_compliance(engine):
    changes = []
    with engine.begin() as connection:
        for table, name, ddl in (
            ("t_session", "needs_human_review", "INTEGER NOT NULL DEFAULT 0"),
            ("t_compliance_log", "details", "JSON NULL"),
        ):
            columns = {c["name"] for c in inspect(connection).get_columns(table)}
            if name not in columns:
                connection.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))
                changes.append(f"{table}.{name}")
    return changes
