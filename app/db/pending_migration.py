"""M4 additive schema; ordinary compliance audit rows remain outside the queue."""
from sqlalchemy import inspect, text
from app.models import KnowledgePending, KnowledgeDraft, BusinessAudit


def migrate_pending(engine):
    with engine.begin() as conn:
        locked = conn.dialect.name == "mysql"
        if locked and conn.execute(text("SELECT GET_LOCK('factoring_pending_schema', 15)")).scalar() != 1:
            raise RuntimeError("pending_schema_busy")
        try:
            return _apply(conn)
        finally:
            if locked:
                conn.execute(text("SELECT RELEASE_LOCK('factoring_pending_schema')"))


def _apply(conn):
    changes = []
    for model in (KnowledgePending,KnowledgeDraft,BusinessAudit):
        if not inspect(conn).has_table(model.__tablename__):
            model.__table__.create(conn)
            changes.append(model.__tablename__)
    draft_columns = {c["name"] for c in inspect(conn).get_columns("t_knowledge_draft")}
    if "publication_snapshot" not in draft_columns:
        conn.execute(text("ALTER TABLE t_knowledge_draft ADD COLUMN publication_snapshot JSON NULL"))
        changes.append("t_knowledge_draft.publication_snapshot")
    columns = {c["name"] for c in inspect(conn).get_columns("t_compliance_log")}
    for name, ddl in (("status","VARCHAR(16) NULL"),("resolution","TEXT NULL"),
                      ("assignee","BIGINT NULL"),("resolved_at","DATETIME NULL"),
                      ("user_id","BIGINT NULL"),("dedup_key","VARCHAR(64) NULL"),
                      ("external_id","VARCHAR(64) NULL")):
        if name not in columns:
            conn.execute(text(f"ALTER TABLE t_compliance_log ADD COLUMN {name} {ddl}"))
            changes.append("t_compliance_log." + name)
    indexes = {i["name"] for i in inspect(conn).get_indexes("t_compliance_log")}
    for name, spec in (("ix_compliance_pending_dedup", "(dedup_key)"),
                       ("uq_compliance_pending_external", "(external_id)")):
        if name not in indexes:
            unique = "UNIQUE " if name.startswith("uq_") else ""
            conn.execute(text(f"CREATE {unique}INDEX {name} ON t_compliance_log {spec}"))
            changes.append(name)
    return changes
