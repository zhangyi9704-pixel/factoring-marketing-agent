"""MySQL 8 知识治理增量迁移：检查后加列，不重建已有表。"""
from sqlalchemy import inspect, text
from app.models import KnowledgeIngestionOperation, KnowledgeDraft


def migrate_knowledge_metadata(engine) -> list[str]:
    changes = []
    definitions = {"doc_id": "VARCHAR(64)", "version": "VARCHAR(64)",
                   "effective_date": "VARCHAR(10)", "status": "VARCHAR(16)",
                   "industry": "VARCHAR(16)", "category": "VARCHAR(16)",
                   "topic": "VARCHAR(256)", "content_snapshot": "TEXT",
                   "content_hash": "VARCHAR(64)", "chunk_manifest": "JSON", "ingestion_operation_id": "VARCHAR(32)"}
    with engine.begin() as connection:
        columns = {c["name"]: c for c in inspect(connection).get_columns("t_knowledge_doc")}
        for name, definition in definitions.items():
            if name not in columns:
                connection.execute(text(f"ALTER TABLE t_knowledge_doc ADD COLUMN {name} {definition} NULL"))
                changes.append(name)
        if connection.dialect.name == "mysql" and columns["source"]["type"].length < 512:
            connection.execute(text("ALTER TABLE t_knowledge_doc MODIFY COLUMN source VARCHAR(512) NOT NULL"))
            changes.append("source:widened")
        indexes = {i["name"] for i in inspect(connection).get_indexes("t_knowledge_doc")}
        constraints = {i["name"] for i in inspect(connection).get_unique_constraints("t_knowledge_doc")}
        if "uq_knowledge_doc_version" not in indexes | constraints:
            connection.execute(text("CREATE UNIQUE INDEX uq_knowledge_doc_version ON t_knowledge_doc (doc_id, version)"))
            changes.append("uq_knowledge_doc_version")
        if not inspect(connection).has_table(KnowledgeIngestionOperation.__tablename__):
            KnowledgeIngestionOperation.__table__.create(connection)
            changes.append(KnowledgeIngestionOperation.__tablename__)
        if inspect(connection).has_table(KnowledgeDraft.__tablename__):
            draft_columns = {c["name"] for c in inspect(connection).get_columns(KnowledgeDraft.__tablename__)}
            if "publication_snapshot" not in draft_columns:
                connection.execute(text("ALTER TABLE t_knowledge_draft ADD COLUMN publication_snapshot JSON NULL"))
                changes.append("t_knowledge_draft.publication_snapshot")
    return changes
