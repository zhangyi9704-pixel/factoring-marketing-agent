"""Verify missing schema fails without implicit DDL or business writes."""
from sqlalchemy import create_engine, event, text
import pytest

from app.db.runtime_schema import verify_runtime_schema
from app.models import Base


def test_current_schema_validates_using_readonly_statements():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    statements = []
    event.listen(engine, "before_cursor_execute", lambda conn, cursor, statement, *args: statements.append(statement))
    assert verify_runtime_schema(engine) == []
    assert statements
    assert all(statement.lstrip().upper().startswith(("SELECT", "PRAGMA")) for statement in statements)


def test_missing_reliability_tables_fail_with_explicit_migration_hint():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(text("DROP TABLE t_knowledge_ingestion_operation"))
        connection.execute(text("DROP TABLE t_session_create_request"))
    with pytest.raises(RuntimeError, match="runtime_schema_missing") as error:
        verify_runtime_schema(engine)
    assert "t_knowledge_ingestion_operation" in str(error.value)
    assert "t_session_create_request" in str(error.value)
    assert "README.md" in str(error.value)
