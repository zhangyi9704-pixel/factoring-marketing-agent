"""Additive and repeatable. Legacy users stay disabled until explicitly provisioned."""
from sqlalchemy import inspect, text
from app.models import User


def migrate_auth(engine):
    changes = []
    with engine.begin() as connection:
        if not inspect(connection).has_table("t_user"):
            User.__table__.create(connection)
            return ["t_user"]
        columns = {c["name"] for c in inspect(connection).get_columns("t_user")}
        for name, ddl in (("salt", "VARCHAR(64) NULL"),
                          ("enabled", "BOOLEAN NOT NULL DEFAULT 0")):
            if name not in columns:
                connection.execute(text(f"ALTER TABLE t_user ADD COLUMN {name} {ddl}"))
                changes.append("t_user." + name)
    return changes
