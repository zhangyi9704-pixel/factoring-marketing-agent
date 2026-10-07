"""Read-only startup validation. Schema changes belong to explicit migration commands."""
from sqlalchemy import inspect
from sqlalchemy.exc import SQLAlchemyError

from app.models import Base


def verify_runtime_schema(engine):
    missing = []
    try:
        with engine.connect() as connection:
            catalog = inspect(connection)
            tables = set(catalog.get_table_names())
            for name, table in Base.metadata.tables.items():
                if name not in tables:
                    missing.append(name)
                    continue
                columns = {column["name"] for column in catalog.get_columns(name)}
                missing.extend(f"{name}.{column.name}" for column in table.columns if column.name not in columns)
    except SQLAlchemyError:
        raise RuntimeError("runtime_schema_check_failed; check database availability using root README.md") from None
    if missing:
        raise RuntimeError("runtime_schema_missing: " + ", ".join(sorted(missing))
                           + "; run explicit initialization/migrations from root README.md")
    return []
