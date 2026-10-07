"""Add only the creation ledger; never change historical sessions/messages."""
from sqlalchemy import inspect
from app.models import SessionCreateRequestRecord


def migrate_session_requests(engine):
    with engine.begin() as connection:
        if inspect(connection).has_table(SessionCreateRequestRecord.__tablename__):
            return []
        SessionCreateRequestRecord.__table__.create(connection)
        return [SessionCreateRequestRecord.__tablename__]
