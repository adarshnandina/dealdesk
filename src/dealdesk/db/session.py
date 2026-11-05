"""Engine and session management."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from dealdesk.config import get_settings


@lru_cache
def get_engine(url: str | None = None) -> Engine:
    # pool_pre_ping drops dead connections (e.g. after a Supabase/pgbouncer restart).
    return create_engine(url or get_settings().database_url, pool_pre_ping=True, future=True)


def get_session_factory(url: str | None = None) -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(url), expire_on_commit=False)


@contextmanager
def session_scope(factory: sessionmaker[Session] | None = None) -> Iterator[Session]:
    """Unit of work: commit on success, roll back on any error."""
    session = (factory or get_session_factory())()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
