"""Shared fixtures.

Tests run against a real Postgres (DEALDESK_TEST_DATABASE_URL), migrated with Alembic, so
the migrations themselves are tested too. LLM calls are replaced by a scripted fake model,
so the suite is fast, deterministic and free.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from dealdesk.agent.graph import build_graph
from dealdesk.agent.service import AgentService
from dealdesk.agent.tools import CRMToolkit
from dealdesk.config import Settings
from dealdesk.db.seed import seed
from dealdesk.db.session import session_scope
from dealdesk.llm.models import ModelSpec
from tests.fakes import ScriptedChatModel

ROOT = Path(__file__).resolve().parents[1]
TEST_DB_URL = os.environ.get(
    "DEALDESK_TEST_DATABASE_URL",
    "postgresql+psycopg://dealdesk:dealdesk@localhost:5432/dealdesk_test",
)
NOW = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)


def fixed_clock() -> datetime:
    return NOW


@pytest.fixture(scope="session")
def db_url() -> str:
    engine = create_engine(TEST_DB_URL)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(f"Postgres not available at {TEST_DB_URL}: {exc}")
    finally:
        engine.dispose()
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", TEST_DB_URL)
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")  # also proves the migrations apply cleanly
    return TEST_DB_URL


@pytest.fixture
def session_factory(db_url: str) -> Iterator[sessionmaker[Session]]:
    engine = create_engine(db_url)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with session_scope(factory) as session:
        seed(session, now=NOW)
    yield factory
    engine.dispose()


@pytest.fixture
def settings(db_url: str) -> Settings:
    return Settings(
        env="test",
        database_url=db_url,
        webhook_secret="test-secret",  # type: ignore[arg-type]
        api_key=None,
    )


@pytest.fixture
def fake_model() -> ScriptedChatModel:
    return ScriptedChatModel()


@pytest.fixture
def toolkit(session_factory: sessionmaker[Session], fake_model: ScriptedChatModel) -> CRMToolkit:
    return CRMToolkit(session_factory=session_factory, model=fake_model, clock=fixed_clock)


@pytest.fixture
def service(toolkit: CRMToolkit, fake_model: ScriptedChatModel, settings: Settings) -> AgentService:
    graph = build_graph(
        fake_model, toolkit.all_tools(), checkpointer=InMemorySaver(), today=lambda: NOW.date()
    )
    return AgentService(graph, ModelSpec("anthropic", "scripted"), toolkit, settings)
