from __future__ import annotations

from collections.abc import Iterator

import pytest
from typer.testing import CliRunner

from dealdesk.cli import app
from dealdesk.config import get_settings

runner = CliRunner()


@pytest.fixture
def cli_env(db_url: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("DEALDESK_DATABASE_URL", db_url)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("chat", "serve", "mcp", "db"):
        assert command in result.output


def test_db_upgrade_and_seed(cli_env: None) -> None:
    upgrade = runner.invoke(app, ["db", "upgrade"])
    assert upgrade.exit_code == 0, upgrade.output
    seed = runner.invoke(app, ["db", "seed"])
    assert seed.exit_code == 0, seed.output
    assert "'deals': 18" in seed.output
