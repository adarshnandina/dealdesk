"""Command-line interface: `dealdesk --help`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from dealdesk.config import Provider, get_settings
from dealdesk.logging_config import configure_logging

app = typer.Typer(help="DealDesk: a CRM assistant agent.", no_args_is_help=True)
db_app = typer.Typer(help="Database commands.", no_args_is_help=True)
app.add_typer(db_app, name="db")

ProviderOpt = Annotated[Provider | None, typer.Option(help="anthropic or openai")]
ModelOpt = Annotated[str | None, typer.Option(help="Model name, e.g. claude-haiku-5-5")]


@app.callback()
def _setup() -> None:
    s = get_settings()
    # CLI output is for humans: only show warnings unless DEBUG logging is requested.
    configure_logging("DEBUG" if s.log_level.upper() == "DEBUG" else "WARNING", s.log_format)


def _alembic_ini() -> Path:
    for candidate in (Path.cwd() / "alembic.ini", Path(__file__).parents[2] / "alembic.ini"):
        if candidate.exists():
            return candidate
    raise typer.BadParameter("alembic.ini not found; run from the project root.")


@db_app.command("upgrade")
def db_upgrade() -> None:
    """Apply CRM migrations (Alembic) and set up LangGraph checkpoint tables."""
    from alembic import command
    from alembic.config import Config

    from dealdesk.agent.service import postgres_checkpointer

    ini = _alembic_ini()
    cfg = Config(str(ini))
    cfg.set_main_option("script_location", str(ini.parent / "migrations"))
    command.upgrade(cfg, "head")
    with postgres_checkpointer(get_settings().psycopg_dsn):
        pass
    typer.secho("Database is up to date.", fg=typer.colors.GREEN)


@db_app.command("seed")
def db_seed() -> None:
    """Reset the CRM tables and load the demo dataset."""
    from dealdesk.db.seed import seed
    from dealdesk.db.session import session_scope

    with session_scope() as session:
        counts = seed(session)
    typer.secho(f"Seeded: {counts}", fg=typer.colors.GREEN)


@app.command()
def chat(
    provider: ProviderOpt = None,
    model: ModelOpt = None,
    thread: Annotated[str | None, typer.Option(help="Resume an existing thread.")] = None,
) -> None:
    """Interactive chat. Write actions are shown for approval before they run."""
    from dealdesk.agent.service import build_service, postgres_checkpointer

    with postgres_checkpointer(get_settings().psycopg_dsn) as checkpointer:
        svc = build_service(checkpointer=checkpointer, provider=provider, model=model)
        thread_id = thread
        typer.secho(f"DealDesk ({svc.spec.label}). Type 'exit' to quit.", fg=typer.colors.CYAN)
        while True:
            message = typer.prompt("you", prompt_suffix=" > ").strip()
            if message.lower() in {"exit", "quit"}:
                break
            result = svc.chat(message, thread_id, tags=["cli"])
            thread_id = result.thread_id
            while result.status == "awaiting_approval":
                typer.secho("\nThe agent wants to:", fg=typer.colors.YELLOW)
                for action in result.pending_actions:
                    if action["requires_approval"]:
                        typer.echo(f"  - {action['description']}")
                approved = typer.confirm("Approve?", default=False)
                reason = None if approved else typer.prompt("Reason", default="not now")
                result = svc.decide(
                    thread_id, "approve" if approved else "reject", reason, tags=["cli"]
                )
            typer.secho(f"\ndealdesk > {result.answer}\n", fg=typer.colors.GREEN)
        typer.echo(f"Thread: {thread_id}")


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: Annotated[bool, typer.Option(help="Auto-reload (development).")] = False,
) -> None:
    """Run the REST API."""
    import uvicorn

    uvicorn.run("dealdesk.api.app:create_app", factory=True, host=host, port=port, reload=reload)


@app.command()
def mcp(
    transport: Annotated[str, typer.Option(help="stdio or http")] = "stdio",
    port: int = 8001,
) -> None:
    """Run the MCP server that exposes the CRM tools."""
    from dealdesk.mcp_server import run

    run("streamable-http" if transport == "http" else "stdio", port=port)


if __name__ == "__main__":
    app()
