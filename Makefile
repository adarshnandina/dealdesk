.PHONY: install db up migrate seed chat serve mcp test lint typecheck check eval eval-offline

install:            ## install dependencies and git hooks
	uv sync
	uv run pre-commit install

db:                 ## start Postgres in Docker
	docker compose up -d db

migrate:            ## apply schema migrations + LangGraph checkpoint tables
	uv run dealdesk db upgrade

seed:               ## load demo CRM data (resets CRM tables)
	uv run dealdesk db seed

up: db migrate seed ## database ready to use

chat:               ## interactive chat in the terminal
	uv run dealdesk chat

serve:              ## REST API on http://127.0.0.1:8000/docs
	uv run dealdesk serve --reload

mcp:                ## MCP server over stdio
	uv run dealdesk mcp

lint:
	uv run ruff check . && uv run ruff format --check .

typecheck:
	uv run mypy

test:
	uv run pytest

check: lint typecheck test

eval:               ## compare models in LangSmith
	uv run python -m evals.run --model anthropic --model openai --judge anthropic

eval-offline:       ## run evals without uploading to LangSmith
	uv run python -m evals.run --model anthropic --offline
