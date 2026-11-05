"""Application settings.

All configuration comes from environment variables (optionally loaded from a `.env` file),
following the twelve-factor app approach: the same image runs locally, in CI and in
production, and only the environment changes.

App settings use the `DEALDESK_` prefix. Provider keys keep their standard names
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `LANGSMITH_API_KEY`, ...) because the LangChain
and LangSmith SDKs read them directly from the environment.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["anthropic", "openai"]

# Export .env values into os.environ so SDKs (Anthropic, OpenAI, LangSmith) can see them.
load_dotenv(override=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DEALDESK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Runtime -----------------------------------------------------------------------
    env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    log_format: Literal["console", "json"] = "console"

    # --- Database ----------------------------------------------------------------------
    # SQLAlchemy URL using the psycopg 3 driver. Works unchanged against Supabase.
    database_url: str = "postgresql+psycopg://dealdesk:dealdesk@localhost:5432/dealdesk"

    # --- LLM ---------------------------------------------------------------------------
    llm_provider: Provider = "anthropic"
    llm_model: str | None = Field(
        default=None, description="Model name; falls back to the provider default."
    )
    llm_temperature: float | None = 0.0
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2
    # Optional OpenAI-compatible endpoint, e.g. GitHub Models or a gateway.
    openai_base_url: str | None = None

    # --- Agent -------------------------------------------------------------------------
    agent_recursion_limit: int = 25
    stale_deal_days: int = 21

    # --- API ---------------------------------------------------------------------------
    api_key: SecretStr | None = None
    webhook_secret: SecretStr = SecretStr("change-me")

    @property
    def psycopg_dsn(self) -> str:
        """Plain libpq DSN (without the SQLAlchemy driver suffix) for psycopg/LangGraph."""
        return self.database_url.replace("postgresql+psycopg://", "postgresql://", 1)


@lru_cache
def get_settings() -> Settings:
    return Settings()
