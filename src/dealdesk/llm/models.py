"""Model-agnostic chat model factory.

Every part of the app asks for a model through `get_chat_model()` and only ever sees
LangChain's `BaseChatModel` interface (`invoke`, `bind_tools`, `with_structured_output`).
Switching provider is therefore a config change (`DEALDESK_LLM_PROVIDER=openai`), not a
code change, and the eval harness can run the same agent against several models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from dealdesk.config import Provider, Settings, get_settings

# Sensible, cheap defaults. Override with DEALDESK_LLM_MODEL.
DEFAULT_MODELS: dict[Provider, str] = {
    "anthropic": "claude-haiku-5-5",
    "openai": "gpt-4.1-mini",
}


@dataclass(frozen=True)
class ModelSpec:
    provider: Provider
    model: str

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


def resolve_spec(
    provider: Provider | None = None, model: str | None = None, settings: Settings | None = None
) -> ModelSpec:
    settings = settings or get_settings()
    prov: Provider = provider or settings.llm_provider
    if prov not in DEFAULT_MODELS:
        raise ValueError(f"Unsupported provider '{prov}'. Choose one of {list(DEFAULT_MODELS)}.")
    # Only use the configured model name if it belongs to the configured provider.
    configured = settings.llm_model if prov == settings.llm_provider else None
    return ModelSpec(provider=prov, model=model or configured or DEFAULT_MODELS[prov])


def get_chat_model(
    provider: Provider | None = None,
    model: str | None = None,
    *,
    temperature: float | None = None,
    settings: Settings | None = None,
) -> BaseChatModel:
    settings = settings or get_settings()
    spec = resolve_spec(provider, model, settings)
    kwargs: dict[str, Any] = {
        "timeout": settings.llm_timeout_s,
        "max_retries": settings.llm_max_retries,
    }
    temp = settings.llm_temperature if temperature is None else temperature
    # Some reasoning models reject a temperature parameter; DEALDESK_LLM_TEMPERATURE= unsets it.
    if temp is not None:
        kwargs["temperature"] = temp
    if spec.provider == "openai" and settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    if spec.provider == "anthropic":
        kwargs["max_tokens"] = 2048
    chat_model = init_chat_model(spec.model, model_provider=spec.provider, **kwargs)
    assert isinstance(chat_model, BaseChatModel)
    return chat_model
