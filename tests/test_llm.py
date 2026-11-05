from __future__ import annotations

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from dealdesk.config import Settings
from dealdesk.llm.models import DEFAULT_MODELS, get_chat_model, resolve_spec


@pytest.fixture(autouse=True)
def _dummy_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    # Constructing a client needs a key; no request is ever sent in these tests.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_KEY", "test")


def test_defaults_per_provider() -> None:
    s = Settings(llm_provider="anthropic", llm_model=None)
    assert resolve_spec(settings=s).model == DEFAULT_MODELS["anthropic"]
    assert resolve_spec("openai", settings=s).model == DEFAULT_MODELS["openai"]


def test_configured_model_only_applies_to_configured_provider() -> None:
    s = Settings(llm_provider="anthropic", llm_model="claude-sonnet-5-5")
    assert resolve_spec(settings=s).label == "anthropic:claude-sonnet-5-5"
    assert resolve_spec("openai", settings=s).model == DEFAULT_MODELS["openai"]


def test_unsupported_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported provider"):
        resolve_spec("mistral", settings=Settings())  # type: ignore[arg-type]


def test_factory_builds_provider_specific_clients() -> None:
    s = Settings(openai_base_url="https://models.github.ai/inference")
    assert isinstance(get_chat_model("anthropic", settings=s), ChatAnthropic)
    openai_model = get_chat_model("openai", settings=s)
    assert isinstance(openai_model, ChatOpenAI)
    assert openai_model.openai_api_base == "https://models.github.ai/inference"


def test_temperature_can_be_unset_for_reasoning_models() -> None:
    model = get_chat_model("openai", "o4-mini", settings=Settings(llm_temperature=None))
    assert isinstance(model, ChatOpenAI)
    assert model.temperature is None
