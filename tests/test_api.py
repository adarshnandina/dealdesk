from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy.orm import Session, sessionmaker

from dealdesk.agent.outputs import DealSummary
from dealdesk.agent.service import AgentService
from dealdesk.api.app import create_app, sign
from dealdesk.config import Settings
from dealdesk.db.models import Deal
from tests.conftest import NOW
from tests.fakes import ScriptedChatModel, ai, tool_call

DUE = (NOW.date() + timedelta(days=2)).isoformat()


@pytest.fixture
def client(service: AgentService, settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(service, settings)) as c:
        yield c


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body == {"status": "ok", "database": True, "model": "anthropic:scripted"}


def test_chat_then_approve(client: TestClient, fake_model: ScriptedChatModel) -> None:
    fake_model.responses = [
        ai("", tool_call("create_task", deal_id=10, title="Send pricing", due_date=DUE)),
        ai("Created the task."),
    ]
    paused = client.post("/v1/chat", json={"message": "Remind me to send pricing"}).json()
    assert paused["status"] == "awaiting_approval"
    thread = paused["thread_id"]
    assert paused["pending_actions"][0]["description"].startswith("Create task 'Send pricing'")

    # A second message on a paused thread is rejected.
    assert client.post("/v1/chat", json={"message": "hi", "thread_id": thread}).status_code == 409

    view = client.get(f"/v1/threads/{thread}").json()
    assert view["status"] == "awaiting_approval"

    done = client.post(f"/v1/threads/{thread}/decision", json={"decision": "approve"}).json()
    assert done["status"] == "completed"
    assert done["answer"] == "Created the task."
    assert (
        client.post(f"/v1/threads/{thread}/decision", json={"decision": "approve"}).status_code
        == 409
    )


def test_validation_error_for_empty_message(client: TestClient) -> None:
    assert client.post("/v1/chat", json={"message": ""}).status_code == 422


def test_api_key_is_enforced_when_configured(service: AgentService, settings: Settings) -> None:
    secured = settings.model_copy(update={"api_key": SecretStr("s3cret")})
    with TestClient(create_app(service, secured)) as c:
        assert c.post("/v1/chat", json={"message": "hi"}).status_code == 401
        assert c.get("/health").status_code == 200  # health stays public


def test_deal_summary_structured_output(client: TestClient, fake_model: ScriptedChatModel) -> None:
    fake_model.structured["DealSummary"] = DealSummary(
        headline="Contract sent; DPA questions open.",
        health="at_risk",
        key_facts=["EUR 180,000", "Contract sent"],
        risks=["Legal review of DPA"],
        next_steps=["Answer DPA questions"],
    )
    body = client.post("/v1/deals/7/summary").json()
    assert body["health"] == "at_risk"
    assert client.post("/v1/deals/999/summary").status_code == 404


def _webhook(client: TestClient, payload: dict[str, object], secret: str = "test-secret"):  # type: ignore[no-untyped-def]
    raw = json.dumps(payload).encode()
    return client.post(
        "/v1/webhooks/deal-stage-changed",
        content=raw,
        headers={"content-type": "application/json", "x-dealdesk-signature": sign(raw, secret)},
    )


def test_webhook_rejects_bad_signature(client: TestClient) -> None:
    payload = {
        "event_id": "e1",
        "deal_id": 8,
        "new_stage": "closedwon",
        "occurred_at": NOW.isoformat(),
    }
    assert _webhook(client, payload, secret="wrong").status_code == 401


def test_webhook_syncs_stage_and_proposes_task(
    client: TestClient,
    fake_model: ScriptedChatModel,
    session_factory: sessionmaker[Session],
) -> None:
    fake_model.responses = [
        ai("", tool_call("create_task", deal_id=8, title="Kickoff with Julia", due_date=DUE)),
    ]
    payload = {
        "event_id": "e42",
        "deal_id": 8,
        "new_stage": "closedwon",
        "occurred_at": NOW.isoformat(),
    }
    accepted = _webhook(client, payload)
    assert accepted.status_code == 202
    thread = accepted.json()["thread_id"]

    with session_factory() as s:
        deal = s.get(Deal, 8)
        assert deal is not None
        assert deal.stage == "closedwon"
    view = client.get(f"/v1/threads/{thread}").json()
    assert view["status"] == "awaiting_approval"
    assert view["pending_actions"][0]["args"]["title"] == "Kickoff with Julia"

    # Retried delivery of the same event is ignored.
    assert _webhook(client, payload).json()["status"] == "duplicate"
