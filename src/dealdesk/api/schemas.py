"""Request/response models for the REST API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from dealdesk.db.models import DealStage


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = Field(default=None, max_length=200)


class DecisionRequest(BaseModel):
    decision: Literal["approve", "reject"]
    reason: str | None = Field(default=None, max_length=500)


class PendingAction(BaseModel):
    tool_call_id: str
    tool: str
    args: dict[str, Any]
    requires_approval: bool
    description: str


class TurnResponse(BaseModel):
    thread_id: str
    status: Literal["completed", "awaiting_approval"]
    answer: str | None
    pending_actions: list[PendingAction]
    tool_calls: list[dict[str, Any]]


class MessageView(BaseModel):
    role: str
    content: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)


class ThreadResponse(BaseModel):
    thread_id: str
    status: Literal["completed", "awaiting_approval", "empty"]
    pending_actions: list[PendingAction]
    messages: list[MessageView]


class DealStageChangedEvent(BaseModel):
    """Shaped like a HubSpot property-change webhook for `dealstage`."""

    event_id: str = Field(min_length=1, max_length=100)
    deal_id: int
    new_stage: DealStage
    occurred_at: datetime


class WebhookAccepted(BaseModel):
    status: Literal["accepted", "duplicate"]
    thread_id: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    database: bool
    model: str
