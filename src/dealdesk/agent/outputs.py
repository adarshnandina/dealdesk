"""Structured output schemas.

`with_structured_output()` makes the model return validated Pydantic objects instead of
free text. That is what lets other systems (an email client, a CRM card in the UI) consume
the agent's output without fragile string parsing.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FollowUpEmail(BaseModel):
    """A follow-up email draft for a deal. It is only a draft; nothing is sent."""

    to: str = Field(description="Recipient name and email address.")
    subject: str = Field(description="Short, specific subject line.")
    body: str = Field(description="Plain-text email body, signed by the deal owner.")
    call_to_action: str = Field(description="The single concrete next step we ask for.")


class DealSummary(BaseModel):
    """A structured account brief for one deal."""

    headline: str = Field(description="One sentence on where the deal stands.")
    health: Literal["on_track", "at_risk", "stalled"] = Field(
        description="on_track: recent activity and clear next step; at_risk: open risk or "
        "close date at risk; stalled: no meaningful activity for a long time."
    )
    key_facts: list[str] = Field(description="3-5 facts from the CRM record.")
    risks: list[str] = Field(description="Concrete risks, empty if none.")
    next_steps: list[str] = Field(description="1-3 recommended next actions.")
