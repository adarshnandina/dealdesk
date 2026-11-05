from __future__ import annotations

import json

from dealdesk.agent.outputs import FollowUpEmail
from dealdesk.agent.tools import WRITE_TOOL_NAMES, CRMToolkit, describe_action
from tests.fakes import ScriptedChatModel


def test_read_and_write_tools_are_split(toolkit: CRMToolkit) -> None:
    assert {t.name for t in toolkit.write_tools()} == WRITE_TOOL_NAMES
    assert not {t.name for t in toolkit.read_tools()} & WRITE_TOOL_NAMES
    assert all(t.description for t in toolkit.all_tools())


def test_list_deals_tool_parses_stage_string(toolkit: CRMToolkit) -> None:
    tool = next(t for t in toolkit.read_tools() if t.name == "list_deals")
    deals = json.loads(tool.invoke({"stage": "contractsent"}))
    assert [d["name"] for d in deals] == ["Claims intake agent", "Partner portal migration"]


def test_tool_returns_error_instead_of_raising(toolkit: CRMToolkit) -> None:
    tool = next(t for t in toolkit.read_tools() if t.name == "get_deal")
    assert json.loads(tool.invoke({"deal_id": 999})) == {"error": "Deal 999 not found"}


def test_write_tool_validation_error_is_returned(toolkit: CRMToolkit) -> None:
    tool = next(t for t in toolkit.write_tools() if t.name == "create_task")
    result = json.loads(tool.invoke({"deal_id": 4, "title": "x", "due_date": "2020-01-01"}))
    assert "in the past" in result["error"]


def test_draft_email_uses_structured_output(
    toolkit: CRMToolkit, fake_model: ScriptedChatModel
) -> None:
    fake_model.structured["FollowUpEmail"] = FollowUpEmail(
        to="Stefan Koch <stefan.koch@rheinkraft.example>",
        subject="3-year pricing",
        body="Hi Stefan, ...",
        call_to_action="Book a 20-minute call",
    )
    tool = next(t for t in toolkit.read_tools() if t.name == "draft_follow_up_email")
    email = json.loads(tool.invoke({"deal_id": 10, "goal": "send pricing"}))
    assert email["subject"] == "3-year pricing"


def test_describe_action() -> None:
    assert (
        describe_action("update_deal_stage", {"deal_id": 8, "stage": "closedwon"})
        == "Move deal #8 to stage 'Closed won'"
    )
    assert "due 2026-11-10" in describe_action(
        "create_task", {"deal_id": 1, "title": "Demo", "due_date": "2026-11-10"}
    )
