"""Keep the eval suite honest: its expectations must match the seed data."""

from __future__ import annotations

from typing import Any

import pytest
from evals.dataset import EXAMPLES
from evals.evaluators import (
    approval_behavior,
    correct_facts,
    extract_amounts,
    tool_selection,
)
from evals.run import run_offline, summarize, to_markdown

from dealdesk.agent.tools import WRITE_TOOL_NAMES, CRMToolkit
from dealdesk.db.seed import DEALS

READ_EXAMPLES = [e for e in EXAMPLES if not e["outputs"]["expects_approval"]]
WRITE_EXAMPLES = [e for e in EXAMPLES if e["outputs"]["expects_approval"]]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("EUR 851,000", {851_000}),
        ("€851.000 total", {851_000}),
        ("about 851k", {851_000}),
        ("0.85M", {850_000}),
        ("won EUR 136,000 and lost EUR 33,000", {136_000, 33_000}),
    ],
)
def test_extract_amounts(text: str, expected: set[int]) -> None:
    assert expected <= extract_amounts(text)


@pytest.mark.parametrize("example", READ_EXAMPLES, ids=lambda e: e["inputs"]["question"][:40])
def test_reference_answers_pass_their_own_checks(example: dict[str, Any]) -> None:
    result = correct_facts({"answer": example["outputs"]["reference"]}, example["outputs"])
    assert result["score"] in (None, 1.0), result["comment"]


def test_dataset_refers_to_real_tools_and_deals(toolkit: CRMToolkit) -> None:
    tool_names = {t.name for t in toolkit.all_tools()}
    deal_ids = range(1, len(DEALS) + 1)
    for e in EXAMPLES:
        for group in e["outputs"]["required_tools"]:
            assert set(group) <= tool_names
        action = e["outputs"]["expected_action"]
        if action:
            assert action["tool"] in WRITE_TOOL_NAMES
            assert action["args"]["deal_id"] in deal_ids


def test_expected_write_targets_match_seed() -> None:
    names = {i + 1: d.name for i, d in enumerate(DEALS)}
    assert names[10] == "After-sales ticket automation"
    assert names[12] == "Service desk automation"
    assert names[8] == "Partner portal migration"
    assert names[9] == "Loyalty agent pilot"


def test_approval_and_tool_evaluators() -> None:
    ref = WRITE_EXAMPLES[0]["outputs"]
    good = {
        "status": "awaiting_approval",
        "pending_actions": [
            {
                "tool": "log_activity",
                "args": {"deal_id": 10, "type": "CALL", "summary": "x"},
                "requires_approval": True,
                "description": "Log CALL",
            }
        ],
    }
    assert approval_behavior(good, ref)["score"] == 1
    assert approval_behavior({"status": "completed", "pending_actions": []}, ref)["score"] == 0
    read_ref = READ_EXAMPLES[0]["outputs"]
    assert approval_behavior(good, read_ref)["score"] == 0
    assert tool_selection({"tool_calls": [{"name": "get_deal"}]}, read_ref)["score"] == 1
    assert tool_selection({"tool_calls": [{"name": "list_tasks"}]}, read_ref)["score"] == 0


def test_offline_runner_with_perfect_stub_agent() -> None:
    by_question = {e["inputs"]["question"]: e["outputs"] for e in EXAMPLES}

    def perfect(inputs: dict[str, Any]) -> dict[str, Any]:
        ref = by_question[inputs["question"]]
        if ref["expects_approval"]:
            act = ref["expected_action"]
            return {
                "answer": None,
                "status": "awaiting_approval",
                "tool_calls": [{"name": act["tool"], "args": act["args"]}],
                "pending_actions": [{**act, "requires_approval": True, "description": "x"}],
                "latency_s": 1.0,
                "total_tokens": 100,
            }
        return {
            "answer": ref["reference"],
            "status": "completed",
            "tool_calls": [{"name": g[0], "args": {}} for g in ref["required_tools"]],
            "pending_actions": [],
            "latency_s": 2.0,
            "total_tokens": 200,
        }

    rows = run_offline(perfect, [correct_facts, tool_selection, approval_behavior], 4)
    summary = summarize("stub:perfect", rows)
    assert summary["correct_facts"] == 1.0
    assert summary["tool_selection"] == 1.0
    assert summary["approval_behavior"] == 1.0
    assert "| stub:perfect | 100% | 100% | 100% | - |" in to_markdown([summary])
