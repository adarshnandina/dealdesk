"""Evaluators. The deterministic ones are free and reproducible; the judge is optional.

All evaluators use LangSmith's `(inputs, outputs, reference_outputs)` signature, so the
same functions run in `client.evaluate(...)` and in the offline runner.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

Result = dict[str, Any]

_AMOUNT = re.compile(r"(\d{1,3}(?:[.,\s]\d{3})+|\d+(?:[.,]\d+)?)\s*([kKmM])?\b")


def extract_amounts(text: str) -> set[int]:
    """Find money amounts regardless of formatting: 851,000 / 851.000 / 851k / 0.85M."""
    found: set[int] = set()
    for number, suffix in _AMOUNT.findall(text):
        if suffix:
            value = float(number.replace(",", "."))
            found.add(round(value * (1_000 if suffix in "kK" else 1_000_000)))
        else:
            digits = re.sub(r"\D", "", number)
            if digits:
                found.add(int(digits))
    return found


def correct_facts(outputs: Result, reference_outputs: Result) -> Result:
    """Share of required facts and amounts present in the answer."""
    checks: list[str | list[str]] = reference_outputs.get("must_mention", [])
    amounts: list[int] = reference_outputs.get("amounts", [])
    if not checks and not amounts:
        return {"key": "correct_facts", "score": None, "comment": "not applicable"}
    answer = (outputs.get("answer") or "").lower()
    missing: list[str] = []
    for item in checks:
        options = [item] if isinstance(item, str) else item
        if not any(opt.lower() in answer for opt in options):
            missing.append(" | ".join(options))
    present = extract_amounts(answer)
    missing += [f"EUR {a:,}" for a in amounts if a not in present]
    total = len(checks) + len(amounts)
    return {
        "key": "correct_facts",
        "score": (total - len(missing)) / total,
        "comment": "missing: " + ", ".join(missing) if missing else "all facts present",
    }


def tool_selection(outputs: Result, reference_outputs: Result) -> Result:
    """1 if at least one tool from every required group was called."""
    groups: list[list[str]] = reference_outputs.get("required_tools", [])
    if not groups:
        return {"key": "tool_selection", "score": None, "comment": "not applicable"}
    called = {c["name"] for c in outputs.get("tool_calls", [])}
    missed = [g for g in groups if not called & set(g)]
    return {
        "key": "tool_selection",
        "score": 0 if missed else 1,
        "comment": f"called {sorted(called)}" + (f"; expected one of {missed}" if missed else ""),
    }


def approval_behavior(outputs: Result, reference_outputs: Result) -> Result:
    """Writes must pause with the right action; reads must never propose a write."""
    pending = [a for a in outputs.get("pending_actions", []) if a.get("requires_approval")]
    if not reference_outputs.get("expects_approval"):
        ok = outputs.get("status") == "completed" and not pending
        return {
            "key": "approval_behavior",
            "score": int(ok),
            "comment": "no write proposed" if ok else f"unexpected write: {pending}",
        }
    expected = reference_outputs["expected_action"]
    for action in pending:
        if action["tool"] == expected["tool"] and _args_match(action["args"], expected["args"]):
            return {"key": "approval_behavior", "score": 1, "comment": action["description"]}
    return {"key": "approval_behavior", "score": 0, "comment": f"got {pending or 'no pause'}"}


def _args_match(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(str(actual.get(k, "")).lower() == str(v).lower() for k, v in expected.items())


class Grade(BaseModel):
    reasoning: str = Field(description="One or two sentences explaining the grade.")
    correct: bool = Field(description="True if the answer agrees with the reference answer.")


JUDGE_PROMPT = """\
You grade answers from a CRM assistant. Compare the ANSWER with the REFERENCE.
The answer is correct if it contains the reference's key facts and states nothing that
contradicts the reference or invents data. Extra helpful detail is fine.

QUESTION: {question}
REFERENCE: {reference}
ANSWER: {answer}
"""


def make_judge(model: BaseChatModel):  # type: ignore[no-untyped-def]
    grader = model.with_structured_output(Grade)

    def llm_judge(inputs: Result, outputs: Result, reference_outputs: Result) -> Result:
        if reference_outputs.get("expects_approval"):
            return {"key": "llm_judge", "score": None, "comment": "write example"}
        grade = grader.invoke(
            JUDGE_PROMPT.format(
                question=inputs["question"],
                reference=reference_outputs["reference"],
                answer=outputs.get("answer") or "(no answer)",
            )
        )
        assert isinstance(grade, Grade)
        return {"key": "llm_judge", "score": int(grade.correct), "comment": grade.reasoning}

    return llm_judge


DETERMINISTIC_EVALUATORS = [correct_facts, tool_selection, approval_behavior]
