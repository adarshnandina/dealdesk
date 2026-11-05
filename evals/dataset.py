"""Evaluation dataset: 24 questions with checkable expectations.

Each example has
- `must_mention`: facts the answer must contain. An inner list means "any of these".
- `amounts`: EUR amounts the answer must state (formatting-insensitive).
- `required_tools`: groups of acceptable tools; at least one tool from each group must be
  called. Groups allow several valid strategies (e.g. a pipeline total can come from
  `get_pipeline_summary` or from adding up `list_deals`).
- `expects_approval` / `expected_action`: write requests must pause for approval with the
  right tool and arguments; read requests must never propose a write.
- `reference`: a short correct answer for the optional LLM-as-judge evaluator.

Answers are fixed because the seed data is hand-written (see `dealdesk/db/seed.py`).
"""

from __future__ import annotations

from typing import Any

DATASET_NAME = "dealdesk-crm-v1"
DATASET_DESCRIPTION = "DealDesk CRM agent: lookups, aggregates, multi-hop, grounding, writes."

LIST = ["list_deals"]
PIPELINE = ["get_pipeline_summary", "list_deals"]
DEAL = ["get_deal", "list_deals", "get_company_overview"]
# Phrases that signal "this is not in the data" (grounding checks; the judge is stricter).
NOT_FOUND = [
    "no company", "no deal", "no record", "not find", "couldn't find", "could not find",
    "not in the crm", "no match", "doesn't exist", "does not exist", "no results",
    "no phone", "not stored", "not available", "don't have", "do not have", "isn't",
    "doesn't include", "does not include", "not listed", "no such",
]  # fmt: skip


def ex(
    question: str,
    category: str,
    reference: str,
    *,
    must_mention: list[str | list[str]] | None = None,
    amounts: list[int] | None = None,
    required_tools: list[list[str]] | None = None,
    expected_action: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "inputs": {"question": question},
        "outputs": {
            "category": category,
            "reference": reference,
            "must_mention": must_mention or [],
            "amounts": amounts or [],
            "required_tools": required_tools or [],
            "expects_approval": expected_action is not None,
            "expected_action": expected_action,
        },
    }


EXAMPLES: list[dict[str, Any]] = [
    # --- lookups -------------------------------------------------------------------------
    ex(
        "What stage is the Seeblick claims intake deal in, and how big is it?",
        "lookup",
        "Claims intake agent is in 'Contract sent', worth EUR 180,000.",
        must_mention=[["Contract sent", "contractsent"]],
        amounts=[180_000],
        required_tools=[DEAL],
    ),
    ex(
        "Who is our contact at Rheinkraft and what did they ask for most recently?",
        "lookup",
        "Stefan Koch (Head of After-Sales) asked for final pricing for a 3-year term.",
        must_mention=["Stefan Koch", ["3-year", "three-year", "3 year"]],
        required_tools=[DEAL],
    ),
    ex(
        "Which open deals does Jonas Berg own?",
        "lookup",
        "Fleet CRM rollout, Service desk automation, Loyalty agent pilot, KYC document agent.",
        must_mention=[
            "Fleet CRM rollout",
            "Service desk automation",
            "Loyalty agent pilot",
            "KYC document agent",
        ],
        required_tools=[PIPELINE],
    ),
    ex(
        "Are any tasks overdue?",
        "lookup",
        "Yes: 'Follow up on IT security questionnaire' on the Fjordway Fleet CRM rollout.",
        must_mention=[["security questionnaire"], ["Fleet CRM rollout", "Fjordway"]],
        required_tools=[["list_tasks"]],
    ),
    ex(
        "Which deals have had no activity for more than three weeks?",
        "lookup",
        "Service desk automation (Ostsee), Fleet CRM rollout (Fjordway), "
        "Loyalty agent pilot (Lindqvist).",
        must_mention=["Fleet CRM rollout", "Loyalty agent pilot", "Service desk automation"],
        required_tools=[LIST],
    ),
    ex(
        "Which deals are in the contract sent stage?",
        "lookup",
        "Claims intake agent (Seeblick, EUR 180,000) and Partner portal migration "
        "(Birkenwald, EUR 48,000).",
        must_mention=["Claims intake agent", "Partner portal migration"],
        required_tools=[LIST],
    ),
    ex(
        "Which deals are expected to close in the next two weeks?",
        "lookup",
        "Claims intake agent and Partner portal migration.",
        must_mention=["Claims intake agent", "Partner portal migration"],
        required_tools=[LIST],
    ),
    ex(
        "What's our biggest open deal?",
        "lookup",
        "Claims intake agent with Seeblick Versicherung, EUR 180,000.",
        must_mention=["Claims intake agent"],
        amounts=[180_000],
        required_tools=[PIPELINE],
    ),
    # --- aggregates ----------------------------------------------------------------------
    ex(
        "What is the total value of our open pipeline?",
        "aggregate",
        "EUR 851,000 across 13 open deals.",
        amounts=[851_000],
        required_tools=[PIPELINE],
    ),
    ex(
        "How much open pipeline does Lena Fischer have?",
        "aggregate",
        "EUR 375,000 across 5 open deals.",
        amounts=[375_000],
        required_tools=[PIPELINE],
    ),
    ex(
        "How much have we won and how much have we lost in total?",
        "aggregate",
        "Won EUR 136,000; lost EUR 33,000.",
        amounts=[136_000, 33_000],
        required_tools=[PIPELINE],
    ),
    # --- multi-hop -----------------------------------------------------------------------
    ex(
        "What did Kranich Medtech buy from us, and is there a new opportunity with them?",
        "multi_hop",
        "They bought the Patient onboarding portal (won); the new opportunity is the Clinic "
        "CRM expansion to 4 more clinics.",
        must_mention=["Patient onboarding portal", "Clinic CRM expansion"],
        required_tools=[DEAL],
    ),
    ex(
        "Why did we lose the Tannenhof deal?",
        "multi_hop",
        "They chose a cheaper competitor because of budget constraints.",
        must_mention=[["competitor"], ["budget"]],
        required_tools=[DEAL],
    ),
    ex(
        "Where does the Fjordway fleet CRM rollout stand and what is blocking it?",
        "multi_hop",
        "EUR 120,000, decision maker bought-in, but stalled for ~5 weeks waiting on our "
        "answers to the IT security questionnaire (follow-up task is overdue).",
        must_mention=[["security questionnaire", "IT security"]],
        amounts=[120_000],
        required_tools=[DEAL],
    ),
    ex(
        "What is Mia Schneider's largest open deal and what is the next step on it?",
        "multi_hop",
        "After-sales ticket automation (Rheinkraft, EUR 95,000); next step is sending "
        "Stefan the 3-year pricing.",
        must_mention=["After-sales ticket automation", ["pricing"]],
        amounts=[95_000],
        required_tools=[PIPELINE],
    ),
    ex(
        "Are there any open issues on the Seeblick claims deal?",
        "multi_hop",
        "Yes: Seeblick legal has questions on the data processing agreement (DPA); a "
        "high-priority task to answer them is due tomorrow.",
        must_mention=[["DPA", "data processing"]],
        required_tools=[DEAL],
    ),
    # --- grounding (answer is not in the data) -------------------------------------------
    ex(
        "What's the status of our deal with Siemens?",
        "grounding",
        "There is no company or deal called Siemens in the CRM.",
        must_mention=[NOT_FOUND],
        required_tools=[["search_companies", "list_deals"]],
    ),
    ex(
        "What is Katrin Vogel's phone number?",
        "grounding",
        "The CRM has no phone numbers; only her email katrin.vogel@nordlicht-energie.example.",
        must_mention=[NOT_FOUND],
    ),
    # --- writes (must pause for approval) ------------------------------------------------
    ex(
        "Log a call on the Rheinkraft after-sales deal: Stefan confirmed he received the pricing.",
        "write",
        "Proposes log_activity(CALL) on deal 10 and waits for approval.",
        expected_action={"tool": "log_activity", "args": {"deal_id": 10, "type": "CALL"}},
    ),
    ex(
        "Create a high-priority task to call Felix at Ostsee next Monday.",
        "write",
        "Proposes create_task(HIGH) on deal 12 and waits for approval.",
        expected_action={"tool": "create_task", "args": {"deal_id": 12, "priority": "HIGH"}},
    ),
    ex(
        "Move the Birkenwald partner portal deal to closed won.",
        "write",
        "Proposes update_deal_stage(closedwon) on deal 8 and waits for approval.",
        expected_action={
            "tool": "update_deal_stage",
            "args": {"deal_id": 8, "stage": "closedwon"},
        },
    ),
    ex(
        "Add a note to the Lindqvist loyalty deal that Anna is back from vacation.",
        "write",
        "Proposes log_activity(NOTE) on deal 9 and waits for approval.",
        expected_action={"tool": "log_activity", "args": {"deal_id": 9, "type": "NOTE"}},
    ),
    # --- read-only requests that must NOT write ------------------------------------------
    ex(
        "Draft a follow-up email to Felix at Ostsee to restart the conversation.",
        "draft",
        "A draft email to Felix Neumann about the service desk automation budget decision; "
        "nothing is sent or logged.",
        must_mention=["Felix"],
        required_tools=[["draft_follow_up_email"]],
    ),
    ex(
        "Which three deals should the team focus on this week, and why?",
        "advice",
        "Grounded picks, e.g. Claims intake agent (contract sent, DPA questions), After-sales "
        "ticket automation (pricing requested), Fleet CRM rollout (stalled, overdue task).",
        required_tools=[PIPELINE],
    ),
]
