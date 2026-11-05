from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from langchain_core.messages import ToolMessage
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from dealdesk.agent.graph import build_graph
from dealdesk.agent.service import AgentService, NoPendingApprovalError, postgres_checkpointer
from dealdesk.agent.tools import CRMToolkit
from dealdesk.config import Settings
from dealdesk.db.models import Source, Task
from dealdesk.llm.models import ModelSpec
from tests.conftest import NOW
from tests.fakes import ScriptedChatModel, ai, tool_call

DUE = (NOW.date() + timedelta(days=3)).isoformat()


def agent_tasks(factory: sessionmaker[Session]) -> list[Task]:
    with factory() as s:
        return list(s.scalars(select(Task).where(Task.source == Source.AGENT)))


def test_read_only_question_completes_without_approval(
    service: AgentService, fake_model: ScriptedChatModel
) -> None:
    fake_model.responses = [
        ai("", tool_call("list_deals", stale_days=21)),
        ai("Three deals are stale."),
    ]
    result = service.chat("Which deals are stale?")
    assert result.status == "completed"
    assert result.answer == "Three deals are stale."
    assert result.tool_calls == [{"name": "list_deals", "args": {"stale_days": 21}}]
    # The tool result reached the model on its second call.
    tool_msg = fake_model.seen[1][-1]
    assert isinstance(tool_msg, ToolMessage)
    assert "Fleet CRM rollout" in str(tool_msg.content)


def test_system_prompt_contains_today(service: AgentService, fake_model: ScriptedChatModel) -> None:
    fake_model.responses = [ai("Hi")]
    service.chat("hello")
    assert NOW.date().isoformat() in str(fake_model.seen[0][0].content)


def test_write_pauses_for_approval_then_runs(
    service: AgentService,
    fake_model: ScriptedChatModel,
    session_factory: sessionmaker[Session],
) -> None:
    fake_model.responses = [
        ai("", tool_call("create_task", deal_id=4, title="Call Erik", due_date=DUE)),
        ai("Task created."),
    ]
    paused = service.chat("Create a task to call Erik")
    assert paused.status == "awaiting_approval"
    assert paused.answer is None
    [action] = paused.pending_actions
    assert action["tool"] == "create_task"
    assert action["requires_approval"]
    assert agent_tasks(session_factory) == []  # nothing written yet

    done = service.decide(paused.thread_id, "approve")
    assert done.status == "completed"
    assert done.answer == "Task created."
    [task] = agent_tasks(session_factory)
    assert task.title == "Call Erik"
    assert task.deal_id == 4


def test_rejected_write_changes_nothing(
    service: AgentService,
    fake_model: ScriptedChatModel,
    session_factory: sessionmaker[Session],
) -> None:
    fake_model.responses = [
        ai("", tool_call("update_deal_stage", deal_id=8, stage="closedwon")),
        ai("Understood, I left the deal unchanged."),
    ]
    paused = service.chat("Mark the Birkenwald deal as won")
    done = service.decide(paused.thread_id, "reject", "not signed yet")
    assert done.answer == "Understood, I left the deal unchanged."
    rejection = fake_model.seen[-1][-1]
    assert isinstance(rejection, ToolMessage)
    assert rejection.status == "error"
    assert "not signed yet" in str(rejection.content)
    with session_factory() as s:
        from dealdesk.db.models import Deal

        deal = s.get(Deal, 8)
        assert deal is not None
        assert deal.stage == "contractsent"


def test_mixed_batch_requires_approval_for_whole_batch(
    service: AgentService, fake_model: ScriptedChatModel
) -> None:
    fake_model.responses = [
        ai(
            "",
            tool_call("get_deal", "c1", deal_id=10),
            tool_call("log_activity", "c2", deal_id=10, type="CALL", summary="Called Stefan"),
        ),
        ai("Logged the call."),
    ]
    paused = service.chat("Log my call with Stefan")
    flags = {a["tool"]: a["requires_approval"] for a in paused.pending_actions}
    assert flags == {"get_deal": False, "log_activity": True}
    assert service.decide(paused.thread_id, "approve").status == "completed"


def test_cannot_chat_while_waiting_and_cannot_decide_without_pending(
    service: AgentService, fake_model: ScriptedChatModel
) -> None:
    with pytest.raises(NoPendingApprovalError):
        service.decide("unknown-thread", "approve")
    fake_model.responses = [
        ai("", tool_call("create_task", deal_id=4, title="x", due_date=DUE)),
    ]
    paused = service.chat("Create a task")
    with pytest.raises(RuntimeError, match="waiting for an approval"):
        service.chat("something else", paused.thread_id)


def test_conversation_memory_keeps_previous_turns(
    service: AgentService, fake_model: ScriptedChatModel
) -> None:
    fake_model.responses = [ai("First answer"), ai("Second answer")]
    first = service.chat("Question one")
    service.chat("Question two", first.thread_id)
    assert [m.text for m in fake_model.seen[1][1:]] == [
        "Question one",
        "First answer",
        "Question two",
    ]


def test_paused_run_survives_restart_with_postgres_checkpointer(
    toolkit: CRMToolkit,
    fake_model: ScriptedChatModel,
    settings: Settings,
    session_factory: sessionmaker[Session],
) -> None:
    fake_model.responses = [
        ai("", tool_call("create_task", deal_id=7, title="Answer DPA", due_date=DUE)),
        ai("Done."),
    ]

    def new_service(checkpointer: Any) -> AgentService:
        graph = build_graph(fake_model, toolkit.all_tools(), checkpointer=checkpointer)
        return AgentService(graph, ModelSpec("anthropic", "scripted"), toolkit, settings)

    with postgres_checkpointer(settings.psycopg_dsn) as saver:
        thread_id = new_service(saver).chat("Create a task for the DPA questions").thread_id
    # "Restart": a brand-new pool, saver, graph and service.
    with postgres_checkpointer(settings.psycopg_dsn) as saver:
        svc = new_service(saver)
        assert svc.pending_actions(thread_id)[0]["tool"] == "create_task"
        assert svc.decide(thread_id, "approve").answer == "Done."
    assert [t.title for t in agent_tasks(session_factory)] == ["Answer DPA"]
