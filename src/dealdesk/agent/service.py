"""Application service around the compiled graph.

The API, the CLI and the eval harness all use this class, so they share one definition of
"run a turn", "approve/reject" and "what is pending", and every run is traced to LangSmith
with the same tags and metadata.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from dealdesk.agent.graph import build_graph
from dealdesk.agent.tools import CRMToolkit
from dealdesk.config import Provider, Settings, get_settings
from dealdesk.db.session import get_session_factory
from dealdesk.llm.models import ModelSpec, get_chat_model, resolve_spec

logger = logging.getLogger(__name__)

Status = Literal["completed", "awaiting_approval"]


class NoPendingApprovalError(RuntimeError):
    pass


@dataclass
class TurnResult:
    thread_id: str
    status: Status
    answer: str | None
    pending_actions: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class AgentService:
    def __init__(
        self,
        graph: CompiledStateGraph[Any, Any, Any, Any],
        spec: ModelSpec,
        toolkit: CRMToolkit,
        settings: Settings,
    ) -> None:
        self.graph = graph
        self.spec = spec
        self.toolkit = toolkit
        self.settings = settings

    # ------------------------------------------------------------------ public API -----

    def chat(
        self,
        message: str,
        thread_id: str | None = None,
        *,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TurnResult:
        thread_id = thread_id or str(uuid.uuid4())
        if self.pending_actions(thread_id):
            raise RuntimeError("This thread is waiting for an approval decision first.")
        config = self._config(thread_id, tags, metadata)
        logger.info("chat turn", extra={"thread_id": thread_id, "model": self.spec.label})
        self.graph.invoke({"messages": [HumanMessage(message)]}, config)
        return self._result(thread_id)

    def decide(
        self,
        thread_id: str,
        decision: Literal["approve", "reject"],
        reason: str | None = None,
        *,
        tags: list[str] | None = None,
    ) -> TurnResult:
        if not self.pending_actions(thread_id):
            raise NoPendingApprovalError(f"Thread {thread_id} has no pending approval.")
        payload: dict[str, str] = {"decision": decision}
        if reason:
            payload["reason"] = reason
        config = self._config(thread_id, [*(tags or []), f"decision:{decision}"], None)
        logger.info("approval decision", extra={"thread_id": thread_id, "decision": decision})
        self.graph.invoke(Command(resume=payload), config)
        return self._result(thread_id)

    def pending_actions(self, thread_id: str) -> list[dict[str, Any]]:
        snapshot = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        actions: list[dict[str, Any]] = []
        for intr in snapshot.interrupts:
            if isinstance(intr.value, dict) and intr.value.get("type") == "approval_request":
                actions.extend(intr.value["actions"])
        return actions

    def history(self, thread_id: str) -> list[AnyMessage]:
        snapshot = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        messages: list[AnyMessage] = snapshot.values.get("messages", [])
        return messages

    # -------------------------------------------------------------------- helpers -----

    def _config(
        self, thread_id: str, tags: list[str] | None, metadata: dict[str, Any] | None
    ) -> RunnableConfig:
        # Tags and metadata make runs filterable in LangSmith (e.g. compare providers).
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self.settings.agent_recursion_limit,
            "run_name": "dealdesk_turn",
            "tags": [self.spec.provider, self.spec.model, *(tags or [])],
            "metadata": {
                "thread_id": thread_id,
                "llm_provider": self.spec.provider,
                "llm_model": self.spec.model,
                "env": self.settings.env,
                **(metadata or {}),
            },
        }

    def _result(self, thread_id: str) -> TurnResult:
        messages = self.history(thread_id)
        pending = self.pending_actions(thread_id)
        turn = _current_turn(messages)
        tool_calls = [
            {"name": c["name"], "args": c["args"]}
            for m in turn
            if isinstance(m, AIMessage)
            for c in m.tool_calls
        ]
        answer = None
        if not pending and turn and isinstance(turn[-1], AIMessage):
            answer = turn[-1].text
        return TurnResult(
            thread_id=thread_id,
            status="awaiting_approval" if pending else "completed",
            answer=answer,
            pending_actions=pending,
            tool_calls=tool_calls,
        )


def _current_turn(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Messages since the latest human message."""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i + 1 :]
    return messages


# ----------------------------------------------------------------------- factories -----


def build_service(
    *,
    checkpointer: BaseCheckpointSaver[Any] | None,
    provider: Provider | None = None,
    model: str | None = None,
    settings: Settings | None = None,
) -> AgentService:
    settings = settings or get_settings()
    spec = resolve_spec(provider, model, settings)
    llm = get_chat_model(spec.provider, spec.model, settings=settings)
    toolkit = CRMToolkit(session_factory=get_session_factory(settings.database_url), model=llm)
    graph = build_graph(
        llm,
        toolkit.all_tools(),
        checkpointer=checkpointer,
        stale_days=settings.stale_deal_days,
    )
    return AgentService(graph, spec, toolkit, settings)


@contextmanager
def postgres_checkpointer(dsn: str) -> Iterator[PostgresSaver]:
    """Durable checkpointer: paused approvals survive restarts and scale across workers."""
    with ConnectionPool(
        dsn,
        min_size=1,
        max_size=10,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    ) as pool:
        saver = PostgresSaver(pool)  # type: ignore[arg-type]
        saver.setup()  # idempotent: creates/migrates LangGraph's checkpoint tables
        yield saver
