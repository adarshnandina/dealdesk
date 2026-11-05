"""The DealDesk agent as an explicit LangGraph state machine.

                 ┌──────────────┐
      START ───▶ │    agent     │ ◀───────────────────────────┐
                 └──────┬───────┘                             │
          no tool calls │   read-only calls   any write call   │
                 ┌──────┘        │                 │          │
                 ▼               ▼                 ▼          │
                END        ┌──────────┐     ┌─────────────┐   │
                           │  tools   │ ◀── │  approval   │   │
                           └────┬─────┘ yes └──────┬──────┘   │
                                │                  │ no       │
                                └──────────────────┴──────────┘

- `agent` calls the LLM with all tools bound.
- `route_after_agent` sends read-only tool calls straight to `tools`, but any batch that
  contains a write goes to `approval` first.
- `approval` calls `interrupt()`: the run pauses, its state is saved by the checkpointer,
  and it resumes when a human sends `Command(resume={"decision": "approve" | "reject"})`.
  Approve runs the tools; reject answers each tool call with a "rejected" ToolMessage, so
  the model sees what happened and can reply.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from dealdesk.agent.prompts import render_system_prompt
from dealdesk.agent.tools import WRITE_TOOL_NAMES, describe_action


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class ApprovalRequest(TypedDict):
    """Payload surfaced to the client while the graph is paused."""

    type: Literal["approval_request"]
    actions: list[dict[str, Any]]


class ApprovalDecision(TypedDict, total=False):
    decision: Literal["approve", "reject"]
    reason: str


def _last_ai_message(state: AgentState) -> AIMessage:
    message = state["messages"][-1]
    if not isinstance(message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(message).__name__}")
    return message


def build_graph(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    *,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    today: Callable[[], date] = date.today,
    stale_days: int = 21,
    write_tool_names: frozenset[str] = WRITE_TOOL_NAMES,
) -> CompiledStateGraph[AgentState, Any, AgentState, AgentState]:
    model_with_tools = model.bind_tools(list(tools))

    def agent(state: AgentState) -> dict[str, list[AnyMessage]]:
        system = SystemMessage(render_system_prompt(today(), stale_days))
        response = model_with_tools.invoke([system, *state["messages"]])
        return {"messages": [response]}

    def route_after_agent(state: AgentState) -> Literal["tools", "approval", "__end__"]:
        calls = _last_ai_message(state).tool_calls
        if not calls:
            return "__end__"
        if any(call["name"] in write_tool_names for call in calls):
            return "approval"
        return "tools"

    def approval(state: AgentState) -> Command[Literal["tools", "agent"]]:
        calls = _last_ai_message(state).tool_calls
        request: ApprovalRequest = {
            "type": "approval_request",
            "actions": [
                {
                    "tool_call_id": c["id"],
                    "tool": c["name"],
                    "args": c["args"],
                    "requires_approval": c["name"] in write_tool_names,
                    "description": describe_action(c["name"], c["args"]),
                }
                for c in calls
            ],
        }
        # Pauses here. On resume, `interrupt()` returns the human's decision.
        decision: ApprovalDecision = interrupt(request)
        if decision.get("decision") == "approve":
            return Command(goto="tools")

        reason = decision.get("reason") or "no reason given"
        rejected = [
            ToolMessage(
                content=(
                    f"Rejected by the user ({reason}). Nothing was changed."
                    if c["name"] in write_tool_names
                    else "Not executed: it was part of a batch the user rejected."
                ),
                tool_call_id=c["id"],
                name=c["name"],
                status="error",
            )
            for c in calls
        ]
        return Command(goto="agent", update={"messages": rejected})

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent)
    builder.add_node("approval", approval)
    builder.add_node("tools", ToolNode(list(tools)))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_agent, ["tools", "approval", END])
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer, name="dealdesk")
