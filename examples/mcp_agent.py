"""Run the same DealDesk graph, but load the CRM tools from the MCP server.

This shows the stack is tool-source agnostic: tools can be in-process Python functions
or remote MCP tools, and the approval gate still applies because it works on tool names.

    uv run python examples/mcp_agent.py "Which deals are stale?"
"""

from __future__ import annotations

import asyncio
import os
import sys

from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver

from dealdesk.agent.graph import build_graph
from dealdesk.llm.models import get_chat_model


async def main(question: str) -> None:
    client = MultiServerMCPClient(
        {
            "dealdesk": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "dealdesk.mcp_server"],
                "env": dict(os.environ),
            }
        }
    )
    tools = await client.get_tools()
    print(f"Loaded {len(tools)} tools over MCP: {', '.join(t.name for t in tools)}")
    graph = build_graph(get_chat_model(), tools, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "mcp-demo"}, "tags": ["mcp-demo"]}
    state = await graph.ainvoke({"messages": [HumanMessage(question)]}, config)  # type: ignore[arg-type]
    if "__interrupt__" in state:
        print("Paused for approval:", state["__interrupt__"][0].value["actions"])
    else:
        print(state["messages"][-1].text)


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or "Which deals are stale?"))
