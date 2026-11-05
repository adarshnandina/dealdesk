"""MCP server exposing the CRM tools.

Any MCP client (Claude Desktop, Claude Code, Cursor, or another team's LangGraph agent
via `langchain-mcp-adapters`) can use the same CRM tools without importing our code.
Write tools are marked with MCP tool annotations so clients can ask the user before
running them.

Run:  dealdesk mcp                      (stdio, for desktop clients)
      dealdesk mcp --transport http     (streamable HTTP on :8001/mcp)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from dealdesk.agent.tools import CRMToolkit
from dealdesk.db.session import get_session_factory


def create_mcp_server(toolkit: CRMToolkit | None = None, port: int = 8001) -> FastMCP:
    toolkit = toolkit or CRMToolkit(session_factory=get_session_factory())
    server = FastMCP(
        "dealdesk-crm",
        instructions="Tools for a HubSpot-style CRM: companies, contacts, deals, tasks.",
        port=port,
    )
    read_only = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
    write = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)

    reads: list[Callable[..., str]] = [
        toolkit.search_companies,
        toolkit.get_company_overview,
        toolkit.list_deals,
        toolkit.get_deal,
        toolkit.get_pipeline_summary,
        toolkit.list_tasks,
    ]
    writes: list[Callable[..., str]] = [
        toolkit.log_activity,
        toolkit.create_task,
        toolkit.update_deal_stage,
    ]
    for fn in reads:
        server.add_tool(fn, name=fn.__name__, annotations=read_only)
    for fn in writes:
        server.add_tool(fn, name=fn.__name__, annotations=write)
    return server


def run(transport: Literal["stdio", "streamable-http"] = "stdio", port: int = 8001) -> None:
    create_mcp_server(port=port).run(transport=transport)


if __name__ == "__main__":
    run()
