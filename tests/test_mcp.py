from __future__ import annotations

import asyncio
import os
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient

from dealdesk.agent.tools import WRITE_TOOL_NAMES, CRMToolkit
from dealdesk.mcp_server import create_mcp_server


def test_server_exposes_crm_tools_with_annotations(toolkit: CRMToolkit) -> None:
    server = create_mcp_server(toolkit)
    tools = {t.name: t for t in asyncio.run(server.list_tools())}
    assert set(tools) >= WRITE_TOOL_NAMES | {"list_deals", "get_deal", "search_companies"}
    for name, tool in tools.items():
        assert tool.annotations is not None
        assert tool.annotations.readOnlyHint is (name not in WRITE_TOOL_NAMES)


def test_langchain_agent_can_load_tools_over_stdio(db_url: str, session_factory: object) -> None:
    """End to end: spawn the MCP server as a subprocess and call it via LangChain."""
    client = MultiServerMCPClient(
        {
            "dealdesk": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "dealdesk.mcp_server"],
                "env": {**os.environ, "DEALDESK_DATABASE_URL": db_url},
            }
        }
    )

    async def run() -> str:
        tools = {t.name: t for t in await client.get_tools()}
        result = await tools["search_companies"].ainvoke({"query": "fjordway"})
        return str(result)

    output = asyncio.run(run())
    assert "Fjordway Logistics AS" in output
