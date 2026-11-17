"""Smoke-test the local Calendar MCP server through FastMCP's client."""

import asyncio
from datetime import date
import json

from fastmcp import Client
from fastmcp.exceptions import ToolError

from calendar_mcp_server import mcp


async def verify_calendar_tools() -> None:
    async with Client(mcp) as client:
        tools = await client.list_tools()
        tool_names = {tool.name for tool in tools}
        assert tool_names == {"list_events", "search_events"}, tool_names

        list_events = next(tool for tool in tools if tool.name == "list_events")
        assert "day" in list_events.input_schema["properties"]

        result = await client.call_tool(
            "list_events",
            {"day": date.today().isoformat()},
        )
        assert not result.is_error
        event_data = json.loads(result.content[0].text)
        assert event_data["events"], event_data
        assert event_data["events"][0]["title"] == "Team stand-up"

        try:
            await client.call_tool("list_events", {"day": "not-a-date"})
        except ToolError as error:
            assert "valid date" in str(error)
        else:
            raise AssertionError("Invalid calendar dates must be rejected.")


if __name__ == "__main__":
    asyncio.run(verify_calendar_tools())
    print("Calendar MCP tool discovery and calls passed.")
