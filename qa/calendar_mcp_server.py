"""Local, read-only sample calendar tools for MCP connector testing."""

import argparse
import json
from datetime import date, timedelta

from fastmcp import FastMCP

mcp = FastMCP("Calendar")

_today = date.today()
_events: list[dict[str, str]] = [
    {
        "title": "Team stand-up",
        "date": _today.isoformat(),
        "start_time": "09:30",
        "end_time": "10:00",
        "location": "Meeting room A",
    },
    {
        "title": "Project review",
        "date": _today.isoformat(),
        "start_time": "13:00",
        "end_time": "13:45",
        "location": "Video call",
    },
    {
        "title": "Planning session",
        "date": (_today + timedelta(days=1)).isoformat(),
        "start_time": "10:00",
        "end_time": "11:00",
        "location": "Meeting room B",
    },
]


@mcp.tool(annotations={"readOnlyHint": True})
def list_events(day: date) -> str:
    """List demo calendar events for a date in YYYY-MM-DD format."""
    events = [event for event in _events if event["date"] == day.isoformat()]
    return json.dumps({"date": day.isoformat(), "events": events}, indent=2)


@mcp.tool(annotations={"readOnlyHint": True})
def search_events(query: str, day: date | None = None) -> str:
    """Search demo event titles and locations, optionally limited to a date."""
    normalized_query = query.strip().casefold()
    if not normalized_query:
        return json.dumps({"query": query, "events": []})

    events: list[dict[str, str]] = [
        event
        for event in _events
        if normalized_query in event["title"].casefold()
        or normalized_query in event["location"].casefold()
    ]
    if day is not None:
        events = [event for event in events if event["date"] == day.isoformat()]

    return json.dumps({"query": query, "events": events}, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Calendar MCP demo server.")
    parser.add_argument("--port", type=int, default=8001, help="HTTP port (default: 8001).")
    args = parser.parse_args()
    mcp.run(transport="http", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
