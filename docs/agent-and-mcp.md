# LangChain agent and MCP demo

The production text/image chat path uses `langchain-rust`'s `AgentExecutor` with persistent `SqliteBaseMemory`. The default build has no MCP connector commands or controls. MCP support is an experimental, organization-unauthorized demo and is compiled only with the opt-in `mcp-demo` Cargo feature.

## Agent behavior

- Agent memory uses encrypted local SQLite/SQLCipher and is keyed by conversation timeline. Before each turn, the runtime compares its visible transcript with the saved chat history; it rebuilds memory when they diverge, such as after switching or editing a timeline.
- Context is bounded to 20 recent LangChain messages while preserving complete tool-call turns. The app also rejects an oversized visible transcript. This is persistent completed-turn memory, not a durable mid-run checkpoint or resume mechanism.
- Every text/image agent turn includes `get_current_datetime`, a built-in tool that returns the computer's local date, time, weekday, and UTC offset. The prompt requires it for explicit current date/time/day questions and before calendar, agenda, schedule, or appointment MCP calls when the request uses an unspecified or relative date. The agent must pass the returned local date to the calendar tool and must not trust a server's implicit “today” default. Audio-bearing histories keep using the direct model path and do not have agent tools.
- When selected collections contain indexed documents, the agent receives a `search_local_documents` tool. Its prompt explicitly names and requires calling `search_local_documents` when no relevant MCP tool is available or connector results are incomplete. The tool searches indexed documents in the collections selected for that turn. If tools return no relevant evidence, any answer based on pretrained knowledge must be labeled `General knowledge (not verified against MCP tools or search_local_documents):`; it must not be presented as source-backed fact.
- Tool activity shows a prominent current step, then a collapsible chronological log above the assistant response. The log collapses when final-answer streaming starts and can be reopened. Tool names, connector names, status, elapsed time, and safe result summaries are persisted with the assistant message and restored when the conversation is reopened. Raw tool arguments and response bodies are never stored in this log or shown in the UI. Calls still running when a turn ends or the app exits are marked failed or interrupted.
- The tool schemas sent to the agent are capped at 15 per turn, including the built-in date/time tool and local document search. The backend enforces the cap after connector discovery, and the connector selector reserves a slot for the built-in tool plus a slot for document search when collections are selected.
- Input and output token usage is requested from the local OpenAI-compatible endpoint and aggregated across the agent's model calls. Pending turns show thinking and current tool activity without an ETA; completed messages retain per-turn token, throughput, and elapsed-time metrics.
- Output throughput is based on the interval between the first and last observed non-empty response chunks. It is omitted when that interval is unavailable or shorter than 10 ms, rather than displaying an inflated rate from millisecond rounding.
- LangChain text streams to the chat UI. Tool-enabled runs with at least one tool call stream a separate final-answer call after tool execution; pre-tool text and tool arguments/results are withheld.
- Local document-search results are persisted as source records for the assistant message and appear in its expandable Sources section. The agent prompt requires exact `[Source N]` references for relevant returned excerpts, but the model's adherence to that instruction still requires live model testing.
- Audio-bearing histories keep using the existing direct model path because LangChain's message format does not represent the app's audio payload. Image attachments are supported on the agent path.
- While a response is pending, the user message can still be copied, but its edit/delete controls are not shown. The composer accepts only one submission at a time: it is locked while model status/loading is awaited, then its Send button becomes Stop. The backend rejects a new turn while any assistant response is pending, and serializes model wake/load requests. The send path also checks/wakes the chat model before starting inference to cover an idle-unload race.

## Enabling MCP

The Cargo feature is opt-in and not part of the default build:

```powershell
cargo test --offline --manifest-path src-tauri\Cargo.toml --features mcp-demo
```

For a local app build, pass `mcp-demo` through the Tauri CLI's Cargo feature option, for example `cargo tauri dev --features mcp-demo`. Without the feature, connector-management and connector-invocation commands are not registered, and the MCP page, dialog, navigation, and composer control are removed from the UI.

The feature currently supports Streamable HTTP connectors only. Add a connector on the Connectors page with a name, endpoint, and optional custom HTTP headers. Remote endpoints must use HTTPS; plain HTTP is limited to loopback addresses. Header values are stored in encrypted local SQLCipher storage and are never returned to the UI after saving. Editing does not reveal saved secrets; explicitly replace or clear them using the authentication-header controls. Test a connector before selecting it in chat.

Connector availability is checked every ten minutes while the chat model is loaded, and users can request an immediate test. The chat selector shows availability and discovered tool count. Connector selection is persisted for the conversation and inherited across its timelines. A connection failure during a check or tool call marks the connector unavailable, clears its conversation selections, and notifies the user. Reconnect it from the Connectors page before selecting it again.

## Local Calendar demo server

The read-only `Calendar` demo server exposes `list_events` and `search_events` over Streamable HTTP. Its sample events are generated in memory and reset when the server restarts. From the repository root, run:

```powershell
uv sync --project qa
uv run --project qa python qa\calendar_mcp_server.py
```

Keep that terminal open while testing. Add a connector named `Calendar` with endpoint `http://127.0.0.1:8001/mcp`; no authentication headers are needed. Then ask the chat agent about today's schedule or search for an event. To run the server on another local port, append `--port 8002` and update the endpoint accordingly.

Verify the server's tools without starting the app:

```powershell
uv run --project qa python qa\test_calendar_mcp_server.py
```

Only configure MCP servers that are trusted to receive the prompts and context supplied to their tools. Hiding tool arguments/results from the chat is a presentation choice, not a data-loss or authorization boundary.

## Verification and limitations

The Rust tests exercise the local agent, SQLite memory/windowing, MCP discovery/calls, connector validation, selection clearing, and error handling. Both default and `mcp-demo` test suites pass. The Calendar server was tested in-process and over Streamable HTTP with FastMCP's client. The feature-enabled UI was checked in a local browser with a mocked Tauri bridge; full Tauri/WebView runtime behavior has not been verified.
