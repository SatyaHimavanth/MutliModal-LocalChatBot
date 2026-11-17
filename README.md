# Multimodal Local RAG Chatbot

A local-first desktop RAG chatbot/agent in active implementation. The stack is Tauri for the desktop shell and process lifecycle, Rust/`langchain-rust` for chat-agent orchestration, document handling and retrieval, llama.cpp for local model serving, and SQLite/`sqlite-vec` for employee-local history and vector search. An optional MCP connector demo is guarded by a default-off Cargo feature.

## Status

The repository has a Tauri 2 + Rust shell with a production `langchain-rust` agent, streamed local responses, SQLite-backed persistent agent memory, on-demand local collection search with citations, safe Markdown rendering, supported image/audio input, document import, text/media embeddings, saved source excerpts, and forked conversation timelines. Audio-bearing histories retain the direct model path because LangChain messages do not represent the app's audio payload. Collection management shows extracted document text where available. The app starts the local llama.cpp model router and stops it when the app exits. Windows stores conversations, extracted text, chunks, media payloads, vectors, and agent memory in encrypted SQLite/SQLCipher with a random key protected by user-scoped DPAPI. Supported media formats are PNG/JPEG/BMP/GIF/TGA and WAV/MP3/FLAC, up to 8 MB per file; chat media is capped at 16 MB per message. WebP is not accepted because the pinned server build does not document WebP decoding. Image indexing is not guaranteed under the conservative memory configuration: endpoint QA needed `-ub 2048`, which is deliberately not enabled. Durable mid-run agent checkpoints, macOS keychain storage and file picker, memory-aware context sizing, and installer validation remain ahead. See [`docs/ingestion.md`](docs/ingestion.md), [`docs/retrieval.md`](docs/retrieval.md), [`docs/conversations.md`](docs/conversations.md), and [`docs/agent-and-mcp.md`](docs/agent-and-mcp.md) for behavior and limits.

Collection indexing processes files independently: a file whose embeddings fail is removed from the destination collection, while other files in the batch continue. Long audio is decoded and split into overlapping segments locally before embedding so it stays within llama.cpp's default physical batch size. Pending chat turns show the current thinking/tool step without an ETA.

The first release target is Windows 11 x64 on CPU. macOS is a planned platform. IT distributes the application, llama.cpp runtime, and model files.

Text files attached in chat are extracted into the user turn without entering a collection; supported image/audio attachments are passed to the chat model and shown with image previews or audio controls. Attachments are stored with the encrypted message and follow its saved timeline. Direct text attachments are limited to approximately 5,000 estimated tokens. Users can add each attachment to collections after a response.

## Current assets

| Path | Purpose |
|---|---|
| `vendor/llama.cpp/windows-x86_64/` | llama.cpp Windows x64 CPU runtime, currently build `b12359` |
| `models/LFM3-Omni/` | LFM3-Omni chat model and multimodal projector |
| `models/EmbeddingGemma-2/` | EmbeddingGemma 2 embedding model and projector |
| `qa/test_embedding_modalities.py` | Reproducible local text/image/audio endpoint smoke test |
| `qa/artifacts/` | Generated local QA results and logs (gitignored) |
| `models/TEST_MODELS.md` | Model/server setup and test notes |
| `models/start-servers.bat` | Windows developer helper for launching the intended one-server model router |

Model weights, vendor runtimes, QA output, build products, and employee-local databases are excluded by `.gitignore`. The QA script, test notes, and developer batch helper remain trackable. End users launch the app executable; they do not need to run the batch file.

## Development setup

Requirements: Rust stable, the platform prerequisites in the [Tauri setup guide](https://v2.tauri.app/start/prerequisites/), and Perl on build machines for compiling vendored OpenSSL used by SQLCipher (Strawberry Perl was installed on this Windows machine). Perl is build-time only; end-user devices do not need it.

```powershell
cargo install tauri-cli --version "^2.0.0" --locked
cargo tauri dev
cargo tauri build
```

The app expects the IT-provisioned `vendor/llama.cpp/<platform>/` and `models/` directories beside the executable in a packaged deployment. In development, it resolves them from the repository root. It starts `llama-server` on `127.0.0.1:8080` using `--models-max 2`, then loads both configured models. Tauri keeps them loaded during UI activity and active chat/import work, and unloads them after five idle minutes. It does not depend on `models/start-servers.bat`.

## LangChain agent and MCP demo

The production text/image chat path uses `langchain-rust`'s agent executor and encrypted SQLite memory. The agent always has a built-in `get_current_datetime` tool for the computer's local date, time, weekday, and UTC offset. For undated or relative calendar/schedule requests, the prompt requires the agent to call it before a calendar MCP tool and use its local date rather than trusting the server's implicit “today.” The system prompt explicitly names `search_local_documents`: the agent checks relevant selected MCP tools first, then calls `search_local_documents` when no connector is relevant or its results are incomplete. The tool searches indexed documents in the collections selected for that turn. If tools provide no relevant evidence, any fallback from pretrained knowledge must be labeled as general knowledge, not as a verified source fact. Returned document sources are persisted with the assistant message and cited where relevant. At most 15 tools, including the built-in date/time tool and local document search, may be sent to the agent. The default build has no MCP connector commands or controls. An optional `mcp-demo` feature adds custom Streamable HTTP connectors, encrypted custom headers, connection checks, and per-conversation selection. It is an experimental demo and is not authorized for organizational use. Tool inputs and response bodies stay hidden; the expandable activity log shows names, status, timing, and safe summaries, persists when the conversation is reopened, and appears above the assistant response. Output token throughput is based on observed response-chunk timing and is omitted when the interval is too short to measure reliably. While a response is pending, message copy remains available but edit/delete actions are withheld. Duplicate submissions are blocked during model startup/loading and while a response is pending; the backend also rejects starting another turn while an assistant response is pending. Audio-bearing histories retain the direct model path.

Run the deterministic local tests without starting llama.cpp or an MCP server:

```powershell
cargo test --offline --manifest-path src-tauri\Cargo.toml
cargo test --offline --manifest-path src-tauri\Cargo.toml --features mcp-demo
```

The Rust tests use scripted agent/model behavior, SQLite, and an in-memory MCP protocol server; they do not make model or network requests. The user-managed connector transport currently supports Streamable HTTP only. A local Calendar server is included under `qa/`; its endpoint was checked with FastMCP's client, but Rust-app interoperability and full Tauri/WebView runtime behavior remain unverified. LangChain can emit planning text before a tool call, so the app suppresses pre-tool text when tools are enabled and streams a separate final response after tool use. See [`docs/agent-and-mcp.md`](docs/agent-and-mcp.md) for feature behavior, Calendar server commands, and limitations.

## Intended architecture

| Area | Responsibility |
|---|---|
| Tauri desktop app | UI, app lifecycle, supervised local server lifecycle, OS integration |
| Rust application layer | Document extraction/chunking and direct llama.cpp embedding/retrieval calls; `langchain-rust` agent orchestration, persistent SQLite memory, and tool events |
| llama.cpp | Local chat and embedding model router on loopback; models sleep/offload while idle |
| SQLite + `sqlite-vec` + FTS5 | Per-profile chat/collection metadata, encrypted agent memory, local sources and vectors, hybrid similarity and BM25 search |
| OS credential store | Protect a random local data-encryption key; Windows DPAPI first, macOS Keychain when supported |

One router server is started by the Tauri app using the user's command shape from the application root:

```powershell
vendor\llama.cpp\windows-x86_64\llama-server.exe --models-dir ".\models" --host 127.0.0.1 --port 8080 --models-max 2
```

The app resolves the server and model paths relative to the repository root in development and relative to the executable/install directory in packaged builds. The router command has not yet been verified end-to-end with model loading and the current nested model folders. The endpoint QA script tested EmbeddingGemma 2 in direct single-model mode on port 18080.

Hybrid retrieval runs sqlite-vec similarity and SQLite FTS5/BM25 keyword search in parallel, combines their rankings with reciprocal rank fusion, and caps context at six excerpts. Text chunks use 900 characters with 120-character overlap and EmbeddingGemma's 768-dimensional vectors. New chats select all collections; users may choose none or a subset. The selection is saved with each turn, and retrieved source excerpts are saved with the answer. See [`docs/retrieval.md`](docs/retrieval.md). Hybrid retrieval and the app's text embedding batches of up to four have not yet been runtime validated with the packaged model router.

The Rust dependency graph resolved, including `langchain-rust 8.0.0`, `sqlite-vec 0.1.9`, `rusqlite 0.38.0`, Tauri 2.12.5, and `tauri-build 2.7.4`. `cargo fmt`, `cargo check --manifest-path src-tauri/Cargo.toml --offline`, and both default and `mcp-demo` Rust test suites pass on the current Windows development machine. Perl is a build-time prerequisite for vendored OpenSSL. These checks do not build an installer or exercise encrypted sqlite-vec initialization, GUI/server startup, live MCP interoperability, or end-to-end retrieval. Windows DPAPI and SQLCipher history are implemented; macOS keychain storage is not.

## Confirmed v1 scope

- Import `.pdf`, `.docx`, `.xlsx`, `.csv`, `.pptx`, `.txt`, `.md`, PNG/JPEG/BMP/GIF/TGA, and WAV/MP3/FLAC. Media import is bounded at 8 MB and uses one-item embedding requests. Image success depends on server memory settings; do not enable `-ub 2048` without benchmarking against the target memory budget. WebP is not supported by the pinned server build.
- Embed text, images, and audio with EmbeddingGemma 2 for cross-modal retrieval. Do not add a transcription model in v1.
- Defer video ingestion. The current Windows llama.cpp archive does not include FFmpeg/FFprobe, which llama.cpp needs for video files.
- Use a streaming SHA-256 file hash for exact duplicate detection. Store source paths separately; reuse extraction/chunks/vectors across collections only when the full ingestion fingerprint matches.
- Collections are retrieval filters within one employee profile. New chats default to all collections. Users may choose none or any subset. Mid-chat changes affect future turns only and should warn that retrieval context is changing.
- Editing an earlier input creates a new timeline. A branch inherits its fork-point collection selection unless the user changes it during the edit; earlier branches and citations remain intact.
- Deleting a user turn or assistant response updates the selected timeline in place and removes all later messages from that timeline. Other edited timelines are preserved; removing an assistant response leaves its user prompt so it can be regenerated.
- New turns and pending assistant responses are saved before generation, so switching chats preserves their in-progress view. The active chat is marked in the sidebar; Stop cancels generation and saves partial output with an interruption note.
- Keep chat, index, and agent memory local and encrypted. Protect a random data key with user-scoped Windows DPAPI. No recovery mechanism is planned unless policy requires one.
- Budget for only 2–5 GB of available RAM, sometimes 2–3 GB. Idle model offload and context selection from conservative live memory measurements are requirements. Candidate context sizes are 32k/64k/128k, not yet validated across target devices.
- The UI should be simple and make sources, connector progress, and per-turn performance details easy to inspect. See [`docs/ui-reference.md`](docs/ui-reference.md).

## Endpoint QA

Run the smoke test from the repository root:

```powershell
python qa\test_embedding_modalities.py
```

The script starts the local `b12359` server in CPU embedding mode, generates a synthetic PNG and WAV in memory, tests text/image/audio requests to `/v1/embeddings`, writes JSON results and a server log under `qa/artifacts/`, and stops only the server process it launched. It requires the model/runtime files listed in the script. No downloads are required.

Latest result: text, image, and audio each returned a finite 768-dimensional vector. Image embedding required `-ub 2048`; `-ub 64` caused model initialization to fail because the projector's image pixel minimum was not met. The successful setting is a smoke-test configuration, **not** a production memory recommendation. The audio fixture was a synthetic tone, so it verifies request acceptance and vector output, not useful audio semantics or retrieval relevance. Video was not tested. Generated JSON and server logs are local and gitignored.

See [`docs/embedding-endpoint-qa.md`](docs/embedding-endpoint-qa.md) for limitations and evidence, and [`models/TEST_MODELS.md`](models/TEST_MODELS.md) for current model details.

## Project guidance

- [`AGENT.md`](AGENT.md): canonical engineering rules and boundaries.
- [`AGENTS.md`](AGENTS.md): Codex entry point to the canonical guide.
- [`MEMORY.md`](MEMORY.md): accepted decisions, open risks, and current state.
- [`docs/ui-reference.md`](docs/ui-reference.md): written interpretation of the supplied llama.cpp UI screenshots.
- [`docs/ingestion.md`](docs/ingestion.md): current document-import behavior, exact duplicate reuse, and format limits.
