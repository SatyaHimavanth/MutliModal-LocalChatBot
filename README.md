# Multimodal Local RAG Chatbot

A local-first desktop RAG chatbot/agent in active implementation. The stack is Tauri for the desktop shell and process lifecycle, Rust for document handling and retrieval, llama.cpp for local model serving, and SQLite/`sqlite-vec` for employee-local history and vector search. `langchain-rust` is installed but agent/checkpointer orchestration is not wired in yet.

## Status

The repository has a Tauri 2 + Rust shell with streamed local chat, safe Markdown assistant rendering, direct supported image/audio chat input, a workplace assistant prompt, document import, text/media embeddings, hybrid collection retrieval, saved source excerpts, and forked conversation timelines. Collection management shows extracted document text where available. The app starts the local llama.cpp model router and stops it when the app exits. Windows stores conversations, extracted text, chunks, media payloads, and vectors in SQLCipher with a random key protected by user-scoped DPAPI. Supported media formats are PNG/JPEG/BMP/GIF/TGA and WAV/MP3/FLAC, up to 8 MB per file; chat media is capped at 16 MB per message. WebP is not accepted because the pinned server build does not document WebP decoding. Image indexing is not guaranteed under the conservative memory configuration: endpoint QA needed `-ub 2048`, which is deliberately not enabled. Full `langchain-rust` agent/checkpointer integration, macOS keychain storage and file picker, memory-aware context selection, and installer validation remain ahead. See [`docs/ingestion.md`](docs/ingestion.md), [`docs/retrieval.md`](docs/retrieval.md), and [`docs/conversations.md`](docs/conversations.md) for behavior and limits.

Collection indexing processes files independently: a file whose embeddings fail is removed from the destination collection, while other files in the batch continue. Long audio is decoded and split into overlapping segments locally before embedding so it stays within llama.cpp's default physical batch size. Pending chat turns use llama.cpp prompt-progress counters when available and show a size-based approximate estimate before then.

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

## Intended architecture

| Area | Responsibility |
|---|---|
| Tauri desktop app | UI, app lifecycle, supervised local server lifecycle, OS integration |
| Rust application layer | Document extraction/chunking and direct llama.cpp embedding/retrieval calls; `langchain-rust` graph, tool events, and checkpointed timelines are planned |
| llama.cpp | Local chat and embedding model router on loopback; models sleep/offload while idle |
| SQLite + `sqlite-vec` + FTS5 | Per-profile chat/checkpoint/collection metadata, local sources and vectors, hybrid similarity and BM25 search |
| OS credential store | Protect a random local data-encryption key; Windows DPAPI first, macOS Keychain when supported |

One router server is started by the Tauri app using the user's command shape from the application root:

```powershell
vendor\llama.cpp\windows-x86_64\llama-server.exe --models-dir ".\models" --host 127.0.0.1 --port 8080 --models-max 2
```

The app resolves the server and model paths relative to the repository root in development and relative to the executable/install directory in packaged builds. The router command has not yet been verified end-to-end with model loading and the current nested model folders. The endpoint QA script tested EmbeddingGemma 2 in direct single-model mode on port 18080.

Hybrid retrieval runs sqlite-vec similarity and SQLite FTS5/BM25 keyword search in parallel, combines their rankings with reciprocal rank fusion, and caps context at six excerpts. Text chunks use 900 characters with 120-character overlap and EmbeddingGemma's 768-dimensional vectors. New chats select all collections; users may choose none or a subset. The selection is saved with each turn, and retrieved source excerpts are saved with the answer. See [`docs/retrieval.md`](docs/retrieval.md). Hybrid retrieval and the app's text embedding batches of up to four have not yet been runtime validated with the packaged model router.

The Rust dependency graph resolved, including `langchain-rust 8.0.0`, `sqlite-vec 0.1.9`, `rusqlite 0.38.0`, Tauri 2.12.5, and `tauri-build 2.7.4`. `cargo fmt` and `cargo check --manifest-path src-tauri/Cargo.toml --offline` pass on the current Windows development machine. Perl is a build-time prerequisite for vendored OpenSSL. This check does not build an installer or exercise encrypted sqlite-vec initialization, GUI/server startup, or end-to-end retrieval. Windows DPAPI and SQLCipher history are implemented; macOS keychain storage is not.

## Confirmed v1 scope

- Import `.pdf`, `.docx`, `.xlsx`, `.csv`, `.pptx`, `.txt`, `.md`, PNG/JPEG/BMP/GIF/TGA, and WAV/MP3/FLAC. Media import is bounded at 8 MB and uses one-item embedding requests. Image success depends on server memory settings; do not enable `-ub 2048` without benchmarking against the target memory budget. WebP is not supported by the pinned server build.
- Embed text, images, and audio with EmbeddingGemma 2 for cross-modal retrieval. Do not add a transcription model in v1.
- Defer video ingestion. The current Windows llama.cpp archive does not include FFmpeg/FFprobe, which llama.cpp needs for video files.
- Use a streaming SHA-256 file hash for exact duplicate detection. Store source paths separately; reuse extraction/chunks/vectors across collections only when the full ingestion fingerprint matches.
- Collections are retrieval filters within one employee profile. New chats default to all collections. Users may choose none or any subset. Mid-chat changes affect future turns only and should warn that retrieval context is changing.
- Editing an earlier input creates a new timeline. A branch inherits its fork-point collection selection unless the user changes it during the edit; earlier branches and citations remain intact.
- Deleting a user turn or assistant response updates the selected timeline in place and removes all later messages from that timeline. Other edited timelines are preserved; removing an assistant response leaves its user prompt so it can be regenerated.
- New turns and pending assistant responses are saved before generation, so switching chats preserves their in-progress view. The active chat is marked in the sidebar; Stop cancels generation and saves partial output with an interruption note.
- Keep chat, index, and checkpoints local and encrypted. Protect a random data key with user-scoped Windows DPAPI. No recovery mechanism is planned unless policy requires one.
- Budget for only 2–5 GB of available RAM, sometimes 2–3 GB. Idle model offload and context selection from conservative live memory measurements are requirements. Candidate context sizes are 32k/64k/128k, not yet validated across target devices.
- The UI should be simple and make sources, tool calls, and per-turn performance details easy to inspect. See [`docs/ui-reference.md`](docs/ui-reference.md).

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
