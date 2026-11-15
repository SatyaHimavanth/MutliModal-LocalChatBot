# Project Agent Guide

## Product direction

Build a local-first multimodal RAG chat application for organizational employees. The first release targets Windows 11 x64 on CPU. Keep macOS as a planned platform and preserve platform-specific boundaries rather than assuming Windows APIs everywhere.

The product is in active implementation. The Tauri 2 + Rust shell supervises llama.cpp at app startup and stops it at app exit. Implemented slices include streamed local chat with supported image/audio input, collection management and extracted-content viewing, Windows SQLCipher/DPAPI history, text and media document import, EmbeddingGemma vectors in sqlite-vec, FTS5/BM25 hybrid collection retrieval with reciprocal rank fusion, saved per-turn selection, source excerpts, safe Markdown assistant rendering, a workplace system prompt, and persisted forked timelines. `langchain-rust` agent/checkpointer orchestration, live memory-aware context sizing, macOS credential storage, and installer validation are not implemented yet. Hybrid retrieval, current embedding batches, and full model-router startup have not been runtime validated.

## Architecture constraints

- Tauri owns the desktop shell, application lifecycle, local UI, and supervised llama.cpp process lifecycle.
- Rust currently owns document extraction/chunking and direct llama.cpp embedding/retrieval calls. The intended `langchain-rust` agent orchestration, tool events, and checkpointed conversation state remain to be wired in; do not describe them as working until implemented.
- SQLite owns local chat history, collection metadata, source references, and agent state where supported. `sqlite-vec` stores/searches vectors and FTS5 provides BM25 ranking; neither generates embeddings.
- Use one localhost llama.cpp model-router server, based on the intended command in `README.md`. Do not bring back a two-server/8080+8081 layout without an explicit decision.
- IT provisions the app, llama.cpp runtime, and model files. The app must validate required assets and show actionable errors; it must not silently download models. End users launch only the desktop app executable; a batch helper is for developers.
- First-release data is per employee profile and stays local. Protect the database and other sensitive local artifacts with encryption using a random data key protected by the OS credential store. Use user-scoped Windows DPAPI for Windows; never use a MAC address as a secret. No recovery flow is planned unless organizational policy requires one.

## Product behavior

- Ingest `.pdf`, `.docx`, `.xlsx`, `.csv`, `.pptx`, `.txt`, `.md`, PNG/JPEG/BMP/GIF/TGA images, and WAV/MP3/FLAC audio. Current media limit is 8 MB per file. Preserve source locations and support the same content appearing at multiple paths. WebP requires conversion before it can be sent to the pinned llama.cpp build and is not accepted yet.
- Deduplicate exact files with a streaming SHA-256. Cache extraction/chunks/vectors by content identity plus an ingestion fingerprint (parser/chunker settings and versions, model identity/version/hash, and preprocessing settings). An extracted-text hash may suggest equivalent documents but must not prove equivalence for multimodal files.
- Collections are employee-local retrieval filters over reusable ingestion artifacts, not separate security principals. New chats default to all collections; users may select none or any subset. A mid-chat selection change affects future turns only and should warn that context changes.
- Files added with the chat paperclip are message attachments, not collection imports. Extract text for the user turn, store it with the encrypted message so later turns in that timeline retain its context, cap direct chat attachments at about 5,000 estimated tokens, and let users explicitly add each file to selected collections after the response. Supported images/audio may also be sent directly to the chat model, capped at 16 MB per message. Collection-page imports remain the indexing path for reusable cross-modal retrieval.
- Editing an earlier input creates a new timeline. A fork inherits the edited turn's saved collection selection unless the user changes it during the edit. Keep prior timelines, answers, and citations unchanged.
- Deleting a user turn or assistant response updates only the selected timeline in place: hide that message and all later messages there, while preserving sibling edited timelines. Deleting an assistant response leaves its user prompt available for regeneration.
- Persist the collection selection and retrieved source citations with each answer. Combine vector and BM25 candidates with reciprocal rank fusion, deduplicate, and cap prompt context. Show tool calls, retrieved sources, and timing/performance details in the UI. Stream chat deltas and render assistant Markdown using text-safe DOM construction.
- Index collection documents independently. A document whose embedding or vector persistence fails must not remain a member of the destination collection; remove unreferenced cached content, chunks, FTS entries, and vectors, and continue processing other files in the batch. Show per-file import status and offer one retry for failed files. Keep the status open while processing and after partial/failure outcomes; auto-dismiss only after complete success. Dismissing a failed/partial import must explain that only successful files were saved.
- While a turn is pending, show an explicitly approximate response-time estimate and elapsed time, updating once per second. Poll llama.cpp `/slots` prompt counters when available to account for actual multimodal prompt expansion; use conservative text-size/output-speed fallbacks otherwise. Never present the estimate as guaranteed, and do not expose slot prompt text to the UI.
- Save user turns and pending assistant placeholders before retrieval/generation, so switching chats can reopen the transcript while the stream continues. Mark the active conversation in the sidebar and allow cancellation; save partial output followed by “User interrupted generation.” when stopped.
- The llama.cpp UI screenshot is a visual reference, not a requirement to reproduce its model management. The composer uses a collections selector where the reference shows the model selector, plus a paperclip native file picker. See `docs/ui-reference.md`.

## Resource and model rules

- Assume only 2–5 GB of RAM may be available to the app; competing organizational software can reduce this to 2–3 GB. Offload/sleep idle models and reserve memory for the OS and other apps.
- Context candidates are 32k, 64k, and 128k, but runtime selection must be based on measured model-specific memory behavior and conservative current availability, not a VRAM-only heuristic. Do not claim these sizes are safe on every device before measuring.
- Benchmark `-b` and `-ub` on fixed hardware/model/prompt with warm-up, prompt-processing throughput, generation throughput, peak RAM/VRAM, and failures recorded. Do not treat the QA smoke-test values as production defaults.
- The tested EmbeddingGemma 2 endpoint needed `-ub 2048` to initialize its image projector; a test at `-ub 64` failed its minimum image-pixel requirement. This remains a memory-fit risk.
- EmbeddingGemma 2 is used for text, image, and audio embeddings and cross-modal retrieval. Image embedding uses single-item requests and may fail under current conservative router settings because endpoint QA needed `-ub 2048`. Do not enable that setting without memory benchmarking. Audio is decoded locally to 16 kHz mono and embedded as overlapping eight-second windows to stay within the default 512-token physical batch; the current maximum audio duration is 20 minutes. Audio transcription and video ingestion are out of v1. Do not conflate audio input to the chat model with audio embedding or transcription.

## Change and verification rules

- Read `MEMORY.md`, the relevant docs, and the actual source/config files before changing behavior. Prefer official project documentation when a library or server API is involved.
- Keep vendor binaries and model weights untouched. They are ignored by `.gitignore`, along with generated QA output and local databases. Keep QA source and setup documentation trackable.
- Dependencies for the initial Tauri/LangChain/SQLite foundation are authorized. For later dependency additions, changes to the storage/encryption design, model distribution, or v1 formats/platform scope, document the tradeoff and ask first.
- For llama.cpp behavior, use `qa/test_embedding_modalities.py` and record the exact build, request shape, model files, settings, and result. A successful API response proves endpoint acceptance and vector shape, not retrieval quality or memory suitability.
- Update `README.md`, `MEMORY.md`, and this guide when confirmed product decisions change. Keep `models/TEST_MODELS.md` aligned with filenames that exist in the repository.
- Keep status claims precise: streamed chat, Markdown assistant rendering, collection management with extracted-content viewing, text/media import and retrieval, bounded image/audio embedding requests, image/audio chat attachments, Windows encryption, source persistence, in-chat timeline navigation, and conversation management are implemented; do not imply successful image indexing under the production memory profile, agent/checkpointer orchestration, macOS key storage, or production packaging are complete.

## Source of truth

1. The user's latest explicit decision.
2. `MEMORY.md` for accepted project decisions and current evidence.
3. `README.md` for project status, assets, and runnable commands.
4. This guide for agent workflow and boundaries.
