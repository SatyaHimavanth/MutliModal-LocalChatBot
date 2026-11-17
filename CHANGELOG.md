# Changelog

## 1.0.0 — Initial Windows release

### Added

- Local streamed chat with LFM3-Omni, image/audio chat attachments, Markdown responses, and saved conversation timelines.
- Local collections with document extraction, embeddings, hybrid vector and BM25 retrieval, and cited source excerpts.
- Windows user-scoped encrypted chat history and app-managed llama.cpp startup and idle unloading.
- Windows 11 x64 per-user installer and a separate setup guide for provisioning llama.cpp and model files.

### Supported platform

- Windows 11 x64 on CPU.

### Known limitations

- macOS packaging and credential storage are not implemented.
- `langchain-rust` agent/checkpointer orchestration and broad hardware/runtime validation are not complete.
- Image indexing may exceed the conservative memory profile; precise image reasoning is unreliable on the example chat model.
- Audio retrieval segments audio but does not transcribe it; speech retrieval quality is not established.
- The installer is unsigned and does not bundle llama.cpp or model weights. See [`docs/setup-guide.md`](docs/setup-guide.md).
