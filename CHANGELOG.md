# Changelog

## 1.0.0 — Initial application baseline

This version establishes the first reviewable project snapshot for Windows 11 x64 human testing.

### Included

- Tauri desktop shell with local llama.cpp process startup and model lifecycle management.
- Streaming chat, conversation timelines, collections, document ingestion, hybrid retrieval, and source display.
- Encrypted local history and indexed document storage on Windows.
- Text, image, and audio workflows, including locally segmented audio indexing.
- Project setup, QA evidence, and implementation guidance.

### Known limits

- The full `langchain-rust` agent/checkpointer integration is not wired in yet.
- macOS credential storage and file-picker behavior are not implemented or validated.
- Installer packaging and production memory profiles are not validated; image indexing may need more memory than the target budget allows.
- Cross-modal audio retrieval quality and multimodal chat accuracy remain limited and need device-level evaluation.
