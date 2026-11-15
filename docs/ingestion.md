# Document import (current slice)

## Current behavior

- Create a named collection or use the default `General` collection.
- Select the import destination and drop PDF, DOCX, XLSX, CSV, PPTX, PNG, JPEG, BMP, GIF, TGA, WAV, MP3, or FLAC files into the app window. Plain text and Markdown are also accepted by the importer.
- Individual files are limited to 50 MB, each drop batch to 20 files and 100 MB total. Expanded Office XML is capped at 20 MB per import, and extracted text at 1.5 million characters.
- The app streams a SHA-256 hash before parsing. Matching content reuses the stored extracted text and attaches the new source path and collection membership. Different file names and source paths remain separate metadata.
- Extracted text, hash, and source paths are stored in the encrypted local SQLCipher database. The original source files stay where the employee keeps them.
- Supported image and audio files are stored as encrypted database blobs (maximum 8 MB each) and sent to EmbeddingGemma 2. Images are embedded individually. Audio is decoded locally with Symphonia, resampled to 16 kHz mono, and split into eight-second windows with one-second overlap before embedding. This keeps long recordings within llama.cpp's default 512-token physical batch; audio longer than 20 minutes is rejected with a clear reason. The current image projector needs a high `-ub` value in endpoint QA; production settings intentionally do not enable that value, so image indexing can fail under the conservative server configuration. Such failures are reported and the import remains available for retry. Audio embedding acceptance is verified for synthetic fixtures; semantic retrieval quality is not established. WebP is not accepted because the pinned llama.cpp server build does not document its decoder support.
- Chat attachments support the same server-documented image/audio formats and are passed directly to LFM3-Omni; media is limited to 8 MB per file and 16 MB per message. Attached images show a bounded preview and audio files have native playback controls. Text-bearing attachments remain capped at approximately 5,000 estimated tokens.
- Collection file rows include a content viewer. It displays extracted text for text-bearing documents; image/audio files show a note that no textual extraction was performed while their binary payload remains available for cross-modal indexing.
- PDF import extracts embedded text. Scanned/image-only PDFs require OCR, which is not part of this slice.
- Collection imports report file extraction and embedding progress in the Collections page, including the destination collection, current filename, completed chunk count, and a rough ETA based on the current operation's observed rate. The collection card shows a spinner while its import/indexing is active. Extraction progress is per file and embedding is isolated per document; parser-internal progress and cancellation are not available.
- Indexing is isolated per document. If embedding or saving a document's vectors fails, that document is removed from the destination collection and its orphaned cached content/chunks/vectors are deleted; other files in the same batch continue. A failed file is reported as failed and is not counted as imported.
- The import status card shows an individual state and error for every file, and offers one in-place retry for failures. It cannot be dismissed while work is active. It auto-clears eight seconds after every file succeeds; partial or failed imports stay visible until dismissed. Dismissing a failed/partial import warns that only successfully indexed files are saved.
- While the assistant is processing a turn, the chat shows elapsed time and an approximate remaining-time estimate. For the active model request it polls llama.cpp's `/slots` counters for actual prompt size, prompt tokens processed, and cached tokens, then updates from measured prompt throughput. Before the server exposes prompt counts, media turns say the estimate is being prepared; text turns use a conservative size-based fallback. Output time uses observed generation speed. The estimate is still approximate and not a completion guarantee.

## Not yet implemented

- WebP, video, and quality evaluation for cross-modal retrieval.
- PowerPoint image/chart extraction, spreadsheet embedded-image extraction, OCR, parser-internal progress/cancellation, folder watching, and document removal/re-indexing.
- A native file picker; this slice uses Tauri's native file-drop event. The backend accepts only paths recorded from a native drop event.

Text chunks, embeddings, retrieval, citations, and collection-aware chat are now implemented. See [`retrieval.md`](retrieval.md) for that path and its limitations.

## Cache behavior

The document cache is keyed by source file SHA-256 and parser version (`multimodal-v1`). A separate index fingerprint covers the current parser/chunker settings and expected EmbeddingGemma model/dimension contract. When the parser changes for an imported file, its old chunks/vectors are discarded. The fingerprint does not include a cryptographic model-file hash or preprocessing configuration yet.
