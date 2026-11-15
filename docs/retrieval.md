# Text retrieval (current slice)

## Import and indexing

- Text extracted by the document importer is split into overlapping 900-character chunks, preferring paragraph and sentence boundaries. Overlap is 120 characters.
- Each PNG/JPEG image becomes one media chunk; audio is decoded and split into eight-second windows with one-second overlap, resampled to 16 kHz mono WAV, and embedded separately. Audio is not transcribed. At answer time, retrieval gives the chat model a segment label and filename for an audio match, not the audio payload. The UI can cite that source filename, but the chat model cannot ground a description of collection audio in the original payload yet.
- Newly imported or not-yet-indexed chunks are embedded in small batches through the single llama.cpp router's `/v1/embeddings` endpoint. The model is selected by an `EmbeddingGemma` ID in `/v1/models`.
- The router needs an embedding-mode preset for that model. `models/app-models.ini` configures `EmbeddingGemma-2` with `embedding = true` and `pooling = mean`; `src-tauri` passes this preset at startup. Without it, the router can discover the model but return HTTP 501 for `/v1/embeddings`.
- The current implementation expects the verified 768-dimensional EmbeddingGemma output. Values are checked for numeric finiteness and dimension before storage.
- A persisted index fingerprint covers parser/chunking settings and the expected embedding model/dimension contract. Changing that fingerprint clears derived chunks and vectors so they can be rebuilt; it does not include a cryptographic hash of the provisioned model files yet.
- Chunk text, vectors, source filenames, and citations live in the SQLCipher database. sqlite-vec performs nearest-neighbor search; SQLite FTS5 provides BM25 keyword ranking. Neither creates embeddings.
- A failed indexing request is shown per file with its reason and an in-place retry action available once. A failed media embedding is not left as a collection member. If the retry fails, the reason remains visible and the user can drop the file again later. No automatic retry scheduler exists yet.
- If the parser version changes, the old chunks and vectors for that document are deleted and rebuilt from refreshed extracted text.
- Collection management lists source filenames and paths. Removing documents deletes their collection membership; extracted content and vectors are deleted only after the last collection reference is removed.

## Chat behavior

- New chats start with all available collections selected. A user can choose any subset, including none.
- Collection selection is saved per user turn. Changing it during an existing chat warns that the new selection affects future turns only.
- Retrieval embeds the new user query, searches up to 160 nearest vector chunks, and runs a parallel FTS5/BM25 search over up to 60 matching chunks within selected collections. Reciprocal rank fusion combines both rankings, deduplicates chunks, and includes at most six excerpts in the chat context. This bounds prompt growth while letting exact names and terms complement semantic matches.
- Excerpts are explicitly described to the model as untrusted reference text. The answer stores its source filename, excerpt, and vector distance in the encrypted database. The UI shows those sources under the response.
- Every chat request includes a default workplace-assistant system prompt. Retrieved excerpts are appended to it as untrusted evidence with `[Source N]` citation guidance; the prompt asks for clear, concise answers and Markdown where useful.
- Chat completions use llama.cpp's OpenAI-compatible SSE stream. Rust forwards text deltas as Tauri events while the request is active; the completed answer and usage metrics are then saved to the conversation. The UI renders completed assistant text as safe Markdown without inserting model HTML.
- Chat paperclip prepares text-bearing files as message attachments without creating collection membership or vectors. Up to about 5,000 estimated tokens of extracted text may be attached to a turn; the encrypted message stores attachment text and file path so later turns and editing retain context. Supported image/audio attachments are passed to the chat model in multimodal content parts and shown again in the assistant's expandable Sources section; after a response they may be added to selected collections. The Windows file picker also allows the same files to be imported into collections.
- The sqlite-vec KNN query uses its `k` virtual-table constraint (`k = ?`) to request up to 160 candidates before selected-collection filtering. FTS5 index triggers track inserted, updated, and deleted chunks; existing chunks are backfilled at database open.
- Collection indexing reports completed embedding chunks and an approximate remaining time. Text progress advances by embedding batch and media progress by each single-item embedding request; vectors are stored after embedding completes.
- When no indexed documents exist in selected collections, the app sends a normal chat request without retrieval.

## Limits and next work

- The candidate search currently fetches up to 160 nearest chunks globally before collection filtering. Very large corpora with many unrelated collections may need a collection-aware vector index.
- Hybrid retrieval quality, latency, throughput, router model switching, and peak RAM have not been runtime-validated with the packaged router. BM25 currently searches extracted text and filenames; raw image/audio payloads remain vector-only. The previous endpoint QA tested direct single-model mode and one item per request. The app sends small batches of up to four text inputs.
- PNG/JPEG files are stored as encrypted media blobs and submitted one at a time using EmbeddingGemma's typed embedding input. Image embedding may fail under conservative runtime batch settings; the endpoint QA required `-ub 2048` for its image fixture, which this app intentionally does not set because of the low-memory target. Long audio previously failed when a 60-second MP3 expanded to 1,504 tokens against the default physical batch of 512. Audio segmentation now keeps each eight-second overlapping WAV window below that limit in the tested fixture. Failed media indexing is surfaced rather than marked searchable. Useful audio retrieval quality has not been validated.
- Media retrieval, document removal/reindexing, import cancellation, and `langchain-rust` agent/checkpointer orchestration still need end-to-end runtime validation.
- Direct image chat accepted a typed image payload in the local LFM3-Omni router, but a controlled chessboard fixture was read inaccurately. Treat precise visual identification as unverified and unreliable; see [`chat-image-qa.md`](chat-image-qa.md).
- `langchain-rust` is present as a dependency but this retrieval path currently uses direct typed Rust code and HTTP calls; agent graph/checkpointer orchestration has not been wired in.
- Context-window selection from live memory, token-aware history trimming, and `-b`/`-ub` benchmarks remain open.
