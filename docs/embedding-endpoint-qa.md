# Embedding Endpoint QA

## Result

The local Windows x64 CPU llama.cpp `b12359` build successfully returned a 768-dimensional finite vector for each tested input: text, PNG image, and WAV audio.

The test script writes a machine-readable result and server log under the local, gitignored `qa/artifacts/` directory. Re-run using [`../qa/test_embedding_modalities.py`](../qa/test_embedding_modalities.py).

## Environment and method

- Server binary: `vendor/llama.cpp/windows-x86_64/llama-server.exe`
- Version: `1.0.0-dev (build 12359, commit b1b770d90)`, Windows x86_64
- Model: `models/EmbeddingGemma-2/embeddinggemma-2-Q8_0.gguf`
- Projector: `models/EmbeddingGemma-2/mmproj-Q8_0.gguf`
- Server mode: direct single-model CPU embedding mode, `--embedding --pooling mean`, loopback port 18080, one parallel slot
- Test data: plain text, an in-memory synthetic 64x64 PNG, and an in-memory one-second 16 kHz mono WAV tone
- Successful test batch settings: `-b 2048 -ub 2048`
- The test process shut down the server it started after completion.

## What this proves

- The local endpoint accepts the tested text, image, and audio request shapes and returns correctly shaped finite embeddings.
- The model metadata advertised `text`, `image`, `audio`, and `video` input modalities.

## What this does not prove

- Audio embeddings are semantically useful for organizational audio or cross-modal retrieval. The fixture was a synthetic tone, not speech or environmental audio.
- Retrieval quality, ranking, chunking, or source citation behavior.
- Video embedding or video extraction. Video remains out of v1, and the checked llama.cpp archive does not include FFmpeg/FFprobe.
- That `-b 2048 -ub 2048` fits within the app's 2–5 GB available RAM budget. The values were used to satisfy the image projector's minimum pixel requirement during this smoke test, not recommended as production defaults.
- That the desired `--models-dir` router discovers and loads the current nested model directories and projector sidecars correctly. This run used direct single-model mode.

## Configuration finding

An initial run with `-ub 64` failed before accepting requests. llama.cpp capped the projector image-token budget below EmbeddingGemma 2's minimum image-pixel requirement and refused to load the projector. The successful run used `-ub 2048`. This makes image quality/capability a memory-tuning constraint that needs dedicated measurements on target devices.

## Collection audio indexing follow-up

On 2026-11-15, the app's live `b12359` model router reproduced the collection-import failure with a generated 60-second MP3 sent as one `/v1/embeddings` audio input. The server returned HTTP 500: `input (1504 tokens) is too large to process. increase the physical batch size (current batch size: 512)`. The earlier audio QA passed only a one-second WAV, so it did not cover long recordings.

The app now decodes WAV/MP3/FLAC locally with Symphonia, converts audio to 16 kHz mono WAV, and emits eight-second segments with one-second overlap. The 60-second fixture produced nine segments; all nine were accepted by the live router and returned 768-dimensional vectors without changing `-ub`. A Rust unit test checks segment count, overlap, WAV output, and segment duration.

This confirms endpoint acceptance and avoids the observed 512-token failure for this fixture. The generated fixture is a tone, so speech retrieval quality and target-device peak memory still need evaluation.
