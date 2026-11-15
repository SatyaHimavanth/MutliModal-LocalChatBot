# Local Model and llama.cpp QA Notes

Run commands from the repository root (`D:\Projects\MutliModal-LocalChatBot`) unless noted. Paths below match the files currently present in this checkout.

## Local assets

### LFM3-Omni chat model

```text
models/LFM3-Omni/LFM3-Omni-2B-Q4_K_M.gguf
models/LFM3-Omni/mmproj-LFM3-Omni-2B-Q8_0.gguf
```

The model is intended for text and supported multimodal chat input/output. LFM3-Omni is a non-reasoning model; when starting it directly, use `--reasoning off` as described by its llama.cpp model template guidance.

### EmbeddingGemma 2

```text
models/EmbeddingGemma-2/embeddinggemma-2-Q8_0.gguf
models/EmbeddingGemma-2/mmproj-Q8_0.gguf
```

EmbeddingGemma 2 is the planned embedding model for text, image, and audio cross-modal retrieval. It is not an ASR/transcription model in this application. Current llama.cpp model metadata also advertises video input, but video is deferred from v1.

## Intended application server layout

The application is intended to supervise one localhost llama.cpp model-router server, allowing up to two models to be loaded and sleeping idle models after five minutes:

```powershell
vendor\llama.cpp\windows-x86_64\llama-server.exe --models-dir ".\models" --host 127.0.0.1 --port 8080 --models-max 2 --sleep-idle-seconds 300
```

The Tauri application starts this server itself when launched and stops it on app exit. End users do not run `start-servers.bat`; it remains a developer-only helper. The router invocation has not yet been verified with the current nested folders and sidecar names.

## Verified endpoint smoke test

The reproducible smoke test starts the local `b12359` Windows x64 CPU server in **direct single-model embedding mode** on port 18080. It tests `/v1/embeddings` with text, a generated PNG, and a generated WAV. It does not use the intended router mode.

From the repository root:

```powershell
python qa\test_embedding_modalities.py
```

Latest recorded result: all three inputs returned a finite 768-dimensional vector. The test uses `-b 2048 -ub 2048` because a run with `-ub 64` failed to initialize the projector: the image-token budget was capped below EmbeddingGemma 2's minimum pixel requirement. The passing settings are smoke-test settings, not a production recommendation or proof of fit within the 2–5 GB app RAM budget.

The script writes a JSON result and server log to `qa/artifacts/` locally. That generated directory is gitignored; the test script and report stay trackable. See `docs/embedding-endpoint-qa.md` for the recorded summary and limitations.

The audio fixture is a synthetic tone. Its passing result proves that this request shape produced a vector, not that the embedding is useful for real audio or cross-modal ranking.

## API request shape tested

For direct single-model mode, the test uses the `model` ID returned by `GET /v1/models` and sends one typed content part per input to `POST /v1/embeddings`. The media-bearing input is wrapped in a `content` array:

```json
{
  "model": "<model-id-from-v1-models>",
  "input": [
    {
      "content": [
        {
          "type": "image_url",
          "image_url": { "url": "data:image/png;base64,<base64-png>" }
        }
      ]
    }
  ],
  "encoding_format": "float"
}
```

Audio uses a content part with `type: "input_audio"` and a base64 WAV in `input_audio.data`. For exact payload generation, see `qa/test_embedding_modalities.py` and the official [llama.cpp server documentation at b12359](https://github.com/ggml-org/llama.cpp/blob/b12359/tools/server/README.md).

## Runtime notes

- The checked Windows archive is CPU x64 and contains `llama-server.exe` and `mtmd.dll`.
- The archive does not include `ffmpeg.exe` or `ffprobe.exe`; llama.cpp needs FFmpeg for video inputs. Video is not part of v1.
- Bind the local service to `127.0.0.1`; do not expose it on `0.0.0.0` for this desktop app.
- Resolve all packaged paths from the app's resource/install directory rather than relying on the current working directory.
- Per-model context, `-b`, and `-ub` choices must be benchmarked under the expected 2–5 GB available RAM budget. Do not infer safe context from model metadata alone.
