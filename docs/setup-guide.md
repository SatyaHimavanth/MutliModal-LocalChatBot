# Windows setup guide — 1.0.0

This guide installs the Windows 11 x64 CPU build. The app installer does not contain the llama.cpp runtime or model weights. Install those separately, or have IT stage them, before using chat or document indexing.

## Release files

Download from the [1.0.0 GitHub Release](https://github.com/SatyaHimavanth/MutliModal-LocalChatBot/releases/tag/v1.0.0):

- `Multimodal-Local-Chatbot_1.0.0_x64-setup.exe` — Windows per-user installer. The exact filename may include Tauri's product-name punctuation; use the `.exe` ending in `-setup.exe`.
- `setup-guide.md` — this guide.
- `app-models.ini` — model-router preset to place in the `models` directory.
- `*.sha256` — SHA-256 checksum for the installer.

The installer is unsigned. Follow your organization's software approval and code-signing policy before broad deployment. The app is intended for Windows 11 x64; macOS packaging and credential storage are not part of this release.

## 1. Install the application

1. Run the `-setup.exe` installer and install for the current Windows user. The default location is under `%LOCALAPPDATA%\Programs\Multimodal Local Chatbot`.
2. Do not launch the app until the runtime and model files have been placed in the folders below. The app starts and supervises llama.cpp itself; users do not run `start-servers.bat`.
3. Windows 11 normally includes WebView2. In managed or offline environments, IT should confirm the WebView2 Runtime is available before deployment.

## 2. Add the llama.cpp Windows CPU runtime

Use the tested [llama.cpp b12359 Windows x64 CPU archive](https://github.com/ggml-org/llama.cpp/releases/download/b12359/llama-b12359-bin-win-cpu-x64.zip). Newer llama.cpp builds may change model-router behavior and should be validated before substitution.

Extract the full archive contents, including all DLLs, to:

```text
%LOCALAPPDATA%\Programs\Multimodal Local Chatbot\vendor\llama.cpp\windows-x86_64\
```

The folder must contain `llama-server.exe` and its companion DLL files together. Do not copy only the EXE.

## 3. Add the example models

The example filenames below match the model folder layout and app preset. Create the folders exactly as written. Keep the main model and its `mmproj` sidecar together in each model folder.

```text
%LOCALAPPDATA%\Programs\Multimodal Local Chatbot\
├── Multimodal Local Chatbot.exe
├── vendor\llama.cpp\windows-x86_64\
│   ├── llama-server.exe
│   └── [all DLL files from the llama.cpp archive]
└── models\
    ├── app-models.ini
    ├── LFM3-Omni\
    │   ├── LFM3-Omni-2B-Q4_K_M.gguf
    │   └── mmproj-LFM3-Omni-2B-Q8_0.gguf
    └── EmbeddingGemma-2\
        ├── embeddinggemma-2-Q8_0.gguf
        └── mmproj-Q8_0.gguf
```

Create the folder layout under the install directory, then download these files:

| Folder / file | Example download | Approx. size |
|---|---|---:|
| `models/LFM3-Omni/LFM3-Omni-2B-Q4_K_M.gguf` | [LiquidAI official LFM3-Omni GGUF](https://huggingface.co/LiquidAI/LFM3-Omni-2B-GGUF/resolve/main/LFM3-Omni-2B-Q4_K_M.gguf) | 1.1 GB |
| `models/LFM3-Omni/mmproj-LFM3-Omni-2B-Q8_0.gguf` | [LiquidAI official multimodal projector](https://huggingface.co/LiquidAI/LFM3-Omni-2B-GGUF/resolve/main/mmproj-LFM3-Omni-2B-Q8_0.gguf) | 507 MB |
| `models/EmbeddingGemma-2/embeddinggemma-2-Q8_0.gguf` | [ggml-org EmbeddingGemma 2 GGUF](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF/resolve/main/embeddinggemma-2-Q8_0.gguf) | 310 MB |
| `models/EmbeddingGemma-2/mmproj-Q8_0.gguf` | [ggml-org EmbeddingGemma 2 projector](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF/resolve/main/mmproj-embeddinggemma-2-Q8_0.gguf) | 555 MB |

For the last file, save it using the app's expected filename `mmproj-Q8_0.gguf` (the upstream download name begins with `mmproj-embeddinggemma-2-`). The app's model-folder names and these filenames are part of the current llama.cpp router setup; changing them may prevent model discovery.

Copy the `app-models.ini` asset from the GitHub Release into the new `models` folder. Do not edit its `EmbeddingGemma-2` model id unless the server preset is updated and tested too.

The four model files require roughly 2.5 GB of disk space, in addition to the app and llama.cpp runtime. Each device has different RAM available after Windows and organizational software; these example quantizations are not a guarantee of acceptable performance on every PC. The application currently keeps both configured models available and unloads them after inactivity. Image indexing can fail with conservative server batch settings; do not raise `-ub` on low-memory devices without benchmarking.

## 4. First launch

Launch **Multimodal Local Chatbot** from the Start menu or its installed executable. The app starts llama.cpp on `127.0.0.1:8080`, loads the configured chat and embedding models, and stops the server when the app exits. No terminal or batch file is needed.

If startup reports that llama.cpp or a model is missing, verify the exact directory structure above and confirm that `llama-server.exe`, its DLLs, the model files, and `app-models.ini` are present. If the server fails to load the embedding model, collect the app's displayed error and the machine's available RAM for IT; do not guess new `-b`/`-ub` values.

## Model and runtime terms

- Review the [LFM Open License](https://huggingface.co/LiquidAI/LFM3-Omni-2B-GGUF/blob/main/LICENSE) and the model card before organizational distribution. This example uses Liquid AI's Q4_K_M chat weights and Q8_0 projector.
- The EmbeddingGemma 2 GGUF page declares Apache-2.0 for its GGUF export. Review the [upstream EmbeddingGemma information](https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2?hl=en) and [Gemma terms](https://ai.google.dev/gemma/terms) with IT before redistributing weights.
- llama.cpp is distributed under the [upstream MIT license](https://github.com/ggml-org/llama.cpp/blob/b12359/LICENSE). Retain upstream notices when redistributing its runtime files.
- Model and runtime downloads are third-party files. Verify their provenance and integrity against the upstream publisher before an organization mirrors them.

## Known release limits

- This is the Windows 11 x64 CPU release. macOS is not packaged or validated.
- The installer is unsigned; SmartScreen or organizational application-control policy may block it until IT signs or approves it.
- The installer does not include models or llama.cpp. A network connection may be needed by the NSIS installer to install WebView2 if it is not already available.
- Full `langchain-rust` agent/checkpointer integration, broad hardware benchmarking, and production installer validation are still outstanding.
- Audio collection indexing is segmented locally. It is not transcription, and real speech retrieval quality has not been established.
- Image embedding requires more physical batch than the conservative profile used by this app may permit. Image indexing is not guaranteed on the target low-memory devices.
