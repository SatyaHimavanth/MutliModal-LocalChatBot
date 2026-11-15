#!/usr/bin/env python3
"""End-to-end smoke test for text, image, and audio EmbeddingGemma inputs."""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave
import zlib
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVER = ROOT / "vendor/llama.cpp/windows-x86_64/llama-server.exe"
DEFAULT_MODEL = ROOT / "models/EmbeddingGemma-2/embeddinggemma-2-Q8_0.gguf"
DEFAULT_MMPROJ = ROOT / "models/EmbeddingGemma-2/mmproj-Q8_0.gguf"
DEFAULT_RESULTS = ROOT / "qa/artifacts/embedding-endpoint-results.json"


def png_bytes() -> bytes:
    """Create a tiny deterministic RGB PNG without third-party packages."""
    width = height = 64
    rows = bytearray()
    for y in range(height):
        rows.append(0)  # PNG filter: None
        for x in range(width):
            rows.extend((220 if x < width // 2 else 30, 40 if y < height // 2 else 180, 70))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(rows)))
        + chunk(b"IEND", b"")
    )


def wav_bytes() -> bytes:
    """Create a short, deterministic mono WAV tone for modality-path testing."""
    sample_rate = 16_000
    duration = 1.0
    samples = bytearray()
    for index in range(int(sample_rate * duration)):
        sample = int(7000 * math.sin(2 * math.pi * 440 * index / sample_rate))
        samples.extend(struct.pack("<h", sample))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(bytes(samples))
    return buffer.getvalue()


def request_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 30.0) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def embedding_payload(model: str, content: list[dict[str, Any]]) -> dict[str, Any]:
    return {"model": model, "input": [{"content": content}], "encoding_format": "float"}


def get_vector(response: dict[str, Any]) -> list[float]:
    vector = response["data"][0]["embedding"]
    if not isinstance(vector, list) or not vector:
        raise ValueError("response has no non-empty embedding vector")
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in vector):
        raise ValueError("embedding contains a non-numeric or non-finite value")
    return vector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--startup-timeout", type=int, default=240)
    parser.add_argument("--server", type=Path, default=DEFAULT_SERVER)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--mmproj", type=Path, default=DEFAULT_MMPROJ)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    for path in (args.server, args.model, args.mmproj):
        if not path.is_file():
            print(f"ERROR: required file not found: {path}", flush=True)
            return 2

    version_result = subprocess.run(
        [str(args.server), "--version"], capture_output=True, text=True, check=True
    )
    version = (version_result.stdout + version_result.stderr).strip()
    if "build 12359" not in version:
        print(f"ERROR: expected llama.cpp build 12359, got: {version}", flush=True)
        return 2

    artifacts = args.results.parent
    artifacts.mkdir(parents=True, exist_ok=True)
    log_path = artifacts / "embedding-endpoint-server.log"
    base_url = f"http://127.0.0.1:{args.port}"
    command = [
        str(args.server),
        "--model", str(args.model),
        "--mmproj", str(args.mmproj),
        "--embedding",
        "--pooling", "mean",
        "--ctx-size", "2048",
        # EmbeddingGemma 2's projector needs a large image-token budget;
        # b12359 otherwise caps it below the model's minimum pixel count.
        "--batch-size", "2048",
        "--ubatch-size", "2048",
        "--threads", "4",
        "--threads-batch", "4",
        "--parallel", "1",
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--no-webui",
    ]

    results: dict[str, Any] = {
        "server_version": version,
        "command": command,
        "endpoint": f"{base_url}/v1/embeddings",
        "tests": [],
    }
    server: subprocess.Popen[str] | None = None
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            server = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )

        deadline = time.monotonic() + args.startup_timeout
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f"server exited during startup with code {server.returncode}; see {log_path}")
            try:
                request_json(f"{base_url}/health", timeout=2)
                break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                time.sleep(1)
        else:
            raise TimeoutError(f"server did not become healthy within {args.startup_timeout}s; see {log_path}")

        models = request_json(f"{base_url}/v1/models")
        model = models["data"][0]["id"]
        results["model_id"] = model
        results["model_info"] = models["data"][0].get("architecture")
        print(f"READY: {version}; API model id={model}", flush=True)

        fixtures = {
            "text": [{"type": "text", "text": "A red and blue geometric test image."}],
            "image": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64," + base64.b64encode(png_bytes()).decode("ascii")
                    },
                }
            ],
            "audio": [
                {
                    "type": "input_audio",
                    "input_audio": {"data": base64.b64encode(wav_bytes()).decode("ascii"), "format": "wav"},
                }
            ],
        }

        for modality, content in fixtures.items():
            started = time.monotonic()
            try:
                response = request_json(
                    f"{base_url}/v1/embeddings",
                    embedding_payload(model, content),
                    timeout=180,
                )
                vector = get_vector(response)
                item = {
                    "modality": modality,
                    "status": "PASS",
                    "dimensions": len(vector),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "vector_norm": round(math.sqrt(sum(value * value for value in vector)), 6),
                }
                results["tests"].append(item)
                print(
                    f"PASS: {modality} embedding returned {len(vector)} finite values "
                    f"in {item['elapsed_seconds']}s",
                    flush=True,
                )
            except Exception as error:  # Record each modality independently.
                item = {
                    "modality": modality,
                    "status": "FAIL",
                    "error": f"{type(error).__name__}: {error}",
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                }
                results["tests"].append(item)
                print(f"FAIL: {modality}: {item['error']}", flush=True)

        results["overall"] = "PASS" if all(test["status"] == "PASS" for test in results["tests"]) else "FAIL"
        return 0 if results["overall"] == "PASS" else 1
    except Exception as error:
        results["overall"] = "ERROR"
        results["error"] = f"{type(error).__name__}: {error}"
        print(f"ERROR: {results['error']}", flush=True)
        return 2
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)
        args.results.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"RESULT: {args.results}", flush=True)
        if log_path.exists():
            print(f"SERVER LOG: {log_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
