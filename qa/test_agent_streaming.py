#!/usr/bin/env python3
"""Small feasibility probe for streamed agent-like chat and tool-call support.

This script starts the local llama.cpp server with the LFM3-Omni chat model,
streams a normal chat response token-by-token, then sends a tool schema request
as a simple agent-like probe. The goal is to confirm that streaming is possible
and that the model/router can accept tool-invocation metadata at the API boundary.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = ROOT / "vendor/llama.cpp/windows-x86_64/llama-server.exe"
MODEL_PATH = ROOT / "models/LFM3-Omni/LFM3-Omni-2B-Q4_K_M.gguf"
LOG_PATH = ROOT / "qa/artifacts/agent-streaming.log"


def start_server(port: int) -> subprocess.Popen[str]:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            [
                str(SERVER_PATH),
                "--model",
                str(MODEL_PATH),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--no-webui",
                "--threads",
                "4",
                "--threads-batch",
                "4",
                "--parallel",
                "1",
                "--ctx-size",
                "4096",
            ],
            cwd=str(ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    return proc


def request_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 30.0) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def stream_chat(prompt: str, port: int, include_tools: bool = False) -> tuple[str, bool, str]:
    model_id = request_json(f"http://127.0.0.1:{port}/v1/models")["data"][0]["id"]
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if include_tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "lookup_weather",
                    "description": "Look up weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
        payload["tool_choice"] = "auto"

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    collected: list[str] = []
    saw_tool_call = False
    saw_text = False
    with urllib.request.urlopen(req, timeout=180) as response:
        for raw_line in response.read().decode("utf-8").splitlines():
            if not raw_line.startswith("data:"):
                continue
            data = raw_line[5:].strip()
            if data == "[DONE]":
                break
            if not data:
                continue
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            if "choices" not in event or not event["choices"]:
                continue
            delta = event["choices"][0].get("delta", {})
            if "content" in delta and delta["content"]:
                text = delta["content"]
                collected.append(text)
                saw_text = True
                print(text, end="", flush=True)
            if "tool_calls" in delta and delta["tool_calls"]:
                saw_tool_call = True
                print("\n[TOOL_CALL_DETECTED]", flush=True)
                print(json.dumps(delta["tool_calls"], indent=2), flush=True)
    return "".join(collected), saw_tool_call, "text" if saw_text else "none"


def wait_for_server(url: str, timeout_seconds: int = 180) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=5):
                return
        except Exception as exc:  # pragma: no cover - test probe only
            last_error = exc
        time.sleep(1)
    raise RuntimeError(f"Server did not become healthy at {url}; last error: {last_error}")


def main() -> int:
    port = 18081
    server = None
    result = {"port": port, "streaming": {}, "tool_probe": {} }
    try:
        server = start_server(port)
        wait_for_server(f"http://127.0.0.1:{port}", timeout_seconds=180)

        print("\n=== normal streaming probe ===", flush=True)
        reply, tool_detected, saw_kind = stream_chat(
            "Reply in one short sentence: what is the difference between a chain and an agent?",
            port,
            include_tools=False,
        )
        result["streaming"] = {"reply_prefix": reply[:120], "saw_tool_call": tool_detected, "saw_kind": saw_kind}

        print("\n\n=== tool-call probe ===", flush=True)
        reply2, tool_detected2, saw_kind2 = stream_chat(
            "Use the weather lookup tool for Seattle.",
            port,
            include_tools=True,
        )
        result["tool_probe"] = {"reply_prefix": reply2[:120], "saw_tool_call": tool_detected2, "saw_kind": saw_kind2}

        print("\n\n=== summary ===", flush=True)
        print(json.dumps(result, indent=2), flush=True)
        return 0
    except Exception as exc:  # pragma: no cover - test probe only
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
