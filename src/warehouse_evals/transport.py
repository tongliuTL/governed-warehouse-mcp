"""HTTP transports that let the real Anthropic SDK run offline.

- ``ScriptedTransport`` answers each Messages API request with a turn produced by a Python
  policy. Used for the harness self-test and for grader validation.
- ``RecordingTransport`` wraps the real network transport and saves every streamed response
  to a cassette (JSONL). Request bodies are fingerprinted, not stored, and no headers are kept,
  so API keys never reach disk.
- ``ReplayTransport`` serves a cassette back in order and fails loudly when the agent's
  requests drift from the recording (different model, system prompt, tool schemas or turn
  count), which is the signal to re-record.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx2

from warehouse_evals.sse import Turn, encode_turn

CASSETTE_FORMAT = 1
Policy = Callable[[dict[str, Any]], Turn]


def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def fingerprint(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": body.get("model"),
        "system": _sha(body.get("system")),
        "tools": _sha(body.get("tools")),
        "messages": len(body.get("messages", [])),
    }


class CassetteMismatch(AssertionError):
    pass


def _sse_response(content: bytes, status: int = 200) -> httpx2.Response:
    return httpx2.Response(status, headers={"content-type": "text/event-stream"}, content=content)


class ScriptedTransport(httpx2.AsyncBaseTransport):
    def __init__(self, policy: Policy) -> None:
        self.policy = policy
        self.requests: list[dict[str, Any]] = []

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(await request.aread())
        self.requests.append(body)
        return _sse_response(encode_turn(self.policy(body), model=str(body.get("model"))))


@dataclass
class Interaction:
    fingerprint: dict[str, Any]
    status: int
    body: str


class RecordingTransport(httpx2.AsyncBaseTransport):
    def __init__(self, inner: httpx2.AsyncBaseTransport, path: Path | str) -> None:
        self.inner = inner
        self.path = Path(path)
        self.interactions: list[Interaction] = []

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(await request.aread())
        response = await self.inner.handle_async_request(request)
        content = await response.aread()
        self.interactions.append(
            Interaction(fingerprint(body), response.status_code, content.decode("utf-8"))
        )
        return httpx2.Response(
            response.status_code,
            headers={"content-type": response.headers.get("content-type", "text/event-stream")},
            content=content,
        )

    def save(self, meta: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({"format": CASSETTE_FORMAT, **meta}, sort_keys=True) + "\n")
            for item in self.interactions:
                fh.write(
                    json.dumps(
                        {"fingerprint": item.fingerprint, "status": item.status, "body": item.body},
                        sort_keys=True,
                    )
                    + "\n"
                )

    async def aclose(self) -> None:
        await self.inner.aclose()


class ReplayTransport(httpx2.AsyncBaseTransport):
    def __init__(self, path: Path | str) -> None:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        self.meta = json.loads(lines[0])
        if self.meta.get("format") != CASSETTE_FORMAT:
            raise CassetteMismatch(f"{path}: unsupported cassette format {self.meta.get('format')}")
        self.path = Path(path)
        self.interactions = [json.loads(line) for line in lines[1:]]
        self.position = 0
        # The SDK wraps transport exceptions in APIConnectionError, so the mismatch is also kept
        # here for the runner to report as a failed replay rather than a network blip.
        self.failure: str | None = None

    def _fail(self, message: str) -> CassetteMismatch:
        self.failure = message
        return CassetteMismatch(message)

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(await request.aread())
        if self.position >= len(self.interactions):
            raise self._fail(f"{self.path.name}: agent made more requests than were recorded")
        item = self.interactions[self.position]
        self.position += 1
        got = fingerprint(body)
        if got != item["fingerprint"]:
            diff = {
                k: (item["fingerprint"].get(k), v) for k, v in got.items() if item["fingerprint"].get(k) != v
            }
            raise self._fail(f"{self.path.name} request {self.position}: drift {diff}; re-record this case")
        return httpx2.Response(
            item["status"],
            headers={"content-type": "text/event-stream"},
            content=item["body"].encode("utf-8"),
        )
