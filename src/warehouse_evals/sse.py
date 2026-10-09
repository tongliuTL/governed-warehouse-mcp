"""Encode assistant turns as Messages API server-sent events.

Scripted turns and recorded cassettes are both served to the real Anthropic SDK as SSE bytes,
so offline runs exercise the same streaming parser, tool-input accumulation and final-message
assembly as live runs.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class TextBlock:
    text: str


@dataclass(frozen=True)
class ToolUseBlock:
    name: str
    input: dict[str, object] | None = None
    raw_json: str | None = None  # send exactly this text instead of json.dumps(input), e.g. malformed JSON
    id: str = field(default_factory=lambda: f"toolu_{uuid.uuid4().hex[:24]}")


Block = TextBlock | ToolUseBlock
StopReason = Literal["end_turn", "tool_use", "max_tokens", "refusal", "pause_turn"]


@dataclass(frozen=True)
class Turn:
    blocks: tuple[Block, ...]
    stop_reason: StopReason | None = None
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def resolved_stop_reason(self) -> StopReason:
        if self.stop_reason:
            return self.stop_reason
        return "tool_use" if any(isinstance(b, ToolUseBlock) for b in self.blocks) else "end_turn"


def _event(kind: str, payload: dict[str, object]) -> bytes:
    return f"event: {kind}\ndata: {json.dumps({'type': kind, **payload})}\n\n".encode()


def _chunks(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


def encode_turn(turn: Turn, model: str, chunk_size: int = 24) -> bytes:
    out = [
        _event(
            "message_start",
            {
                "message": {
                    "id": f"msg_{uuid.uuid4().hex[:24]}",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": turn.input_tokens, "output_tokens": 1},
                }
            },
        )
    ]
    for index, block in enumerate(turn.blocks):
        if isinstance(block, TextBlock):
            out.append(
                _event("content_block_start", {"index": index, "content_block": {"type": "text", "text": ""}})
            )
            for piece in _chunks(block.text, chunk_size):
                out.append(
                    _event(
                        "content_block_delta",
                        {"index": index, "delta": {"type": "text_delta", "text": piece}},
                    )
                )
        else:
            out.append(
                _event(
                    "content_block_start",
                    {
                        "index": index,
                        "content_block": {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": {},
                        },
                    },
                )
            )
            raw = block.raw_json if block.raw_json is not None else json.dumps(block.input or {})
            for piece in _chunks(raw, chunk_size):
                out.append(
                    _event(
                        "content_block_delta",
                        {"index": index, "delta": {"type": "input_json_delta", "partial_json": piece}},
                    )
                )
        out.append(_event("content_block_stop", {"index": index}))
    out.append(
        _event(
            "message_delta",
            {
                "delta": {"stop_reason": turn.resolved_stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": turn.output_tokens},
            },
        )
    )
    out.append(_event("message_stop", {}))
    return b"".join(out)
