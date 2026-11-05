"""Langfuse tracing for live runs: the agent loop, its tool calls, and the LLM judge.

Tracing is off unless a command that calls the real Claude API turns it on (``gwm-eval live``,
``gwm-eval calibrate --live``) and Langfuse credentials are set. Scripted and replayed runs are
never traced: their model turns are fixtures, and tracing them would record invented usage
and cost as if a model had produced them.

Trace shape for one agent case (one trace per case and trial):

    answer-question                 agent       input: the question, output: the final answer
      call-model                    generation  one per Messages call, with thinking and usage
      <tool name>                   tool        one per tool call, sibling of the generation
      call-model                    generation
      ...
    scores                                      one boolean score per grader, plus ``passed``

Trace for one judge verdict (only on a cache miss, the only time the API is called):

    grade-trajectory                evaluator   input: rendered trajectory, output: verdicts
      judge-trajectory              generation

Every string attribute is scanned for the warehouse's PII values before export, whatever role
the run had: a role that may read PII is not the same audience as whoever reads the traces.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from typing import Any

from langfuse import Langfuse, is_langfuse_span
from langfuse.types import MaskOtelSpansParams, MaskOtelSpansResult, OtelSpanPatch

REDACTED = "[REDACTED_PII]"
# Only this module's observations. The MCP SDK also emits a ``tools/call`` span (it passes the
# SDK's default filter on its gen_ai attributes) that duplicates each tool observation with no
# input or output.
SHOULD_EXPORT_SPAN = is_langfuse_span
DEFAULT_ENVIRONMENT = "eval"


class _State:
    live: Langfuse | None = None  # set by enable()
    noop: Langfuse | None = None
    pii: re.Pattern[str] | None = None


_state = _State()
# Placeholder keys for the no-op client: with tracing disabled nothing is sent, and the SDK
# does not log a missing-key error on every run.
_NOOP_PUBLIC_KEY, _NOOP_SECRET_KEY = "pk-lf-disabled", "sk-lf-disabled"


def credentials_present() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


def _mask_spans(*, params: MaskOtelSpansParams) -> MaskOtelSpansResult | None:
    pattern = _state.pii
    if pattern is None:
        return None
    patches = {}
    for ident, span in params.spans.items():
        changed = {
            key: pattern.sub(REDACTED, value)
            for key, value in span.attributes.items()
            if isinstance(value, str) and pattern.search(value)
        }
        if changed:
            patches[ident] = OtelSpanPatch(set_attributes=changed)
    return MaskOtelSpansResult(span_patches=patches)


def set_pii_values(values: Iterable[str]) -> None:
    """The exact strings to redact. Longest first, so a full name wins over a part of it."""
    unique = sorted({v for v in values if v}, key=len, reverse=True)
    _state.pii = re.compile("|".join(re.escape(v) for v in unique)) if unique else None


def enable() -> bool:
    """Turn tracing on for this process when credentials are set. Returns whether it is on
    (``LANGFUSE_TRACING_ENABLED=false`` still turns it off)."""
    if not credentials_present():
        return False
    if _state.live is None:
        _state.live = Langfuse(
            environment=os.environ.get("LANGFUSE_TRACING_ENVIRONMENT", DEFAULT_ENVIRONMENT),
            mask_otel_spans=_mask_spans,
            should_export_span=SHOULD_EXPORT_SPAN,
        )
    return enabled()


def enabled() -> bool:
    return _state.live is not None and bool(_state.live._tracing_enabled)


def client(traced: bool = True) -> Langfuse:
    """The live client when ``traced`` and tracing is on, else one whose observations are no-ops.

    The no-op client lets instrumented code run unchanged in scripted, replayed and test runs.
    """
    if traced and _state.live is not None:
        return _state.live
    if _state.noop is None:
        _state.noop = Langfuse(
            public_key=_NOOP_PUBLIC_KEY, secret_key=_NOOP_SECRET_KEY, tracing_enabled=False
        )
    return _state.noop


def flush() -> None:
    if _state.live is not None:
        _state.live.flush()


def usage_details(usage: Any) -> dict[str, int]:
    """Anthropic usage as Langfuse buckets. Anthropic's ``input_tokens`` already excludes cache
    reads and writes, so the buckets do not overlap and cost is not double counted."""
    details = {"input": usage.input_tokens, "output": usage.output_tokens}
    for key in ("cache_read_input_tokens", "cache_creation_input_tokens"):
        value = getattr(usage, key, None)
        if isinstance(value, int) and value:
            details[key] = value
    return details


def content_dump(content: Any) -> Any:
    """Message content as plain JSON, for generation input and output (thinking included)."""
    if isinstance(content, str):
        return content
    return [b.model_dump(mode="json", exclude_none=True) if hasattr(b, "model_dump") else b for b in content]


def chat_input(system: str, messages: list[Any]) -> list[dict[str, Any]]:
    """What the model saw on this call, in order: the system prompt and the full history."""
    return [{"role": "system", "content": system}] + [
        {"role": m["role"], "content": content_dump(m["content"])} for m in messages
    ]
