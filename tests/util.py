from __future__ import annotations

import json
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

import anthropic
import anyio
import httpx2
from mcp import Client

from warehouse_evals.agent import Trajectory, WarehouseAgent
from warehouse_mcp.policy import Principal
from warehouse_mcp.server import GovernedWarehouse

T = TypeVar("T")


def run(fn: Callable[..., Coroutine[Any, Any, T]], *args: Any) -> T:
    """Run an async function from a sync test."""
    return anyio.run(fn, *args)


def arun(coro: Coroutine[Any, Any, T]) -> T:
    """Await one coroutine from a sync test."""

    async def main() -> T:
        return await coro

    return anyio.run(main)


def message_reply(
    text: str = "",
    stop: str = "end_turn",
    usage: dict[str, int] | None = None,
    model: str = "claude-opus-5-5",
    tool_use: tuple[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """A non-streaming Messages API reply: one text block, then an optional (name, input) tool call."""
    content: list[dict[str, Any]] = [{"type": "text", "text": text}] if text else []
    if tool_use is not None:
        content.append({"type": "tool_use", "id": "toolu_offline", "name": tool_use[0], "input": tool_use[1]})
    return {
        "id": "msg_offline",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": usage or {"input_tokens": 10, "output_tokens": 10},
    }


class ReplyTransport(httpx2.AsyncBaseTransport):
    """Answers the n-th request with the n-th reply (the last one repeats) and records the bodies."""

    def __init__(self, *replies: dict[str, Any]) -> None:
        self.replies = replies
        self.bodies: list[dict[str, Any]] = []

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        self.bodies.append(json.loads(await request.aread()))
        return httpx2.Response(200, json=self.replies[min(len(self.bodies), len(self.replies)) - 1])


def run_agent(
    gw: GovernedWarehouse,
    principal: Principal,
    client: anthropic.AsyncAnthropic,
    question: str = "question?",
    **kwargs: Any,
) -> Trajectory:
    """Run the agent once against a fresh MCP session on ``gw`` acting as ``principal``."""

    async def go() -> Trajectory:
        async with Client(gw.build_mcp(lambda: principal)) as mcp:
            return await WarehouseAgent(client, mcp, **kwargs).run(question)

    return run(go)
