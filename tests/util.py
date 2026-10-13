from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

import anthropic
import anyio
from mcp import Client

from warehouse_evals.agent import Trajectory, WarehouseAgent
from warehouse_mcp.policy import Principal
from warehouse_mcp.server import GovernedWarehouse

T = TypeVar("T")


def run(fn: Callable[..., Coroutine[Any, Any, T]], *args: Any) -> T:
    """Run an async function from a sync test."""
    return anyio.run(fn, *args)


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
