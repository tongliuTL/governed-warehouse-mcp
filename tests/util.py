from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

import anyio

T = TypeVar("T")


def run(fn: Callable[..., Coroutine[Any, Any, T]], *args: Any) -> T:
    """Run an async function from a sync test."""
    return anyio.run(fn, *args)
