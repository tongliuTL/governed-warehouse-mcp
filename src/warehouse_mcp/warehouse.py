"""Read-only execution against the DuckDB warehouse.

The connection is the second layer of defence behind :mod:`warehouse_mcp.sql_guard`:

- the database file is opened ``read_only``, so no statement can write to it;
- external access is disabled, so file, HTTP and extension functions cannot reach outside;
- configuration is locked after start-up, so a query cannot turn those settings back on;
- every query runs under a wall-clock timeout and a memory limit.
"""

from __future__ import annotations

import datetime as dt
import decimal
import math
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from warehouse_mcp.policy import PolicyViolation
from warehouse_mcp.sql_guard import GuardedQuery

JsonScalar = str | int | float | bool | None


@dataclass(frozen=True)
class QueryOutput:
    query_id: str
    columns: list[str]
    rows: list[list[JsonScalar]]
    truncated: bool
    elapsed_ms: int


def _to_json(value: Any) -> JsonScalar:
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else round(value, 6)
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, dt.datetime | dt.date | dt.time):
        return value.isoformat()
    if isinstance(value, dt.timedelta):
        return value.total_seconds()
    return str(value)


class Warehouse:
    def __init__(self, path: Path | str, *, timeout_s: float = 10.0, memory_limit: str = "1GB") -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"warehouse not found at {self.path}; run `gwm generate` first")
        self.timeout_s = timeout_s
        self._con = duckdb.connect(
            str(self.path),
            read_only=True,
            config={
                "enable_external_access": False,
                "autoload_known_extensions": False,
                "autoinstall_known_extensions": False,
                "memory_limit": memory_limit,
                "threads": 4,
                "lock_configuration": True,
            },
        )

    def close(self) -> None:
        self._con.close()

    def execute(self, query: GuardedQuery) -> QueryOutput:
        query_id = uuid.uuid4().hex[:16]
        cursor = self._con.cursor()
        timer = threading.Timer(self.timeout_s, cursor.interrupt)
        start = time.perf_counter()
        timer.start()
        try:
            result = cursor.execute(query.sql)
            columns = [d[0] for d in result.description or []]
            raw = result.fetchmany(query.fetch_limit)
        except duckdb.InterruptException:
            raise PolicyViolation(
                "TIMEOUT", f"query exceeded {self.timeout_s:g}s; aggregate or filter more"
            ) from None
        except duckdb.Error as err:
            message = str(err).splitlines()[0]
            raise PolicyViolation("EXECUTION_ERROR", message) from None
        finally:
            timer.cancel()
            cursor.close()
        elapsed_ms = round((time.perf_counter() - start) * 1000)
        truncated = len(raw) > query.max_rows
        rows = [[_to_json(v) for v in row] for row in raw[: query.max_rows]]
        return QueryOutput(
            query_id=query_id, columns=columns, rows=rows, truncated=truncated, elapsed_ms=elapsed_ms
        )

    def execute_trusted(self, sql: str) -> list[tuple[Any, ...]]:
        """Run SQL without the guard. Only for the eval harness's reference answers."""
        cursor = self._con.cursor()
        try:
            return cursor.execute(sql).fetchall()
        finally:
            cursor.close()
