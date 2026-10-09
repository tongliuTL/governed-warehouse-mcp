"""Structured audit trail: one JSON record per tool call, allowed or denied.

Records carry the principal, the tool, the decision and its stable reason code, the models
touched, a hash of the executed SQL (and the SQL itself), row counts and latency. The
``internal_detail`` of a denial is kept here even when the agent-facing message is generic.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

Decision = Literal["allow", "deny", "error"]


@dataclass
class AuditRecord:
    tool: str
    principal: str
    decision: Decision
    catalog_version: str
    client_id: str | None = None
    reason_code: str | None = None
    detail: str | None = None
    query_id: str | None = None
    models: list[str] = field(default_factory=list)
    sql: str | None = None
    sql_sha256: str | None = None
    row_count: int | None = None
    truncated: bool | None = None
    elapsed_ms: int | None = None
    arguments: dict[str, object] = field(default_factory=dict)
    ts: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="milliseconds"))


class AuditLog:
    """Thread-safe sink. Keeps records in memory and optionally appends JSONL to a file."""

    def __init__(self, path: Path | str | None = None, *, keep: int = 10_000) -> None:
        self.path = Path(path) if path else None
        self._keep = keep
        self._records: list[AuditRecord] = []
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: AuditRecord) -> None:
        line = json.dumps(asdict(record), sort_keys=True, default=str)
        with self._lock:
            self._records.append(record)
            del self._records[: -self._keep]
            if self.path:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")

    def records(self) -> list[AuditRecord]:
        with self._lock:
            return list(self._records)

    def __iter__(self) -> Iterator[AuditRecord]:
        return iter(self.records())

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
