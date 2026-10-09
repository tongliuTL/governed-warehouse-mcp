from __future__ import annotations

import json
from pathlib import Path

import pytest

from warehouse_mcp.audit import AuditLog, AuditRecord
from warehouse_mcp.ratelimit import RateLimiter


def test_audit_jsonl_and_bounded_memory(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "nested" / "audit.jsonl", keep=2)
    for i in range(3):
        log.write(AuditRecord(tool=f"t{i}", principal="p", decision="allow", catalog_version="v"))
    assert [r.tool for r in log] == ["t1", "t2"]
    lines = [json.loads(line) for line in (tmp_path / "nested" / "audit.jsonl").read_text().splitlines()]
    assert [line["tool"] for line in lines] == ["t0", "t1", "t2"] and lines[0]["ts"].endswith("+00:00")
    log.clear()
    assert log.records() == []


def test_token_bucket() -> None:
    now = [0.0]
    limiter = RateLimiter(capacity=2, refill_per_s=1.0, clock=lambda: now[0])
    assert [limiter.allow("a") for _ in range(3)] == [True, True, False]
    assert limiter.allow("b")  # buckets are per principal
    now[0] = 1.0
    assert limiter.allow("a") and not limiter.allow("a")


def test_rate_limiter_validation() -> None:
    with pytest.raises(ValueError, match="capacity"):
        RateLimiter(capacity=0)
