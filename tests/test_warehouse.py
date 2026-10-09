"""The engine is the second layer: it must hold even if the SQL guard were bypassed."""

from __future__ import annotations

from pathlib import Path

import pytest

from warehouse_mcp.policy import PolicyViolation
from warehouse_mcp.sql_guard import GuardedQuery
from warehouse_mcp.warehouse import Warehouse, _to_json


def raw(sql: str, limit: int = 10) -> GuardedQuery:
    """A query that skipped the guard, to test the engine on its own."""
    return GuardedQuery(sql=sql, models=frozenset(), fetch_limit=limit + 1, max_rows=limit)


@pytest.mark.parametrize(
    ("sql", "fragment"),
    [
        ("select * from read_csv('/etc/passwd')", "disabled by configuration"),
        ("set enable_external_access = true", "locked"),
        ("set lock_configuration = false", "locked"),
        ("create table x (a int)", "read-only"),
        ("delete from dim_lanes", "read-only"),
        ("install httpfs", ""),
    ],
)
def test_engine_blocks_unguarded_sql(warehouse: Warehouse, sql: str, fragment: str) -> None:
    with pytest.raises(PolicyViolation, match="EXECUTION_ERROR") as info:
        warehouse.execute(raw(sql))
    assert fragment in info.value.message


def test_timeout_interrupts_long_queries(db_path: Path) -> None:
    wh = Warehouse(db_path, timeout_s=0.5)
    try:
        with pytest.raises(PolicyViolation, match="TIMEOUT"):
            wh.execute(
                raw(
                    "select count(*) from fct_tracking_events a, fct_tracking_events b, fct_tracking_events c"
                )
            )
        assert wh.execute(raw("select 1")).rows == [[1]]  # the connection survives an interrupt
    finally:
        wh.close()


def test_truncation_flag(warehouse: Warehouse) -> None:
    out = warehouse.execute(raw("select lane_id from dim_lanes order by 1", limit=5))
    assert out.truncated and len(out.rows) == 5 and out.columns == ["lane_id"]
    out = warehouse.execute(GuardedQuery("select lane_id from dim_lanes limit 3", frozenset(), 3, 5))
    assert not out.truncated and len(out.rows) == 3


def test_values_are_json_safe(warehouse: Warehouse) -> None:
    out = warehouse.execute(
        raw(
            "select date '2026-01-02', timestamp '2026-01-02 03:04:05', 1.5::decimal(4,2), 'nan'::double, "
            "interval 1 day, true, null"
        )
    )
    assert out.rows == [["2026-01-02", "2026-01-02T03:04:05", 1.5, None, 86400.0, True, None]]
    assert _to_json(object()).startswith("<object")


def test_missing_database(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="gwm generate"):
        Warehouse(tmp_path / "nope.duckdb")
