"""The engine is the second layer: it must hold even if the SQL guard were bypassed."""

from __future__ import annotations

import shutil
import time
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


def test_timeout_fires_on_time(db_path: Path) -> None:
    wh = Warehouse(db_path, timeout_s=0.5)
    start = time.perf_counter()
    try:
        with pytest.raises(PolicyViolation, match="TIMEOUT") as info:
            wh.execute(
                raw(
                    "select count(*) from fct_tracking_events a, fct_tracking_events b, fct_tracking_events c"
                )
            )
    finally:
        wh.close()
    elapsed = time.perf_counter() - start
    assert 0.5 <= elapsed < 3.0, elapsed  # interrupted near the limit, not when the query ends
    assert "0.5s" in info.value.message


def test_memory_limit_is_hard(db_path: Path, tmp_path: Path) -> None:
    """No spill to disk: a query that outgrows the limit fails, and the connection survives."""
    # A copy: DuckDB refuses a second in-process connection to one file with other settings.
    copy = tmp_path / "small.duckdb"
    shutil.copy(db_path, copy)
    wh = Warehouse(copy, memory_limit="64MB")
    try:
        assert wh._con.sql("select current_setting('max_temp_directory_size')").fetchone() == ("0 bytes",)
        with pytest.raises(PolicyViolation, match="EXECUTION_ERROR") as info:
            wh.execute(
                raw(
                    "select string_agg(md5(i::varchar), ',') from "
                    "(select i from range(3000000) t(i) order by md5(i::varchar))"
                )
            )
        assert "Out of Memory" in info.value.message
        assert wh.execute(raw("select 1")).rows == [[1]]
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
