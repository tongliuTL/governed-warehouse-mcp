from __future__ import annotations

import re
from pathlib import Path

import duckdb

from warehouse_mcp.synthetic import INJECTION_NOTES, TABLES, VIEWS, build_warehouse, table_fingerprints

# Pinned content hashes. A change here means the synthetic data changed: every gold answer is
# recomputed from the data, so evals stay correct, but the change should be deliberate.
PINNED = {
    "dim_carriers": "2563b8fc00606110",
    "dim_customers": "63ce39c969ce23ad",
    "dim_lanes": "e5de51c632e4377c",
    "fct_documents": "afca89c0e243e62b",
    "fct_invoices": "24460d23bd7f5697",
    "fct_quotes": "e1beef13a35bc2d7",
    "fct_shipment_notes": "295586a39113f3ba",
    "fct_shipments": "08ca82244d187c21",
    "fct_tracking_events": "d93bc51cd22748bd",
}


def test_generation_is_deterministic_and_pinned(raw_db: Path, tmp_path: Path) -> None:
    again = build_warehouse(tmp_path / "again.duckdb")
    assert table_fingerprints(raw_db) == table_fingerprints(again) == PINNED


def test_different_seed_changes_data(raw_db: Path, tmp_path: Path) -> None:
    other = build_warehouse(tmp_path / "other.duckdb", seed=7)
    assert table_fingerprints(other) != table_fingerprints(raw_db)


def test_all_tables_and_views_exist_with_rows(raw_db: Path) -> None:
    with duckdb.connect(str(raw_db), read_only=True) as con:
        for name in (*TABLES, *VIEWS):
            (count,) = con.execute(f"SELECT count(*) FROM {name}").fetchone()  # type: ignore[misc]
            assert count > 0, name


def test_pii_is_obviously_synthetic(raw_db: Path) -> None:
    with duckdb.connect(str(raw_db), read_only=True) as con:
        rows = con.execute("SELECT contact_email, contact_phone FROM dim_customers").fetchall()
    assert all(email.endswith(".example") for email, _ in rows)  # RFC 2606 reserved TLD
    assert all(re.fullmatch(r"\+1-555-01\d\d", phone) for _, phone in rows)  # fictional range


def test_referential_integrity(raw_db: Path) -> None:
    with duckdb.connect(str(raw_db), read_only=True) as con:
        orphans = con.execute("""
            SELECT
              (SELECT count(*) FROM fct_shipments s ANTI JOIN dim_customers c USING (customer_id)),
              (SELECT count(*) FROM fct_shipments s ANTI JOIN dim_carriers c USING (carrier_id)),
              (SELECT count(*) FROM fct_shipments s ANTI JOIN dim_lanes l USING (lane_id)),
              (SELECT count(*) FROM fct_invoices i ANTI JOIN fct_shipments s USING (shipment_id)),
              (SELECT count(*) FROM fct_quotes q ANTI JOIN dim_lanes l USING (lane_id))
        """).fetchone()
        assert orphans == (0, 0, 0, 0, 0)
        bad_dates = con.execute("""
            SELECT count(*) FROM fct_shipments
            WHERE delivered_date < pickup_date OR booked_date > pickup_date
               OR (status = 'delivered') <> (delivered_date IS NOT NULL)
        """).fetchone()
        assert bad_dates == (0,)


def test_injection_notes_are_planted_on_known_shipments(raw_db: Path) -> None:
    with duckdb.connect(str(raw_db), read_only=True) as con:
        planted = dict(
            con.execute(
                "SELECT note_text, shipment_id FROM fct_shipment_notes WHERE note_text IN (?, ?, ?)",
                list(INJECTION_NOTES),
            ).fetchall()
        )
    assert planted == {
        INJECTION_NOTES[0]: "S001000",
        INJECTION_NOTES[1]: "S002000",
        INJECTION_NOTES[2]: "S003000",
    }
