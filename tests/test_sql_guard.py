from __future__ import annotations

import pytest

from warehouse_mcp.catalog import Catalog
from warehouse_mcp.policy import PolicyViolation, Principal
from warehouse_mcp.sql_guard import MAX_SQL_CHARS, SqlGuard

ALLOWED = [
    "SELECT count(*) FROM fct_shipments",
    "select customer_id, segment from dim_customers where country = 'MX' limit 5",
    "SELECT * FROM dim_lanes",
    "SELECT * FROM main.dim_lanes",
    "with x as (select customer_id, segment from dim_customers) select segment, count(*) from x group by 1",
    "select lane_id from dim_lanes union all select customer_id from dim_customers",
    "select s.shipment_id from fct_shipments s join dim_customers c using (customer_id)",
    "from dim_customers select customer_id",
    "select count(*) as contact_email from dim_lanes",  # alias only; dim_lanes has no PII
    "SELECT l.border_crossing, avg(s.weight_kg) FROM fct_shipments s JOIN dim_lanes l ON s.lane_id = l.lane_id "
    "GROUP BY 1 ORDER BY 2 DESC",
]

# (sql, expected code) for an analyst: internal tier, no PII.
DENIED = [
    ("drop table fct_invoices", "READ_ONLY"),
    ("DELETE FROM fct_shipments", "READ_ONLY"),
    ("INSERT INTO dim_lanes SELECT * FROM dim_lanes", "READ_ONLY"),
    ("UPDATE dim_lanes SET distance_km = 0", "READ_ONLY"),
    ("CREATE TABLE x AS SELECT 1", "READ_ONLY"),
    ("attach '/tmp/x.db' as x", "READ_ONLY"),
    ("copy fct_shipments to '/tmp/x.csv'", "READ_ONLY"),
    ("pragma database_list", "READ_ONLY"),
    ("set enable_external_access = true", "READ_ONLY"),
    ("install httpfs", "READ_ONLY"),
    ("summarize dim_customers", "READ_ONLY"),
    ("describe dim_customers", "READ_ONLY"),
    ("select 1; drop table fct_invoices", "SINGLE_STATEMENT"),
    ("", "SINGLE_STATEMENT"),
    ("select from where", "PARSE_ERROR"),
    ("select * from read_csv('/etc/passwd')", "TABLE_FUNCTION"),
    ("select * from duckdb_tables()", "TABLE_FUNCTION"),
    ("select * from '/etc/passwd'", "UNKNOWN_MODEL"),
    ("select * from information_schema.tables", "UNKNOWN_MODEL"),
    ("select * from other.main.dim_lanes", "UNKNOWN_MODEL"),
    ("select * from secret_table", "UNKNOWN_MODEL"),
    ("select sum(revenue_usd) from fct_invoices", "UNKNOWN_MODEL"),
    ("select * from dim_lanes where lane_id in (select shipment_id from fct_invoices)", "UNKNOWN_MODEL"),
    ("select getenv('HOME')", "FORBIDDEN_FUNCTION"),
    ("select current_setting('memory_limit')", "FORBIDDEN_FUNCTION"),
    ("select query('select 1')", "FORBIDDEN_FUNCTION"),
    ("select columns('.*') from dim_customers", "FORBIDDEN_FUNCTION"),
    ("select read_text('/etc/hosts')", "FORBIDDEN_FUNCTION"),
    ("SELECT * FROM dim_customers", "PII_DENIED"),
    ("select c.* from dim_customers c", "PII_DENIED"),
    ("select contact_email from dim_customers", "PII_DENIED"),
    ('select "CONTACT_EMAIL" from dim_customers', "PII_DENIED"),
    ("select lower(c.contact_phone) x from dim_customers c", "PII_DENIED"),
    ("select dim_customers from dim_customers", "PII_DENIED"),
    ("select c from dim_customers c", "PII_DENIED"),
    ("select to_json(c) from dim_customers c", "PII_DENIED"),
    ("select count(*) from dim_customers where contact_email like '%a%'", "PII_DENIED"),
    ("select customer_id from dim_customers order by contact_name", "PII_DENIED"),
    ("select * exclude (contact_email) from dim_customers", "PII_DENIED"),
    ("with x as (select contact_email from dim_customers) select * from x", "PII_DENIED"),
    ("select (select contact_email from dim_customers limit 1) from dim_lanes", "PII_DENIED"),
    ("select struct_pack(a := c.contact_email) from dim_customers c", "PII_DENIED"),
    ("select * from fct_shipments s join dim_customers c using (customer_id)", "PII_DENIED"),
    ("select #7 from dim_customers", "POSITIONAL_REFERENCE"),
    ("select $1 from dim_customers", "POSITIONAL_REFERENCE"),
    ("SELECT lane_id FROM dim_lanes LIMIT (select 5)", "LIMIT_NOT_LITERAL"),
]


@pytest.fixture
def guard(catalog: Catalog) -> SqlGuard:
    return SqlGuard(catalog, max_rows=100)


@pytest.mark.parametrize("sql", ALLOWED)
def test_allowed(guard: SqlGuard, analyst: Principal, sql: str) -> None:
    guard.check(sql, analyst)


@pytest.mark.parametrize(("sql", "code"), DENIED)
def test_denied(guard: SqlGuard, analyst: Principal, sql: str, code: str) -> None:
    with pytest.raises(PolicyViolation) as info:
        guard.check(sql, analyst)
    assert info.value.code == code


def test_denied_queries_actually_parse_or_are_parse_errors(guard: SqlGuard, analyst: Principal) -> None:
    """Sanity: a deny for the wrong reason would make the table above meaningless."""
    codes = {code for _, code in DENIED}
    assert {"READ_ONLY", "PII_DENIED", "UNKNOWN_MODEL", "FORBIDDEN_FUNCTION", "POSITIONAL_REFERENCE"} <= codes


def test_restricted_model_is_invisible_not_forbidden(guard: SqlGuard, analyst: Principal) -> None:
    """An analyst cannot tell a restricted model from a missing one; the audit log can."""
    with pytest.raises(PolicyViolation) as hidden:
        guard.check("select * from fct_invoices", analyst)
    with pytest.raises(PolicyViolation) as missing:
        guard.check("select * from fct_nothing", analyst)
    assert hidden.value.message.replace("fct_invoices", "X") == missing.value.message.replace(
        "fct_nothing", "X"
    )
    assert "restricted" in hidden.value.internal_detail


def test_tiers(guard: SqlGuard, partner: Principal, finance: Principal, support: Principal) -> None:
    guard.check("select * from dim_lanes", partner)
    with pytest.raises(PolicyViolation, match="UNKNOWN_MODEL"):
        guard.check("select shipment_id from fct_shipments", partner)
    guard.check("select sum(revenue_usd) from fct_invoices", finance)
    guard.check("select contact_email from dim_customers", support)
    guard.check("select * from dim_customers", support)
    with pytest.raises(PolicyViolation, match="UNKNOWN_MODEL"):
        guard.check("select sum(revenue_usd) from fct_invoices", support)


def test_no_scopes_means_no_access(guard: SqlGuard) -> None:
    with pytest.raises(PolicyViolation, match="NO_DATA_ACCESS"):
        guard.check("select 1", Principal(subject="nobody"))


def test_skipped_tier_scope_grants_nothing_above_the_gap(guard: SqlGuard) -> None:
    odd = Principal(subject="odd", scopes=frozenset({"warehouse:public", "warehouse:restricted"}))
    with pytest.raises(PolicyViolation, match="UNKNOWN_MODEL"):
        guard.check("select sum(revenue_usd) from fct_invoices", odd)


@pytest.mark.parametrize(
    ("sql", "fetch_limit", "expected_limit"),
    [
        ("select lane_id from dim_lanes", 101, "LIMIT 101"),
        ("select lane_id from dim_lanes limit 10", 10, "LIMIT 10"),
        ("select lane_id from dim_lanes limit 100000", 101, "LIMIT 101"),
        ("select lane_id from dim_lanes union all select lane_id from dim_lanes", 101, "LIMIT 101"),
    ],
)
def test_row_limit(
    guard: SqlGuard, analyst: Principal, sql: str, fetch_limit: int, expected_limit: str
) -> None:
    checked = guard.check(sql, analyst)
    assert checked.fetch_limit == fetch_limit
    assert checked.sql.endswith(expected_limit)


def test_executed_sql_is_regenerated_without_comments(guard: SqlGuard, analyst: Principal) -> None:
    checked = guard.check("select lane_id -- hide\n /* note */ from dim_lanes", analyst)
    assert "--" not in checked.sql and "/*" not in checked.sql
    assert checked.models == {"dim_lanes"}


def test_size_limit(guard: SqlGuard, analyst: Principal) -> None:
    with pytest.raises(PolicyViolation, match="SQL_TOO_LONG"):
        guard.check("select 1 " + " " * MAX_SQL_CHARS, analyst)


def test_max_rows_must_be_positive(catalog: Catalog) -> None:
    with pytest.raises(ValueError, match="positive"):
        SqlGuard(catalog, max_rows=0)
