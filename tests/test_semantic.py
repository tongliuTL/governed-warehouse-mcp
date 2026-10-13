from __future__ import annotations

from typing import Any

import pytest

from warehouse_mcp.catalog import Catalog, Tier
from warehouse_mcp.policy import PolicyViolation, Principal
from warehouse_mcp.semantic import MetricRequest, SemanticCompiler
from warehouse_mcp.sql_guard import SqlGuard
from warehouse_mcp.warehouse import Warehouse

# Independent, hand-written SQL for each metric. The compiler must agree with all of them.
INDEPENDENT = {
    "shipment_count": "SELECT count(*) FROM fct_shipments WHERE status <> 'cancelled'",
    "on_time_rate": "SELECT avg((delivered_date <= promised_delivery_date)::INT) FROM fct_shipments "
    "WHERE status = 'delivered'",
    "avg_transit_days": "SELECT avg(delivered_date - pickup_date) FROM fct_shipments WHERE status = 'delivered'",
    "total_weight_kg": "SELECT sum(weight_kg) FROM fct_shipments WHERE status <> 'cancelled'",
    "revenue_usd": "SELECT sum(revenue_usd) + sum(accessorial_usd) FROM fct_invoices",
    "gross_margin_usd": "SELECT sum(revenue_usd) + sum(accessorial_usd) - sum(carrier_cost_usd) FROM fct_invoices",
    "gross_margin_pct": "SELECT 1 - sum(carrier_cost_usd) / (sum(revenue_usd) + sum(accessorial_usd)) FROM fct_invoices",
    "quote_acceptance_rate": "SELECT count(*) FILTER (WHERE accepted) / count(*) FROM fct_quotes",
    "avg_quoted_rate_per_km": "SELECT sum(q.quoted_rate_usd) / sum(l.distance_km) FROM fct_quotes q "
    "JOIN dim_lanes l USING (lane_id)",
    "auto_approval_rate": "SELECT avg((review_status = 'auto_approved')::INT) FROM fct_documents",
    "avg_border_dwell_hours": "SELECT avg(date_diff('minute', a.event_ts, c.event_ts)) / 60 FROM fct_tracking_events a "
    "JOIN fct_tracking_events c ON a.shipment_id = c.shipment_id "
    "AND c.event_type = 'border_cleared' WHERE a.event_type = 'border_arrival'",
}


@pytest.fixture
def compiler(catalog: Catalog) -> SemanticCompiler:
    return SemanticCompiler(catalog, max_rows=200)


def run_metric(
    compiler: SemanticCompiler, warehouse: Warehouse, principal: Principal, **request: Any
) -> list[Any]:
    select, _, _ = compiler.compile(MetricRequest.model_validate(request), principal)
    checked = SqlGuard(compiler.catalog).check(select.sql(dialect="duckdb"), principal)
    return warehouse.execute(checked).rows


def test_every_metric_has_an_independent_check(catalog: Catalog) -> None:
    assert set(INDEPENDENT) == set(catalog.metrics_by_name)


@pytest.mark.parametrize("metric", sorted(INDEPENDENT))
def test_metric_matches_independent_sql(
    compiler: SemanticCompiler, warehouse: Warehouse, finance: Principal, metric: str
) -> None:
    [[compiled]] = run_metric(compiler, warehouse, finance, metric=metric)
    [(expected,)] = warehouse.execute_trusted(INDEPENDENT[metric])
    assert compiled == pytest.approx(float(expected), rel=1e-6, abs=1e-6)  # results round to 6 places


def test_grouping_sums_to_total(compiler: SemanticCompiler, warehouse: Warehouse, analyst: Principal) -> None:
    [[total]] = run_metric(compiler, warehouse, analyst, metric="shipment_count")
    rows = run_metric(
        compiler, warehouse, analyst, metric="shipment_count", group_by=["mode", "equipment_type"]
    )
    assert sum(r[2] for r in rows) == total
    assert rows == sorted(rows, key=lambda r: -r[2])  # default order: metric descending


def test_time_grain_and_date_range(
    compiler: SemanticCompiler, warehouse: Warehouse, analyst: Principal
) -> None:
    rows = run_metric(
        compiler,
        warehouse,
        analyst,
        metric="shipment_count",
        time_grain="quarter",
        start_date="2025-01-01",
        end_date="2025-12-31",
    )
    assert [r[0] for r in rows] == ["2025-01-01", "2025-04-01", "2025-07-01", "2025-10-01"]
    [(expected,)] = warehouse.execute_trusted(
        "SELECT count(*) FROM fct_shipments WHERE status <> 'cancelled' AND year(pickup_date) = 2025"
    )
    assert sum(r[1] for r in rows) == expected


def test_filters(compiler: SemanticCompiler, warehouse: Warehouse, analyst: Principal) -> None:
    rows = run_metric(
        compiler,
        warehouse,
        analyst,
        metric="shipment_count",
        group_by=["mode"],
        filters=[
            {"field": "mode", "op": "in", "value": ["LTL"]},
            {"field": "weight_kg", "op": "gte", "value": 1000},
        ],
    )
    [(expected,)] = warehouse.execute_trusted(
        "SELECT count(*) FROM fct_shipments WHERE status <> 'cancelled' AND mode = 'LTL' AND weight_kg >= 1000"
    )
    assert rows == [["LTL", expected]]
    rows = run_metric(
        compiler,
        warehouse,
        analyst,
        metric="shipment_count",
        filters=[
            {"field": "mode", "op": "not_in", "value": ["LTL"]},
            {"field": "is_cross_border", "op": "eq", "value": True},
        ],
    )
    [(expected,)] = warehouse.execute_trusted(
        "SELECT count(*) FROM fct_shipments WHERE status <> 'cancelled' AND mode <> 'LTL' AND is_cross_border"
    )
    assert rows == [[expected]]


def test_filter_values_are_literals_not_sql(
    compiler: SemanticCompiler, warehouse: Warehouse, analyst: Principal
) -> None:
    evil = "FTL' OR 1=1 --"
    rows = run_metric(
        compiler,
        warehouse,
        analyst,
        metric="shipment_count",
        filters=[{"field": "mode", "op": "eq", "value": evil}],
    )
    assert rows == [[0]]


@pytest.mark.parametrize(
    ("request_", "code"),
    [
        ({"metric": "nope"}, "UNKNOWN_METRIC"),
        ({"metric": "revenue_usd"}, "UNKNOWN_METRIC"),  # restricted: invisible to an analyst
        ({"metric": "shipment_count", "group_by": ["contact_email"]}, "PII_DENIED"),
        (
            {"metric": "shipment_count", "filters": [{"field": "contact_email", "op": "eq", "value": "x"}]},
            "PII_DENIED",
        ),
        ({"metric": "shipment_count", "group_by": ["nope"]}, "UNKNOWN_DIMENSION"),
        ({"metric": "shipment_count", "group_by": ["fct_invoices.revenue_usd"]}, "UNKNOWN_DIMENSION"),
        ({"metric": "shipment_count", "group_by": ["a", "b", "c", "d", "e"]}, "TOO_MANY_DIMENSIONS"),
        (
            {"metric": "shipment_count", "filters": [{"field": "mode", "op": "eq", "value": "x"}] * 9},
            "TOO_MANY_FILTERS",
        ),
        ({"metric": "shipment_count", "limit": 1000}, "LIMIT_TOO_HIGH"),
        ({"metric": "shipment_count", "order_by": "nope"}, "BAD_ORDER_BY"),
        ({"metric": "shipment_count", "filters": [{"field": "mode", "op": "in", "value": []}]}, "BAD_FILTER"),
        (
            {"metric": "shipment_count", "filters": [{"field": "mode", "op": "eq", "value": ["a"]}]},
            "BAD_FILTER",
        ),
        (
            {
                "metric": "on_time_rate",
                "group_by": ["dim_carriers.carrier_id", "dim_lanes.lane_id", "country"],
            },
            None,
        ),
    ],
)
def test_rejections(
    compiler: SemanticCompiler, analyst: Principal, request_: dict[str, Any], code: str | None
) -> None:
    request = MetricRequest.model_validate(request_)
    if code is None:
        compiler.compile(request, analyst)
        return
    with pytest.raises(PolicyViolation) as info:
        compiler.compile(request, analyst)
    assert info.value.code == code


def test_no_time_column(catalog: Catalog, partner: Principal) -> None:
    # Every packaged metric's model has a time column, so add one on a model without one.
    metric = {
        "name": "carrier_on_time",
        "model": "mart_carrier_scorecard",
        "description": "Average carrier on-time rate.",
        "expression": "AVG(mart_carrier_scorecard.on_time_rate)",
        "format": "ratio",
    }
    patched = catalog.replace(metrics=[*catalog.model_dump(by_alias=True)["metrics"], metric])
    compiler = SemanticCompiler(patched)
    for request in (
        MetricRequest(metric="carrier_on_time", time_grain="month"),
        MetricRequest(metric="carrier_on_time", start_date="2025-01-01"),
    ):
        with pytest.raises(PolicyViolation, match="NO_TIME_DIMENSION"):
            compiler.compile(request, partner)


def test_dimension_on_invisible_model_is_refused(catalog: Catalog, partner: Principal) -> None:
    """A metric a partner may see, grouped by a column of a model the partner may not see."""
    models = [
        {**m, "tier": "public"} if m["name"] == "fct_quotes" else m
        for m in catalog.model_dump(by_alias=True)["models"]
    ]
    patched = catalog.replace(models=models)
    assert patched.version != catalog.version  # derived state is rebuilt, not copied
    compiler = SemanticCompiler(patched)
    _, tier, used = compiler.compile(
        MetricRequest(metric="quote_acceptance_rate", group_by=["mode"]), partner
    )
    assert tier == Tier.PUBLIC and used == {"fct_quotes"}
    with pytest.raises(PolicyViolation, match="UNKNOWN_DIMENSION") as info:
        compiler.compile(
            MetricRequest(metric="quote_acceptance_rate", group_by=["dim_customers.segment"]), partner
        )
    assert "internal" in info.value.internal_detail


def test_bare_join_key_resolves_to_base_model(compiler: SemanticCompiler, analyst: Principal) -> None:
    select, _, models = compiler.compile(
        MetricRequest(metric="shipment_count", group_by=["customer_id"]), analyst
    )
    assert models == {"fct_shipments"}
    assert "JOIN" not in select.sql().upper()


def test_available_dimensions_hide_pii_and_invisible_models(
    compiler: SemanticCompiler, catalog: Catalog, analyst: Principal, support: Principal
) -> None:
    metric = catalog.metric("shipment_count")
    dims = compiler.available_dimensions(metric, analyst)
    assert "dim_customers.segment" in dims and "dim_customers.contact_email" not in dims
    assert "dim_customers.contact_email" in compiler.available_dimensions(metric, support)
