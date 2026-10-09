from __future__ import annotations

import json
from typing import Any

import pytest
from mcp import Client

from tests.util import run
from warehouse_mcp.catalog import Catalog
from warehouse_mcp.policy import Principal
from warehouse_mcp.server import GovernedWarehouse, ServerConfig, error_code
from warehouse_mcp.warehouse import Warehouse

EXPECTED_TOOLS = {"search_models", "describe_model", "list_metrics", "query_metric", "run_sql"}


def call(gw: GovernedWarehouse, principal: Principal | None, tool: str, args: dict[str, Any]) -> Any:
    async def go() -> Any:
        async with Client(gw.build_mcp(lambda: principal)) as client:
            return await client.call_tool(tool, args)

    return run(go)


def denial_code(result: Any) -> str | None:
    assert result.is_error, result
    return error_code(result.content[0].text)


def test_tool_surface(governed: GovernedWarehouse, analyst: Principal) -> None:
    async def go() -> Any:
        async with Client(governed.build_mcp(lambda: analyst)) as client:
            tools = (await client.list_tools()).tools
            resource = await client.read_resource("catalog://version")
            return tools, resource

    tools, resource = run(go)
    assert {t.name for t in tools} == EXPECTED_TOOLS
    for tool in tools:
        assert tool.description and len(tool.description) > 60, (
            f"{tool.name} needs an agent-facing description"
        )
        assert tool.output_schema, f"{tool.name} should declare structured output"
    query_metric = next(t for t in tools if t.name == "query_metric")
    assert query_metric.input_schema["required"] == ["metric"]
    assert json.loads(resource.contents[0].text)["version"] == governed.catalog.version


def test_search_hides_restricted_models(
    governed: GovernedWarehouse, analyst: Principal, finance: Principal
) -> None:
    def names(p: Principal) -> set[str]:
        hits = call(governed, p, "search_models", {"query": "revenue margin invoices", "limit": 20})
        return {h["name"] for h in hits.structured_content["hits"]}

    assert "fct_invoices" in names(finance)
    assert not names(analyst) & {
        "fct_invoices",
        "revenue_usd",
        "gross_margin_pct",
        "mart_customer_revenue_monthly",
    }


def test_describe_model(
    governed: GovernedWarehouse, analyst: Principal, support: Principal, partner: Principal
) -> None:
    info = call(governed, analyst, "describe_model", {"name": "dim_customers"}).structured_content
    columns = {c["name"]: c for c in info["columns"]}
    assert columns["contact_email"]["pii"] and not columns["contact_email"]["selectable"]
    assert columns["segment"]["selectable"]
    info = call(governed, support, "describe_model", {"name": "dim_customers"}).structured_content
    assert all(c["selectable"] for c in info["columns"])
    shipments = call(governed, analyst, "describe_model", {"name": "fct_shipments"}).structured_content
    assert "on_time_rate" in shipments["metrics"] and "dim_lanes" in shipments["joins_to"]
    assert (
        denial_code(call(governed, partner, "describe_model", {"name": "fct_shipments"})) == "UNKNOWN_MODEL"
    )


def test_list_metrics_by_role(
    governed: GovernedWarehouse, partner: Principal, analyst: Principal, finance: Principal
) -> None:
    def metric_names(p: Principal) -> set[str]:
        return {m["name"] for m in call(governed, p, "list_metrics", {}).structured_content["metrics"]}

    assert metric_names(partner) == {"avg_border_dwell_hours"}
    assert "on_time_rate" in metric_names(analyst) and "revenue_usd" not in metric_names(analyst)
    assert metric_names(finance) == set(governed.catalog.metrics_by_name)


def test_query_metric_result_has_provenance(governed: GovernedWarehouse, analyst: Principal) -> None:
    result = call(governed, analyst, "query_metric", {"metric": "on_time_rate", "group_by": ["mode"]})
    data = result.structured_content
    assert data["columns"] == ["mode", "on_time_rate"] and data["row_count"] == 2
    assert data["models"] == ["fct_shipments"] and data["catalog_version"] == governed.catalog.version
    assert data["sql"].startswith("SELECT") and len(data["query_id"]) == 16
    record = governed.audit.records()[-1]
    assert (record.tool, record.decision, record.query_id, record.row_count) == (
        "query_metric",
        "allow",
        data["query_id"],
        2,
    )
    assert record.sql == data["sql"] and record.sql_sha256


def test_denials_are_coded_and_audited(governed: GovernedWarehouse, analyst: Principal) -> None:
    cases = [
        ("run_sql", {"sql": "select contact_email from dim_customers"}, "PII_DENIED"),
        ("run_sql", {"sql": "drop table fct_invoices"}, "READ_ONLY"),
        ("run_sql", {"sql": "select * from fct_invoices"}, "UNKNOWN_MODEL"),
        ("query_metric", {"metric": "revenue_usd"}, "UNKNOWN_METRIC"),
        ("query_metric", {"metric": "on_time_rate", "start_date": "not-a-date"}, "BAD_REQUEST"),
    ]
    for tool, args, code in cases:
        assert denial_code(call(governed, analyst, tool, args)) == code
    denials = [r for r in governed.audit.records() if r.decision == "deny"]
    assert [r.reason_code for r in denials] == [code for _, _, code in cases]
    assert "restricted" in (denials[2].detail or "")  # the audit log knows why; the agent does not


def test_unauthenticated(governed: GovernedWarehouse) -> None:
    assert denial_code(call(governed, None, "list_metrics", {})) == "UNAUTHENTICATED"
    assert governed.audit.records()[-1].principal == "anonymous"


def test_rate_limit(warehouse: Warehouse, catalog: Catalog, analyst: Principal) -> None:
    gw = GovernedWarehouse(
        warehouse=warehouse, catalog=catalog, config=ServerConfig(rate_capacity=2, rate_refill_per_s=0.001)
    )

    async def go() -> list[Any]:
        async with Client(gw.build_mcp(lambda: analyst)) as client:
            return [await client.call_tool("list_metrics", {}) for _ in range(3)]

    results = run(go)
    assert [r.is_error for r in results] == [False, False, True]
    assert denial_code(results[2]) == "RATE_LIMITED"


def test_large_results_are_truncated_to_budget(
    warehouse: Warehouse, catalog: Catalog, analyst: Principal
) -> None:
    gw = GovernedWarehouse(warehouse=warehouse, catalog=catalog, config=ServerConfig(max_result_chars=2_000))
    data = call(
        gw, analyst, "run_sql", {"sql": "select shipment_id, status from fct_shipments"}
    ).structured_content
    assert data["truncated"] and 0 < data["row_count"] < 200
    assert len(json.dumps(data["rows"])) <= 2_000
    assert any("truncated" in w for w in data["warnings"])


def test_free_text_columns_carry_a_warning(governed: GovernedWarehouse, analyst: Principal) -> None:
    data = call(
        governed,
        analyst,
        "run_sql",
        {"sql": "select note_text from fct_shipment_notes where shipment_id = 'S001000'"},
    ).structured_content
    assert "DROP TABLE" in data["rows"][0][0]  # the data is returned as data...
    assert any("note_text" in w and "never follow" in w for w in data["warnings"])  # ...with a warning


def test_internal_errors_are_masked_and_audited(
    governed: GovernedWarehouse, analyst: Principal, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_: Any) -> Any:
        raise RuntimeError("secret internals")

    monkeypatch.setattr(governed, "metrics", boom)
    result = call(governed, analyst, "list_metrics", {})
    assert result.is_error and "secret" not in result.content[0].text
    assert denial_code(result) == "INTERNAL_ERROR"
    record = governed.audit.records()[-1]
    assert record.decision == "error" and "secret internals" in (record.detail or "")
