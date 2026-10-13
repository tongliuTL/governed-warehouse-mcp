from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
import yaml
from pydantic import ValidationError

from warehouse_mcp.catalog import Catalog, Tier, load_catalog

CATALOG_YAML = Path(__file__).parents[1] / "src/warehouse_mcp/catalog/catalog.yaml"


def test_catalog_matches_warehouse_in_both_directions(catalog: Catalog, raw_db: Path) -> None:
    """Every documented column exists with the stated type, and nothing in the warehouse is undocumented."""
    with duckdb.connect(str(raw_db), read_only=True) as con:
        objects = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
        assert objects == set(catalog.by_name), "catalog and warehouse disagree on which models exist"
        for model in catalog.models:
            actual = {r[0]: r[1] for r in con.execute(f"DESCRIBE {model.name}").fetchall()}
            declared = {c.name: c.type for c in model.columns}
            assert actual == declared, model.name


def test_every_metric_compiles_against_the_warehouse(catalog: Catalog, raw_db: Path) -> None:
    with duckdb.connect(str(raw_db), read_only=True) as con:
        for metric in catalog.metrics:
            joins = " ".join(
                f"LEFT JOIN {j.model} ON {metric.model}.{j.key} = {j.model}.{j.key}"
                for j in catalog.model(metric.model).joins
            )
            where = f"WHERE {metric.where}" if metric.where else ""
            (value,) = con.execute(  # type: ignore[misc]
                f"SELECT {metric.expression} FROM {metric.model} {joins} {where}"
            ).fetchone()
            assert value is not None, metric.name


def test_metric_tiers(catalog: Catalog) -> None:
    tiers = {m.name: catalog.metric_tier(m) for m in catalog.metrics}
    assert tiers["revenue_usd"] == tiers["gross_margin_pct"] == Tier.RESTRICTED
    assert tiers["on_time_rate"] == Tier.INTERNAL
    assert tiers["avg_border_dwell_hours"] == Tier.PUBLIC


def test_pii_columns(catalog: Catalog) -> None:
    assert catalog.pii_column_names == {"contact_name", "contact_email", "contact_phone"}


def test_version_is_stable_and_content_addressed(tmp_path: Path) -> None:
    first, second = load_catalog(), load_catalog(CATALOG_YAML)
    assert first.version == second.version
    data = yaml.safe_load(CATALOG_YAML.read_text())
    data["models"][0]["description"] += " Edited."
    edited = tmp_path / "catalog.yaml"
    edited.write_text(yaml.safe_dump(data))
    assert load_catalog(edited).version != first.version


def _broken(mutate: object) -> dict[str, object]:
    data = yaml.safe_load(CATALOG_YAML.read_text())
    mutate(data)  # type: ignore[operator]
    return data  # type: ignore[no-any-return]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda d: d["models"][3]["joins"].append({"model": "nope", "key": "lane_id"}), "unknown model"),
        (lambda d: d["models"][3]["joins"].append({"model": "dim_lanes", "key": "nope"}), "join key"),
        (
            lambda d: d["metrics"].append(
                {
                    **d["metrics"][0],
                    "name": "leaky",
                    "expression": "COUNT(DISTINCT dim_customers.contact_email)",
                }
            ),
            "PII",
        ),
        (
            lambda d: d["metrics"].append(
                {**d["metrics"][0], "name": "bare", "expression": "SUM(weight_kg)"}
            ),
            "model-qualified",
        ),
        (
            lambda d: d["metrics"].append(
                {**d["metrics"][0], "name": "far", "expression": "SUM(fct_quotes.quoted_rate_usd)"}
            ),
            "not reachable",
        ),
        (lambda d: d["metrics"].append(d["metrics"][0]), "duplicate metric"),
        (lambda d: d["models"].append(d["models"][0]), "duplicate model"),
        (lambda d: d["models"][0]["columns"].append(d["models"][0]["columns"][0]), "duplicate column"),
        (lambda d: d["models"][3].update(time_column="nope"), "time_column"),
        (lambda d: d["models"][0].update(surprise=True), "Extra inputs"),
    ],
)
def test_invalid_catalogs_are_rejected(mutation: object, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        Catalog.model_validate(_broken(mutation))


def test_lookup_errors(catalog: Catalog) -> None:
    with pytest.raises(KeyError, match="unknown model"):
        catalog.model("nope")
    with pytest.raises(KeyError, match="unknown metric"):
        catalog.metric("nope")
