from __future__ import annotations

import pytest

from warehouse_mcp.catalog import Catalog
from warehouse_mcp.discovery import BM25, Discovery, tokenize
from warehouse_mcp.policy import Principal


@pytest.mark.parametrize(
    ("text", "tokens"),
    [
        ("border_crossings", ["border", "crossing"]),
        ("onTimeRate", ["time", "rate"]),  # "on" is a stopword
        ("What are the carriers' fleet sizes?", ["carrier", "fleet", "size"]),
        ("deliveries, boxes and status", ["delivery", "box", "status"]),
    ],
)
def test_tokenize(text: str, tokens: list[str]) -> None:
    assert tokenize(text) == tokens


def test_bm25_prefers_rarer_and_denser_matches() -> None:
    bm25 = BM25([["border", "wait"], ["border", "shipment", "shipment"], ["invoice"]])
    scores = bm25.scores(["border", "wait"])
    assert scores[0] > scores[1] > scores[2] == 0


def test_search_finds_metrics_and_models(catalog: Catalog) -> None:
    hits = Discovery(catalog).search("how long do trucks wait at the border", k=3)
    assert {"mart_border_crossing_dwell", "avg_border_dwell_hours"} & {h.name for h in hits}
    assert all(h.score > 0 for h in hits)


def test_search_respects_access(catalog: Catalog) -> None:
    discovery = Discovery(catalog)
    partner = {
        h.name for h in discovery.search("shipments customers revenue", Principal.for_role("partner"), k=50)
    }
    assert partner <= {m.name for m in catalog.models if m.tier_name == "public"} | {"avg_border_dwell_hours"}
    assert discovery.search("revenue", Principal("nobody"), k=5) == []


@pytest.mark.parametrize("ranker", ["substring", "identifier", "description", "rrf"])
def test_rankers_return_ordered_scores(catalog: Catalog, ranker: str) -> None:
    hits = Discovery(catalog).search("carrier on time rate", k=5, ranker=ranker)  # type: ignore[arg-type]
    assert hits and [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_kind_filter(catalog: Catalog) -> None:
    hits = Discovery(catalog).search("on time rate", k=5, kind=lambda e: e.kind == "metric")
    assert hits and all(h.kind == "metric" for h in hits)
