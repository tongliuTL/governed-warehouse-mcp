"""The optional dense ranker: fusion, caches, model-ID safety and failover, with a fake
embedder (deterministic, no model download). One test uses the real model when it is cached."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Sequence
from pathlib import Path

import pytest

from warehouse_evals.cases import RetrievalCase
from warehouse_evals.retrieval import MIN_MRR_GAIN, RankerScore, dense_verdict, evaluate
from warehouse_mcp.audit import AuditRecord
from warehouse_mcp.catalog import Catalog
from warehouse_mcp.dense import DEFAULT_MODEL, DenseError, DenseRanker, FastEmbedder, default_cache_dir
from warehouse_mcp.discovery import LEXICAL_FALLBACK, RRF_K, Discovery, fuse, tokenize
from warehouse_mcp.policy import Principal
from warehouse_mcp.server import GovernedWarehouse

DIM = 64


class FakeEmbedder:
    """Hashed bag of words: texts that share words point the same way. Counts its calls."""

    def __init__(self, model_id: str = "fake-v1", *, fail: bool = False) -> None:
        self._model_id = model_id
        self.fail = fail
        self.document_calls = 0
        self.query_calls = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    def _vector(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for token in tokenize(text) or ["empty"]:
            v[int(hashlib.sha256(token.encode()).hexdigest(), 16) % DIM] += 1.0
        return v

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if self.fail:
            raise RuntimeError("model not available")
        self.document_calls += 1
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        if self.fail:
            raise RuntimeError("model not available")
        self.query_calls += 1
        return self._vector(text)


def test_rrf_fuses_three_lists() -> None:
    fused = dict(fuse([[(0, 9.0), (1, 5.0)], [(1, 2.0), (2, 1.0)], [(2, 0.9), (1, 0.8)]]))
    assert fused[1] == pytest.approx(1 / (RRF_K + 2) + 1 / (RRF_K + 1) + 1 / (RRF_K + 2))
    assert fused[2] == pytest.approx(1 / (RRF_K + 2) + 1 / (RRF_K + 1))
    assert fused[0] == pytest.approx(1 / (RRF_K + 1))
    assert max(fused, key=fused.__getitem__) == 1  # agreement across lists wins


def test_dense_joins_the_fusion(catalog: Catalog) -> None:
    dense = DenseRanker(FakeEmbedder())
    discovery = Discovery(catalog, dense)
    assert discovery.default_ranker == "rrf+dense"
    lexical, used_lexical = discovery.rank_explained("border wait time", "rrf")
    fused, used = discovery.rank_explained("border wait time", "rrf+dense")
    assert (used_lexical, used) == ("rrf", "rrf+dense")
    assert fused != lexical  # the third list moved the scores
    only_dense, _ = discovery.rank_explained("border wait time", "dense")
    assert only_dense and len(only_dense) <= 20
    assert Discovery(catalog).default_ranker == "rrf"  # off unless a ranker is passed in


def test_query_cache_hits_and_evicts() -> None:
    embedder = FakeEmbedder()
    dense = DenseRanker(embedder, query_cache_size=2)
    first = dense.query("late deliveries")
    assert dense.query("late deliveries") == first
    assert (embedder.query_calls, dense.stats.query_cache_hits) == (1, 1)
    dense.query("unpaid invoices")
    dense.query("border dwell")  # evicts "late deliveries", the least recently used
    dense.query("late deliveries")
    assert embedder.query_calls == 4
    assert math.isclose(sum(x * x for x in first), 1.0)  # normalized for cosine similarity


def test_document_vectors_are_cached_per_catalog_version(tmp_path: Path) -> None:
    names, texts = ["model:a", "model:b"], ["border wait", "unpaid invoices"]
    first = FakeEmbedder()
    DenseRanker(first, cache_dir=tmp_path).documents("v1-abc", names, texts)
    (cached,) = tmp_path.glob("*.json")
    assert cached.name == "v1-abc.fake-v1.json"
    second = FakeEmbedder()
    ranker = DenseRanker(second, cache_dir=tmp_path)
    ranker.documents("v1-abc", names, texts)
    assert (second.document_calls, ranker.stats.document_cache_hits) == (0, 1)
    ranker.documents("v1-def", names, texts)  # a new catalog version re-embeds
    assert second.document_calls == 1


def test_vectors_from_another_model_are_refused(tmp_path: Path) -> None:
    names, texts = ["model:a"], ["border wait"]
    DenseRanker(FakeEmbedder("model-a"), cache_dir=tmp_path).documents("v1", names, texts)
    # A file for model-b that holds model-a's vectors (a copy, a rename, a slug collision).
    (tmp_path / "v1.model-a.json").rename(tmp_path / "v1.model-b.json")
    with pytest.raises(DenseError, match="refusing to mix"):
        DenseRanker(FakeEmbedder("model-b"), cache_dir=tmp_path).documents("v1", names, texts)


def test_failure_fails_over_to_lexical(catalog: Catalog, caplog: pytest.LogCaptureFixture) -> None:
    dense = DenseRanker(FakeEmbedder(fail=True))
    discovery = Discovery(catalog, dense)
    with caplog.at_level(logging.WARNING, logger="warehouse_mcp.discovery"):
        hits, used = discovery.search_explained("how long do trucks wait at the border", k=3)
    assert used == LEXICAL_FALLBACK
    assert hits == Discovery(catalog).search("how long do trucks wait at the border", k=3)
    assert dense.stats.failovers == 1
    assert "dense ranker failed" in caplog.text


def test_failover_is_recorded_in_the_audit_log(governed: GovernedWarehouse, catalog: Catalog) -> None:
    governed.discovery = Discovery(catalog, DenseRanker(FakeEmbedder(fail=True)))
    record = AuditRecord(tool="search_models", principal="t", decision="allow", catalog_version="v")
    result = governed.search(Principal.for_role("analyst"), "border wait", 3, record)
    assert result.hits and record.ranker == LEXICAL_FALLBACK
    governed.discovery = Discovery(catalog, DenseRanker(FakeEmbedder()))
    governed.search(Principal.for_role("analyst"), "border wait", 3, record)
    assert record.ranker == "rrf+dense"


def test_hidden_entries_never_take_a_slot_with_dense(catalog: Catalog) -> None:
    discovery = Discovery(catalog, DenseRanker(FakeEmbedder()))
    analyst = Principal.for_role("analyst")
    hidden = {m.name for m in catalog.models if not analyst.can_see(m)}
    hidden |= {m.name for m in catalog.metrics if not analyst.can_see_tier(catalog.metric_tier(m))}
    hits = discovery.search("invoice revenue margin billing", analyst, k=5)
    assert len(hits) == 5 and not {h.name for h in hits} & hidden


def test_eval_adds_dense_rankers_and_refuses_failover(catalog: Catalog) -> None:
    cases = [
        RetrievalCase(query="how long do trucks wait at the border", relevant=("mart_border_crossing_dwell",))
    ]
    rankers = [s.ranker for s in evaluate(catalog, cases, DenseRanker(FakeEmbedder()))]
    assert rankers[-2:] == ["dense", "rrf+dense"]
    assert "dense" not in [s.ranker for s in evaluate(catalog, cases)]
    with pytest.raises(DenseError, match="failed over"):
        evaluate(catalog, cases, DenseRanker(FakeEmbedder(fail=True)))


def _scores(rrf: tuple[float, float], dense: tuple[float, float]) -> list[RankerScore]:
    return [
        RankerScore("rrf", {1: 0.0, 3: rrf[0], 5: 1.0}, rrf[1]),
        RankerScore("rrf+dense", {1: 0.0, 3: dense[0], 5: 1.0}, dense[1]),
    ]


@pytest.mark.parametrize(
    ("dev", "heldout", "verdict"),
    [
        (((0.95, 0.8), (0.95, 0.8)), ((0.90, 0.8), (0.95, 0.8)), "turn dense on"),  # Recall@3 gain
        (((0.95, 0.8), (0.95, 0.8)), ((1.0, 0.8), (1.0, 0.8 + MIN_MRR_GAIN)), "turn dense on"),
        (((0.95, 0.8), (0.95, 0.8)), ((1.0, 0.8), (1.0, 0.81)), "keep dense off: no gain"),  # noise
        (((0.95, 0.8), (0.85, 0.9)), ((0.90, 0.8), (1.0, 0.9)), "keep dense off: rrf+dense Recall@3"),
    ],
)
def test_dense_verdict(dev: tuple, heldout: tuple, verdict: str) -> None:
    results = {"dev set": _scores(*dev), "held-out set": _scores(*heldout)}
    assert (dense_verdict(results) or "").startswith(verdict)
    assert dense_verdict({"dev set": [RankerScore("rrf", {3: 1.0}, 1.0)]}) is None


def test_fast_embedder_without_the_extra_is_a_dense_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "fastembed", None)  # import fails as if not installed
    with pytest.raises(DenseError, match="'dense' extra"):
        FastEmbedder().embed_query("x")


def _model_cached() -> bool:
    models = default_cache_dir() / "models"
    return models.exists() and any(models.iterdir())


@pytest.mark.skipif(not _model_cached(), reason="the dense model is not downloaded (gwm-eval all --dense)")
def test_real_model_finds_paraphrases(catalog: Catalog) -> None:
    pytest.importorskip("fastembed")
    embedder = FastEmbedder(DEFAULT_MODEL, default_cache_dir() / "models")
    hits = Discovery(catalog, DenseRanker(embedder)).search("how long do trucks wait at the border", k=3)
    assert {"mart_border_crossing_dwell", "avg_border_dwell_hours"} & {h.name for h in hits}
    vector = embedder.embed_query("border")
    assert len(vector) == 384 and json.dumps(vector)
