"""Retrieval eval with an ablation across rankers: Recall@k and MRR."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from warehouse_evals.cases import RetrievalCase
from warehouse_mcp.catalog import Catalog
from warehouse_mcp.discovery import Discovery, Ranker

RANKERS: tuple[Ranker, ...] = ("substring", "identifier", "description", "rrf")
KS = (1, 3, 5)


@dataclass
class RankerScore:
    ranker: str
    recall: dict[int, float]
    mrr: float
    misses_at_3: list[str] = field(default_factory=list)


def evaluate(catalog: Catalog, cases: Sequence[RetrievalCase]) -> list[RankerScore]:
    discovery = Discovery(catalog)
    scores = []
    for ranker in RANKERS:
        hits_at = dict.fromkeys(KS, 0)
        reciprocal = 0.0
        misses: list[str] = []
        for case in cases:
            names = [h.name for h in discovery.search(case.query, None, k=max(KS), ranker=ranker)]
            rank = next((i + 1 for i, n in enumerate(names) if n in case.relevant), None)
            for k in KS:
                hits_at[k] += int(rank is not None and rank <= k)
            reciprocal += 1.0 / rank if rank else 0.0
            if rank is None or rank > 3:
                misses.append(case.query)
        n = len(cases)
        scores.append(RankerScore(ranker, {k: hits_at[k] / n for k in KS}, reciprocal / n, misses))
    return scores
