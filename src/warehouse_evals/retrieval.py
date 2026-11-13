"""Retrieval eval with an ablation across rankers: Recall@k and MRR.

With a ``DenseRanker`` the ablation adds ``dense`` (embeddings alone) and ``rrf+dense`` (the
two lexical lists plus dense, fused). ``dense_verdict`` applies the rule for turning dense on.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from warehouse_evals.cases import RetrievalCase
from warehouse_mcp.catalog import Catalog
from warehouse_mcp.dense import DenseError, DenseRanker
from warehouse_mcp.discovery import Discovery, Ranker

RANKERS: tuple[Ranker, ...] = ("substring", "identifier", "description", "rrf")
DENSE_RANKERS: tuple[Ranker, ...] = ("dense", "rrf+dense")
GATE_RECALL_AT_3 = 0.9
# The smallest held-out MRR gain that counts. 0.02 is about one query of 24 moving from second
# to first place; less is within what a single query's rank can swing.
MIN_MRR_GAIN = 0.02
KS = (1, 3, 5)


@dataclass
class RankerScore:
    ranker: str
    recall: dict[int, float]
    mrr: float
    misses_at_3: list[str] = field(default_factory=list)


def evaluate(
    catalog: Catalog, cases: Sequence[RetrievalCase], dense: DenseRanker | None = None
) -> list[RankerScore]:
    discovery = Discovery(catalog, dense)
    scores = []
    for ranker in RANKERS + (DENSE_RANKERS if dense is not None else ()):
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
    if dense is not None and dense.stats.failovers:
        # A failover serves lexical results under the dense name: the numbers would be wrong.
        raise DenseError(f"dense ranker failed over {dense.stats.failovers} times; see the log")
    return scores


def dense_verdict(results: dict[str, list[RankerScore]], heldout: str = "held-out set") -> str | None:
    """Turn dense on only with no regression on the gate and a measured gain where it should
    help: rrf+dense keeps Recall@3 at or above the gate on every set, and beats rrf on the
    held-out set (higher Recall@3, or MRR up by at least ``MIN_MRR_GAIN`` at equal Recall@3).
    None when dense was not measured."""
    by_set = {name: {s.ranker: s for s in scores} for name, scores in results.items()}
    if not all("rrf+dense" in rankers for rankers in by_set.values()):
        return None
    low = [n for n, r in by_set.items() if r["rrf+dense"].recall[3] < GATE_RECALL_AT_3]
    if low:
        return f"keep dense off: rrf+dense Recall@3 is below {GATE_RECALL_AT_3} on " + ", ".join(low)
    base, cand = by_set[heldout]["rrf"], by_set[heldout]["rrf+dense"]
    r3, mrr = cand.recall[3] - base.recall[3], cand.mrr - base.mrr
    if r3 > 0 or (r3 == 0 and mrr >= MIN_MRR_GAIN):
        return f"turn dense on: on the {heldout}, Recall@3 {r3:+.3f} and MRR {mrr:+.3f} over rrf"
    return (
        f"keep dense off: no gain on the {heldout} beyond noise (Recall@3 {r3:+.3f}, "
        f"MRR {mrr:+.3f} over rrf; a gain needs Recall@3 or MRR +{MIN_MRR_GAIN})"
    )
