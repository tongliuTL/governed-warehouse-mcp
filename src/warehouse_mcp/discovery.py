"""Find the right model or metric for a natural-language question.

Two lexical rankers are fused with Reciprocal Rank Fusion (RRF, k=60):

- an identifier ranker: BM25 over names, aliases, synonyms and column names, after splitting
  snake_case and camelCase and folding plurals, so "border crossings" matches
  ``border_crossing`` and ``mart_border_crossing_dwell``;
- a description ranker: BM25 over descriptions, grain and column descriptions.

Access is enforced by construction: there is one index per visibility tier, built only from the
entries that tier may see, and a caller searches the index for its highest tier. Filtering
after ranking would hide restricted entries but still let them shape IDF, length
normalisation and RRF ranks, so a caller could infer from scores that a restricted entry
matching its query exists.

An optional third list, a dense ranker over a local embedding model (``dense.py``), joins the
fusion when a ``DenseRanker`` is passed in (``rrf+dense``). Entry vectors are computed once for
the whole catalog (an entry's vector depends on its own text only), but the dense list is
ranked over the caller's tier index alone, so its positions, and the RRF scores they feed,
never depend on an entry the caller cannot see. It is off by default, so the server has no
model dependency and results are deterministic. If the dense ranker fails for any reason, the
query is served by the two lexical lists, a warning is logged and a failover is counted.

A naive substring baseline is kept for the retrieval eval's ablation.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from warehouse_mcp.catalog import Catalog, Tier
from warehouse_mcp.dense import DenseRanker
from warehouse_mcp.policy import Principal

log = logging.getLogger(__name__)

DENSE_CANDIDATES = 20  # the dense list joins the fusion with its top entries only
RRF_K = 60
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "in",
        "is",
        "it",
        "its",
        "many",
        "me",
        "much",
        "of",
        "on",
        "or",
        "our",
        "per",
        "show",
        "that",
        "the",
        "their",
        "them",
        "there",
        "these",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "you",
        "your",
        "give",
        "list",
        "tell",
        "get",
        "all",
    ]
)
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SPLIT = re.compile(r"[^A-Za-z0-9]+")


def _fold(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("ches", "shes", "sses", "xes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in _SPLIT.split(_CAMEL.sub(" ", text)):
        token = raw.lower()
        if token and token not in _STOPWORDS:
            tokens.append(_fold(token))
    return tokens


class BM25:
    def __init__(self, docs: Sequence[list[str]], k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = [Counter(d) for d in docs]
        self.lengths = [len(d) for d in docs]
        self.avg_len = (sum(self.lengths) / len(docs)) if docs else 0.0
        df: Counter[str] = Counter()
        for doc in self.docs:
            df.update(doc.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for doc, length in zip(self.docs, self.lengths, strict=True):
            score = 0.0
            norm = self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1.0))
            for term in query:
                tf = doc.get(term, 0)
                if tf:
                    score += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
            out.append(score)
        return out


@dataclass(frozen=True)
class Entry:
    kind: Literal["model", "metric"]
    name: str
    description: str
    tier: Tier
    identifier_text: str
    description_text: str


@dataclass(frozen=True)
class Hit:
    kind: Literal["model", "metric"]
    name: str
    description: str
    score: float


@dataclass
class _Index:
    entries: list[Entry]
    positions: list[int]  # each entry's position in the full catalog list (for dense vectors)
    identifier: BM25 = field(init=False)
    description: BM25 = field(init=False)

    def __post_init__(self) -> None:
        self.identifier = BM25([tokenize(e.identifier_text) for e in self.entries])
        self.description = BM25([tokenize(e.description_text) for e in self.entries])


Ranker = Literal["substring", "identifier", "description", "rrf", "dense", "rrf+dense"]
LEXICAL_FALLBACK = "rrf (dense failover)"


class Discovery:
    def __init__(self, catalog: Catalog, dense: DenseRanker | None = None) -> None:
        self.catalog = catalog
        self.dense = dense
        self.default_ranker: Ranker = "rrf+dense" if dense is not None else "rrf"
        entries: list[Entry] = []
        for model in catalog.models:
            entries.append(
                Entry(
                    kind="model",
                    name=model.name,
                    description=model.description,
                    tier=model.tier,
                    identifier_text=" ".join([model.name, *model.aliases, *(c.name for c in model.columns)]),
                    description_text=" ".join(
                        [model.description, model.grain, *(c.description for c in model.columns)]
                    ),
                )
            )
        for metric in catalog.metrics:
            entries.append(
                Entry(
                    kind="metric",
                    name=metric.name,
                    description=metric.description,
                    tier=catalog.metric_tier(metric),
                    identifier_text=" ".join([metric.name, *metric.synonyms]),
                    description_text=metric.description,
                )
            )
        self.entries = entries
        # One index per visibility tier, each built only from the entries that tier may see.
        # BM25 statistics (IDF, average length) and the ranks RRF fuses then never depend on an
        # entry the caller cannot see, so neither scores nor order can hint that one exists.
        self._tiers = {
            tier: _Index(
                [e for e in entries if e.tier <= tier], [i for i, e in enumerate(entries) if e.tier <= tier]
            )
            for tier in Tier
        }

    def _index_for(self, principal: Principal | None) -> _Index | None:
        """The index this caller searches; ``None`` (unfiltered, for offline evals) gets the
        full catalog, a principal with no data scope gets nothing."""
        if principal is None:
            return self._tiers[max(Tier)]
        tier = principal.max_tier
        return None if tier is None else self._tiers[tier]

    def _dense_ranked(self, query: str, index: _Index) -> list[tuple[int, float]] | None:
        """The dense list over ``index`` (positions within it), or None when there is no dense
        ranker or it failed (a failover). Vectors cover the whole catalog and are cached once per
        catalog version; only the tier's entries are scored and ranked."""
        if self.dense is None:
            return None
        try:
            scores = self.dense.scores(
                query,
                self.catalog.version,
                [f"{e.kind}:{e.name}" for e in self.entries],
                [_dense_text(e) for e in self.entries],
            )
        except Exception as err:  # any failure: serve the lexical result instead
            self.dense.stats.failovers += 1
            log.warning("dense ranker failed, serving lexical results: %s", err)
            return None
        return self._order([scores[p] for p in index.positions])[:DENSE_CANDIDATES]

    def rank(
        self, query: str, ranker: Ranker = "rrf", principal: Principal | None = None
    ) -> list[tuple[Entry, float]]:
        """Entries with scores, best first, from the index the principal may search."""
        return self.rank_explained(query, ranker, principal)[0]

    def rank_explained(
        self, query: str, ranker: Ranker = "rrf", principal: Principal | None = None
    ) -> tuple[list[tuple[Entry, float]], str]:
        """``rank`` plus the ranker that actually served the query, which differs from the one
        asked for when the dense ranker failed over to the lexical lists."""
        index = self._index_for(principal)
        if index is None:
            return [], ranker
        tokens = tokenize(query)
        used: str = ranker
        if ranker == "substring":
            words = [w for w in query.lower().split() if w not in _STOPWORDS]
            raw = [float(sum(w in e.identifier_text.lower() for w in words)) for e in index.entries]
            ranked = self._order(raw)
        elif ranker == "identifier":
            ranked = self._order(index.identifier.scores(tokens))
        elif ranker == "description":
            ranked = self._order(index.description.scores(tokens))
        else:
            lists = [
                self._order(index.identifier.scores(tokens)),
                self._order(index.description.scores(tokens)),
            ]
            wants_dense = ranker in ("dense", "rrf+dense")
            dense = self._dense_ranked(query, index) if wants_dense else None
            if dense is None:
                used = LEXICAL_FALLBACK if wants_dense else "rrf"
                ranked = fuse(lists)
            elif ranker == "dense":
                ranked = dense
            else:
                ranked = fuse([*lists, dense])
        return [(index.entries[i], score) for i, score in ranked], used

    @staticmethod
    def _order(scores: list[float]) -> list[tuple[int, float]]:
        ranked = [(i, s) for i, s in enumerate(scores) if s > 0]
        return sorted(ranked, key=lambda kv: (-kv[1], kv[0]))

    def search(
        self,
        query: str,
        principal: Principal | None = None,
        k: int = 5,
        ranker: Ranker | None = None,
        kind: Callable[[Entry], bool] | None = None,
    ) -> list[Hit]:
        return self.search_explained(query, principal, k, ranker, kind)[0]

    def search_explained(
        self,
        query: str,
        principal: Principal | None = None,
        k: int = 5,
        ranker: Ranker | None = None,
        kind: Callable[[Entry], bool] | None = None,
    ) -> tuple[list[Hit], str]:
        """``search`` plus the ranker that served it (see ``rank_explained``)."""
        hits: list[Hit] = []
        ranked, used = self.rank_explained(query, ranker or self.default_ranker, principal)
        for entry, score in ranked:
            if kind is not None and not kind(entry):
                continue
            hits.append(
                Hit(kind=entry.kind, name=entry.name, description=entry.description, score=round(score, 4))
            )
            if len(hits) == k:
                break
        return hits, used


def fuse(lists: Sequence[Sequence[tuple[int, float]]]) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: each list adds 1 / (k + rank), one-based, k = 60."""
    fused: dict[int, float] = {}
    for ranked in lists:
        for position, (i, _) in enumerate(ranked):
            fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + position + 1)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


def _dense_text(entry: Entry) -> str:
    """What the dense ranker embeds: the name in words, then the description text."""
    return f"{entry.name.replace('_', ' ')}: {entry.description_text}"
