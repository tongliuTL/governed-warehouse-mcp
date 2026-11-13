"""Optional dense ranker: a third list for Reciprocal Rank Fusion, off by default.

The lexical rankers (``discovery.py``) need no model and give the same answer every time, so
they stay the default. The dense ranker embeds each catalog entry and the query with a local
model (``fastembed``, ONNX; install with the ``dense`` extra), so no catalog text or question
leaves the machine.

- **Document vectors are computed once per catalog version** and cached on disk with the
  embedding model ID. A cache written by a different model is refused, never mixed in:
  cosine similarity between vectors from two models is meaningless.
- **Query vectors go through an LRU cache** (1,000 queries by default).
- **Failure is not an error for the caller.** ``Discovery`` catches any exception from this
  module, logs a warning, counts a failover and serves the lexical result.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_CACHE_SIZE = 1_000


def default_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "governed-warehouse-mcp" / "dense"


class DenseError(Exception):
    """The dense ranker cannot serve this query (model missing, cache mismatch, ...)."""


class Embedder(Protocol):
    """Turns text into vectors. ``model_id`` names the model, so cached vectors can be matched
    to the model that produced them."""

    @property
    def model_id(self) -> str: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedder:
    """A local ONNX embedding model through ``fastembed``. The model loads on first use, so
    constructing one never downloads anything."""

    def __init__(self, model_id: str = DEFAULT_MODEL, cache_dir: Path | None = None) -> None:
        self._model_id = model_id
        self._cache_dir = cache_dir or default_cache_dir() / "models"
        self._model: object | None = None

    @property
    def model_id(self) -> str:
        return self._model_id

    def _load(self) -> object:
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as err:  # the extra is not installed
                raise DenseError("dense retrieval needs the 'dense' extra (fastembed)") from err
            self._model = TextEmbedding(self._model_id, cache_dir=str(self._cache_dir))
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._load()
        return [[float(x) for x in v] for v in model.passage_embed(list(texts))]  # type: ignore[attr-defined]

    def embed_query(self, text: str) -> list[float]:
        model = self._load()
        return [float(x) for x in next(iter(model.query_embed([text])))]  # type: ignore[attr-defined]


def _normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        raise DenseError("embedder returned a zero vector")
    return [x / norm for x in vector]


@dataclass
class DenseStats:
    queries: int = 0
    query_cache_hits: int = 0
    document_cache_hits: int = 0
    failovers: int = 0


class DenseRanker:
    """Cosine similarity between the query and each entry, with both caches.

    ``names`` and ``texts`` are the entries in index order; ``catalog_version`` keys the
    document cache, so a catalog change re-embeds and a stale file is never read.
    """

    def __init__(
        self,
        embedder: Embedder,
        *,
        cache_dir: Path | None = None,
        query_cache_size: int = QUERY_CACHE_SIZE,
    ) -> None:
        self.embedder = embedder
        self.cache_dir = cache_dir
        self.query_cache_size = query_cache_size
        self.stats = DenseStats()
        self._queries: OrderedDict[str, list[float]] = OrderedDict()
        self._docs: list[list[float]] | None = None
        self._key: tuple[str, tuple[str, ...]] | None = None

    def _cache_file(self, catalog_version: str) -> Path | None:
        if self.cache_dir is None:
            return None
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", self.embedder.model_id)
        return self.cache_dir / f"{catalog_version}.{slug}.json"

    def _read_cache(self, path: Path, catalog_version: str, names: Sequence[str]) -> list[list[float]] | None:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("model_id") != self.embedder.model_id:
            raise DenseError(
                f"cached vectors in {path} come from {data.get('model_id')!r}, not "
                f"{self.embedder.model_id!r}; refusing to mix embedding models"
            )
        if data.get("catalog_version") != catalog_version or data.get("names") != list(names):
            return None  # another catalog: re-embed and overwrite
        return [list(map(float, v)) for v in data["vectors"]]

    def documents(
        self, catalog_version: str, names: Sequence[str], texts: Sequence[str]
    ) -> list[list[float]]:
        """Entry vectors for this catalog version: memory, then disk, then the model."""
        key = (catalog_version, tuple(names))
        if self._docs is not None and self._key == key:
            return self._docs
        path = self._cache_file(catalog_version)
        vectors = self._read_cache(path, catalog_version, names) if path else None
        if vectors is not None:
            self.stats.document_cache_hits += 1
        else:
            vectors = [_normalize(v) for v in self.embedder.embed_documents(texts)]
            if len(vectors) != len(names):
                raise DenseError(f"embedder returned {len(vectors)} vectors for {len(names)} entries")
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "model_id": self.embedder.model_id,
                    "catalog_version": catalog_version,
                    "names": list(names),
                    "vectors": vectors,
                }
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(payload), encoding="utf-8")
                tmp.replace(path)  # atomic: a reader never sees half a file
        self._docs, self._key = vectors, key
        return vectors

    def query(self, text: str) -> list[float]:
        self.stats.queries += 1
        if text in self._queries:
            self.stats.query_cache_hits += 1
            self._queries.move_to_end(text)
            return self._queries[text]
        vector = _normalize(self.embedder.embed_query(text))
        self._queries[text] = vector
        if len(self._queries) > self.query_cache_size:
            self._queries.popitem(last=False)
        return vector

    def scores(
        self, query: str, catalog_version: str, names: Sequence[str], texts: Sequence[str]
    ) -> list[float]:
        docs = self.documents(catalog_version, names, texts)
        q = self.query(query)
        if docs and len(docs[0]) != len(q):
            raise DenseError("query and document vectors have different dimensions")
        return [sum(a * b for a, b in zip(q, d, strict=True)) for d in docs]
