"""Semantic ID construction, with variants that bracket prefix controllability.

Three tokenizer kinds are provided on purpose:

``rq``
    The standard content-based pipeline (TF-IDF -> SVD -> residual k-means ->
    collision suffix), i.e. a TIGER-style Semantic ID. This is the system under
    test.
``category``
    Level 0 is the item's attribute by construction. Attribute constraints are
    exactly prefix-realisable here, so it is the *upper bound* on how well
    prefix masking could ever work.
``random``
    Codes assigned at random. Prefixes carry no attribute information, so this
    is the *lower bound*.

Reporting ``rq`` between these two bounds is what turns a vague claim about
"coarse-to-fine semantics" into a measurement.
"""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from sidctl.attributes import AttributeTable
from sidctl.sid.rq_kmeans import RQKMeans

TokenizerKind = ("rq", "category", "random")


def _embed_texts(
    texts: list[str],
    tfidf_dim: int,
    embed_dim: int,
    random_state: int,
) -> tuple[np.ndarray, TfidfVectorizer, TruncatedSVD]:
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    tfidf = vectorizer.fit_transform(texts)
    svd_dim = max(2, min(tfidf_dim, tfidf.shape[1] - 1, tfidf.shape[0] - 1))
    svd = TruncatedSVD(n_components=svd_dim, random_state=random_state)
    reduced = svd.fit_transform(tfidf)

    if reduced.shape[1] > embed_dim:
        rng = np.random.default_rng(random_state)
        proj = rng.standard_normal((reduced.shape[1], embed_dim))
        proj /= np.linalg.norm(proj, axis=0, keepdims=True) + 1e-8
        embeddings = reduced @ proj
    else:
        embeddings = np.pad(reduced, ((0, 0), (0, embed_dim - reduced.shape[1])))
    return embeddings, vectorizer, svd


class SIDTokenizer:
    """Item index -> hierarchical Semantic ID of length ``num_levels + 1``."""

    def __init__(
        self,
        kind: str = "rq",
        num_levels: int = 3,
        codebook_size: int = 256,
        tfidf_dim: int = 128,
        embed_dim: int = 64,
        random_state: int = 42,
    ):
        if kind not in TokenizerKind:
            raise ValueError(f"kind must be one of {TokenizerKind}, got {kind!r}")
        self.kind = kind
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.tfidf_dim = tfidf_dim
        self.embed_dim = embed_dim
        self.random_state = random_state

        self.vectorizer: TfidfVectorizer | None = None
        self.svd: TruncatedSVD | None = None
        self.rq: RQKMeans | None = None
        self.sid_table: np.ndarray | None = None  # (N, num_levels + 1)
        self.prefix_trie: list[dict[tuple, list[int]]] | None = None
        self.sid_to_item: dict[tuple, int] = {}
        self._prefix_items: dict[int, dict[tuple, np.ndarray]] = {}

    # ------------------------------------------------------------------ build

    def fit(
        self,
        texts: list[str],
        attributes: AttributeTable | None = None,
    ) -> "SIDTokenizer":
        if self.kind == "random":
            codes = self._fit_random(len(texts))
        elif self.kind == "category":
            if attributes is None:
                raise ValueError("kind='category' requires an AttributeTable")
            codes = self._fit_category(texts, attributes)
        else:
            codes = self._fit_rq(texts)

        self.sid_table = self._add_collision_codes(codes)
        self._build_trie()
        return self

    def _fit_rq(self, texts: list[str]) -> np.ndarray:
        embeddings, self.vectorizer, self.svd = _embed_texts(
            texts, self.tfidf_dim, self.embed_dim, self.random_state
        )
        self.rq = RQKMeans(
            num_levels=self.num_levels,
            codebook_size=min(self.codebook_size, len(texts)),
            random_state=self.random_state,
        )
        self.rq.fit(embeddings)
        return self.rq.encode(embeddings)

    def _fit_category(self, texts: list[str], attributes: AttributeTable) -> np.ndarray:
        """Level 0 = dominant attribute; deeper levels = residual quantization."""
        labels = attributes.dominant_labels()
        level0 = labels + 1  # shift so "no attribute" becomes code 0

        embeddings, self.vectorizer, self.svd = _embed_texts(
            texts, self.tfidf_dim, self.embed_dim, self.random_state
        )
        deeper = max(self.num_levels - 1, 0)
        if deeper:
            self.rq = RQKMeans(
                num_levels=deeper,
                codebook_size=min(self.codebook_size, len(texts)),
                random_state=self.random_state,
            )
            self.rq.fit(embeddings)
            rest = self.rq.encode(embeddings)
            return np.concatenate([level0[:, None], rest], axis=1)
        return level0[:, None]

    def _fit_random(self, num_items: int) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        size = min(self.codebook_size, max(num_items, 2))
        return rng.integers(0, size, size=(num_items, self.num_levels))

    @staticmethod
    def _add_collision_codes(codes: np.ndarray) -> np.ndarray:
        """Final token disambiguates items that collide on all semantic codes."""
        counters: dict[tuple, int] = defaultdict(int)
        dedup = np.zeros(len(codes), dtype=np.int64)
        for i, row in enumerate(map(tuple, codes.tolist())):
            dedup[i] = counters[row]
            counters[row] += 1
        return np.concatenate([codes, dedup[:, None]], axis=1)

    def _build_trie(self) -> None:
        assert self.sid_table is not None
        children: list[dict[tuple, set[int]]] = [
            defaultdict(set) for _ in range(self.sid_length)
        ]
        self.sid_to_item = {}
        for item_idx, row in enumerate(self.sid_table):
            sid = tuple(int(v) for v in row)
            for level in range(self.sid_length):
                children[level][sid[:level]].add(sid[level])
            self.sid_to_item[sid] = item_idx
        self.prefix_trie = [
            {p: sorted(tok) for p, tok in level.items()} for level in children
        ]
        self._prefix_items = {}

    # ----------------------------------------------------------------- access

    @property
    def sid_length(self) -> int:
        if self.sid_table is not None:
            return self.sid_table.shape[1]
        return self.num_levels + 1

    @property
    def num_items(self) -> int:
        assert self.sid_table is not None
        return self.sid_table.shape[0]

    @property
    def level_sizes(self) -> list[int]:
        """Per-position code counts, derived from the realised table."""
        assert self.sid_table is not None
        return [int(self.sid_table[:, l].max()) + 1 for l in range(self.sid_length)]

    def get_sid(self, item_idx: int) -> tuple[int, ...]:
        assert self.sid_table is not None
        return tuple(int(v) for v in self.sid_table[item_idx])

    def valid_tokens_at_level(self, prefix: tuple[int, ...]) -> list[int]:
        assert self.prefix_trie is not None
        level = len(prefix)
        if level >= self.sid_length:
            return []
        return self.prefix_trie[level].get(prefix, [])

    def prefix_items(self, level: int) -> dict[tuple, np.ndarray]:
        """
        Map each length-``level`` prefix to the item indices beneath it.

        ``level`` counts semantic positions, so ``level=1`` groups items by
        their first code. Cached, since the analysis sweeps over it repeatedly.
        """
        if level in self._prefix_items:
            return self._prefix_items[level]
        assert self.sid_table is not None
        if not 1 <= level <= self.sid_length:
            raise ValueError(f"level must be in [1, {self.sid_length}], got {level}")

        groups: dict[tuple, list[int]] = defaultdict(list)
        for item_idx, row in enumerate(self.sid_table):
            groups[tuple(int(v) for v in row[:level])].append(item_idx)
        out = {p: np.asarray(v, dtype=np.int64) for p, v in groups.items()}
        self._prefix_items[level] = out
        return out

    def item_prefixes(self, level: int) -> np.ndarray:
        """Array of shape (num_items, level) with each item's prefix."""
        assert self.sid_table is not None
        return self.sid_table[:, :level]

    def stats(self) -> dict:
        assert self.sid_table is not None
        sizes = self.level_sizes
        collisions = int((self.sid_table[:, -1] > 0).sum())
        fanout = {
            f"level{l}_num_prefixes": len(self.prefix_items(l))
            for l in range(1, self.sid_length)
        }
        return {
            "kind": self.kind,
            "num_items": self.num_items,
            "sid_length": self.sid_length,
            "level_sizes": sizes,
            "colliding_items": collisions,
            "collision_rate": round(collisions / max(self.num_items, 1), 4),
            **fanout,
        }

    # -------------------------------------------------------------- serialize

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "kind": self.kind,
                "num_levels": self.num_levels,
                "codebook_size": self.codebook_size,
                "tfidf_dim": self.tfidf_dim,
                "embed_dim": self.embed_dim,
                "random_state": self.random_state,
            },
            "sid_table": self.sid_table,
            "prefix_trie": self.prefix_trie,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str | Path) -> "SIDTokenizer":
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        tok = cls(**payload["config"])
        tok.sid_table = payload["sid_table"]
        tok.prefix_trie = payload["prefix_trie"]
        tok.sid_to_item = {
            tuple(int(v) for v in row): i for i, row in enumerate(tok.sid_table)
        }
        return tok

    def export_json(self, path: str | Path) -> None:
        assert self.sid_table is not None
        Path(path).write_text(
            json.dumps(
                {
                    "stats": self.stats(),
                    "sids": {i: row.tolist() for i, row in enumerate(self.sid_table)},
                },
                indent=2,
            )
        )


def build_tokenizer(
    kind: str,
    texts: list[str],
    attributes: AttributeTable | None = None,
    num_levels: int = 3,
    codebook_size: int = 256,
    tfidf_dim: int = 128,
    embed_dim: int = 64,
    random_state: int = 42,
) -> SIDTokenizer:
    return SIDTokenizer(
        kind=kind,
        num_levels=num_levels,
        codebook_size=codebook_size,
        tfidf_dim=tfidf_dim,
        embed_dim=embed_dim,
        random_state=random_state,
    ).fit(texts, attributes=attributes)
