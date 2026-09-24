"""Semantic ID construction, with variants that bracket prefix controllability.

Tokenizer kinds:

``rq``
    The standard content-based pipeline (TF-IDF -> SVD -> residual k-means ->
    collision suffix), i.e. a TIGER-style Semantic ID. This is the system under
    test.
``category``
    Level 0 is the item's *dominant* attribute by construction. Attribute
    constraints are prefix-realisable only up to the multi-label floor: a
    comedy-horror film occupies one prefix, so banning horror can destroy
    comedy-only neighbours that share that prefix.
``tiled``
    Proposal 2. Each item gets one control *tile* per attribute it carries.
    Level 0 of a tile *is* that attribute; deeper codes are a shared text
    residual (ranking tail). Banning horror removes the horror-channel
    prefixes; an item is excluded if *any* of its tiles is banned (AND).
    Comedy-only films keep their comedy tile, so multi-label overlap no
    longer forces catalog collateral.
``random``
    Codes assigned at random. Prefixes carry no attribute information, so this
    is the *lower bound*.

Reporting ``rq`` between ``category`` / ``tiled`` (upper bounds) and
``random`` (floor) is what turns a vague claim about coarse-to-fine
semantics into a measurement.
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

TokenizerKind = ("rq", "category", "tiled", "random")


def _tile_attribute_ids(
    attributes: AttributeTable,
    item: int,
    max_tiles: int | None,
) -> list[int | None]:
    """Attributes that receive a control tile, rarest first.

    ``None`` means the item has no attributes; it gets a single unlabeled
    tile with level-0 code 0.
    """
    attrs = attributes.attrs_of(item)
    if not attrs:
        return [None]
    prev = attributes.prevalence()
    attrs = sorted(attrs, key=lambda a: (float(prev[a]), int(a)))
    if max_tiles is not None:
        attrs = attrs[: max(1, int(max_tiles))]
    return attrs


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
        max_tiles: int | None = None,
    ):
        if kind not in TokenizerKind:
            raise ValueError(f"kind must be one of {TokenizerKind}, got {kind!r}")
        self.kind = kind
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.tfidf_dim = tfidf_dim
        self.embed_dim = embed_dim
        self.random_state = random_state
        self.max_tiles = max_tiles

        self.vectorizer: TfidfVectorizer | None = None
        self.svd: TruncatedSVD | None = None
        self.rq: RQKMeans | None = None
        self.sid_table: np.ndarray | None = None  # (N, num_levels + 1) primary SID
        # Per-item control tiles. For non-tiled kinds this stays None and
        # ``iter_sids`` falls back to the primary SID.
        self.tile_sids: list[list[tuple[int, ...]]] | None = None
        self.prefix_trie: list[dict[tuple, list[int]]] | None = None
        self.sid_to_item: dict[tuple, int] = {}
        self._prefix_items: dict[int, dict[tuple, np.ndarray]] = {}
        self._channel_prefix_items: dict[tuple[int, int], dict[tuple, np.ndarray]] = {}

    # ------------------------------------------------------------------ build

    def fit(
        self,
        texts: list[str],
        attributes: AttributeTable | None = None,
    ) -> "SIDTokenizer":
        if self.kind == "random":
            codes = self._fit_random(len(texts))
            self.sid_table = self._add_collision_codes(codes)
            self.tile_sids = None
        elif self.kind == "category":
            if attributes is None:
                raise ValueError("kind='category' requires an AttributeTable")
            codes = self._fit_category(texts, attributes)
            self.sid_table = self._add_collision_codes(codes)
            self.tile_sids = None
        elif self.kind == "tiled":
            if attributes is None:
                raise ValueError("kind='tiled' requires an AttributeTable")
            self._fit_tiled(texts, attributes)
        else:
            codes = self._fit_rq(texts)
            self.sid_table = self._add_collision_codes(codes)
            self.tile_sids = None

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

    def _fit_tiled(self, texts: list[str], attributes: AttributeTable) -> None:
        """One control tile per attribute, shared text residual as ranking tail.

        SID layout matches ``category`` (attribute code, residual codes,
        collision suffix) except a multi-label item occupies *several* SIDs
        rather than a single dominant-label prefix.
        """
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
        else:
            rest = np.zeros((len(texts), 0), dtype=np.int64)

        per_item_rows: list[list[list[int]]] = []
        for i in range(len(texts)):
            rows = []
            for attr in _tile_attribute_ids(attributes, i, self.max_tiles):
                code = 0 if attr is None else int(attr) + 1
                if rest.shape[1]:
                    rows.append([code] + [int(v) for v in rest[i]])
                else:
                    rows.append([code])
            per_item_rows.append(rows)

        flat = np.asarray(
            [row for rows in per_item_rows for row in rows], dtype=np.int64
        )
        flat = self._add_collision_codes(flat)

        tile_sids: list[list[tuple[int, ...]]] = []
        primaries = np.zeros((len(texts), flat.shape[1]), dtype=np.int64)
        cursor = 0
        for i, rows in enumerate(per_item_rows):
            tiles = []
            for _ in rows:
                sid = tuple(int(v) for v in flat[cursor])
                tiles.append(sid)
                cursor += 1
            tile_sids.append(tiles)
            primaries[i] = np.asarray(tiles[0], dtype=np.int64)

        self.tile_sids = tile_sids
        self.sid_table = primaries

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
        for item_idx, sid in self._iter_item_sids():
            for level in range(self.sid_length):
                children[level][sid[:level]].add(sid[level])
            self.sid_to_item[sid] = item_idx
        self.prefix_trie = [
            {p: sorted(tok) for p, tok in level.items()} for level in children
        ]
        self._prefix_items = {}
        self._channel_prefix_items = {}

    def _iter_item_sids(self):
        """Yield ``(item_idx, sid)`` for every control path, including tiles."""
        assert self.sid_table is not None
        if self.tile_sids is not None:
            for item_idx, tiles in enumerate(self.tile_sids):
                for sid in tiles:
                    yield item_idx, tuple(int(v) for v in sid)
            return
        for item_idx, row in enumerate(self.sid_table):
            yield item_idx, tuple(int(v) for v in row)

    def iter_sids(self, item_idx: int) -> list[tuple[int, ...]]:
        """All SIDs that decode to ``item_idx`` (one for non-tiled kinds)."""
        if self.tile_sids is not None:
            return [tuple(int(v) for v in sid) for sid in self.tile_sids[item_idx]]
        return [self.get_sid(item_idx)]

    @property
    def is_tiled(self) -> bool:
        return self.kind == "tiled" and self.tile_sids is not None

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
        if self.tile_sids:
            rows = np.asarray(
                [sid for tiles in self.tile_sids for sid in tiles], dtype=np.int64
            )
        else:
            rows = self.sid_table
        return [int(rows[:, l].max()) + 1 for l in range(self.sid_length)]

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
        for item_idx, sid in self._iter_item_sids():
            groups[sid[:level]].append(item_idx)
        # An item can sit under several prefixes when tiled; keep unique
        # indices per prefix so catalog counts are not inflated.
        out = {
            p: np.asarray(sorted(set(v)), dtype=np.int64) for p, v in groups.items()
        }
        self._prefix_items[level] = out
        return out

    def channel_prefix_items(self, attr: int, level: int) -> dict[tuple, np.ndarray]:
        """Prefixes in the control channel for ``attr``.

        For a tiled tokenizer the channel is every tile whose first code is
        ``attr + 1``. Non-tiled kinds fall back to ``prefix_items``.
        """
        if not self.is_tiled:
            return self.prefix_items(level)
        key = (attr, level)
        if key in self._channel_prefix_items:
            return self._channel_prefix_items[key]
        code = int(attr) + 1
        groups: dict[tuple, list[int]] = defaultdict(list)
        assert self.tile_sids is not None
        for item_idx, tiles in enumerate(self.tile_sids):
            for sid in tiles:
                if sid[0] != code:
                    continue
                groups[sid[:level]].append(item_idx)
        out = {
            p: np.asarray(sorted(set(v)), dtype=np.int64) for p, v in groups.items()
        }
        self._channel_prefix_items[key] = out
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
        n_tiles = (
            sum(len(t) for t in self.tile_sids) if self.tile_sids is not None else self.num_items
        )
        return {
            "kind": self.kind,
            "num_items": self.num_items,
            "sid_length": self.sid_length,
            "level_sizes": sizes,
            "colliding_items": collisions,
            "collision_rate": round(collisions / max(self.num_items, 1), 4),
            "n_tile_sids": int(n_tiles),
            "mean_tiles_per_item": round(n_tiles / max(self.num_items, 1), 4),
            "max_tiles": self.max_tiles,
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
                "max_tiles": self.max_tiles,
            },
            "sid_table": self.sid_table,
            "tile_sids": self.tile_sids,
            "prefix_trie": self.prefix_trie,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str | Path) -> "SIDTokenizer":
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        cfg = dict(payload["config"])
        cfg.setdefault("max_tiles", None)
        tok = cls(**cfg)
        tok.sid_table = payload["sid_table"]
        tok.tile_sids = payload.get("tile_sids")
        if tok.tile_sids is not None:
            tok.tile_sids = [
                [tuple(int(v) for v in sid) for sid in tiles]
                for tiles in tok.tile_sids
            ]
        tok.prefix_trie = payload["prefix_trie"]
        tok.sid_to_item = {}
        if tok.sid_table is not None:
            for item_idx, sid in tok._iter_item_sids():
                tok.sid_to_item[sid] = item_idx
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
    max_tiles: int | None = None,
) -> SIDTokenizer:
    return SIDTokenizer(
        kind=kind,
        num_levels=num_levels,
        codebook_size=codebook_size,
        tfidf_dim=tfidf_dim,
        embed_dim=embed_dim,
        random_state=random_state,
        max_tiles=max_tiles,
    ).fit(texts, attributes=attributes)
