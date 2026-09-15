"""Ranking metrics for a single relevant target."""

from __future__ import annotations

import math

import numpy as np


def ndcg_single(ranked: list[int], target: int, k: int) -> float:
    """NDCG@k with exactly one relevant item, so IDCG is 1."""
    for rank, item in enumerate(ranked[:k]):
        if item == target:
            return 1.0 / math.log2(rank + 2)
    return 0.0


def hit_at_k(ranked: list[int], target: int, k: int) -> float:
    return float(target in ranked[:k])


def ndcg_at_k(relevance: list[int], k: int) -> float:
    rel = relevance[:k]
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    ideal = sorted(relevance, reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(relevance: list[int], k: int) -> float:
    return float(any(relevance[:k]))


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")
