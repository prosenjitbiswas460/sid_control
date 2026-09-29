"""Per-user constraint selection for the controllability evaluation.

The central design choice: the constraint is always **compatible with the
held-out target**, i.e. the item the user actually consumed next does *not*
carry the banned attribute. A perfect control mechanism would therefore lose no
accuracy at all, and any drop in NDCG is attributable purely to collateral
damage from the control mechanism rather than to a conflict with the user's
true preference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sidctl.attributes import AttributeTable
from sidctl.data.corpus import Corpus


@dataclass
class ControlInstance:
    """One evaluation case: a history, a held-out target, and a constraint."""

    user_id: int
    history: list[dict]
    target: int
    attr: int
    attr_name: str
    reason: str  # "low_rating" or "high_exposure"

    @property
    def num_history(self) -> int:
        return len(self.history)


def _user_attr_profile(
    history: list[dict], matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-attribute interaction count and mean rating over the user's history."""
    num_attrs = matrix.shape[1]
    counts = np.zeros(num_attrs)
    totals = np.zeros(num_attrs)
    for ev in history:
        attrs = np.flatnonzero(matrix[ev["item_idx"]])
        counts[attrs] += 1
        totals[attrs] += ev["rating"]
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)
    return counts, means


def build_control_instances(
    corpus: Corpus,
    user_ids: list[int] | None = None,
    allowed_attrs: list[int] | None = None,
    min_attr_count: int = 3,
    min_history: int = 3,
    max_users: int | None = None,
    split: str = "test",
    seed: int = 42,
    attributes: AttributeTable | None = None,
) -> list[ControlInstance]:
    """
    Leave-last-positive-out instances paired with a compatible constraint.

    The attribute is chosen as one the user demonstrably dislikes (lowest mean
    rating, ``reason="low_rating"``); failing that, the attribute they are most
    exposed to (``reason="high_exposure"``), which is the realistic "I have had
    enough of this" request and the harder stress test. Vacuous constraints on
    attributes the user never touches are never selected, because they would
    make the control look free.
    """
    table = attributes if attributes is not None else corpus.attributes
    if table.num_items != corpus.num_items:
        raise ValueError(
            f"attribute table has {table.num_items} items, "
            f"corpus has {corpus.num_items}"
        )
    matrix = table.matrix
    allowed = (
        allowed_attrs
        if allowed_attrs is not None
        else table.controllable_attrs()
    )
    allowed_set = set(allowed)
    users = user_ids if user_ids is not None else sorted(corpus.user_events)

    instances: list[ControlInstance] = []
    for uid in users:
        events = corpus.user_events[uid]
        pos_idx = [i for i, e in enumerate(events) if e["polarity"] == "pos"]
        if len(pos_idx) < 3:
            continue

        # Same cut points as GRDataset, so evaluation never sees a trained target.
        cut = pos_idx[-1] if split == "test" else pos_idx[-2]
        target = events[cut]["item_idx"]
        history = events[:cut]
        if sum(1 for e in history if e["polarity"] == "pos") < min_history:
            continue

        target_attrs = set(np.flatnonzero(matrix[target]).tolist())
        candidates = [a for a in allowed_set if a not in target_attrs]
        if not candidates:
            continue

        counts, means = _user_attr_profile(history, matrix)
        user_mean = float(np.mean([e["rating"] for e in history]))

        eligible = [a for a in candidates if counts[a] >= min_attr_count]
        if not eligible:
            continue

        disliked = [a for a in eligible if means[a] < user_mean]
        if disliked:
            attr = min(disliked, key=lambda a: means[a])
            reason = "low_rating"
        else:
            attr = max(eligible, key=lambda a: counts[a])
            reason = "high_exposure"

        instances.append(
            ControlInstance(
                user_id=uid,
                history=history,
                target=target,
                attr=int(attr),
                attr_name=table.names[attr],
                reason=reason,
            )
        )

    if max_users is not None and len(instances) > max_users:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(instances), size=max_users, replace=False)
        instances = [instances[i] for i in sorted(pick.tolist())]
    return instances


def instance_stats(instances: list[ControlInstance]) -> dict:
    reasons: dict[str, int] = {}
    attrs: dict[str, int] = {}
    for inst in instances:
        reasons[inst.reason] = reasons.get(inst.reason, 0) + 1
        attrs[inst.attr_name] = attrs.get(inst.attr_name, 0) + 1
    return {
        "num_instances": len(instances),
        "by_reason": reasons,
        "top_attributes": dict(
            sorted(attrs.items(), key=lambda kv: -kv[1])[:10]
        ),
        "mean_history_len": float(
            np.mean([i.num_history for i in instances]) if instances else 0.0
        ),
    }


def pair_control_instances(
    corpus: Corpus,
    held: AttributeTable,
    *,
    min_attr_count: int = 3,
    min_history: int = 3,
    max_users: int | None = None,
    split: str = "test",
    seed: int = 42,
) -> tuple[list[ControlInstance], list[ControlInstance]]:
    """Same users, one target-compatible constraint per attribute family.

    Eligibility is computed on each table separately, then the user sets are
    intersected and subsampled together so the two columns share a user list.
    """
    kw = dict(
        min_attr_count=min_attr_count,
        min_history=min_history,
        max_users=None,
        split=split,
        seed=seed,
    )
    inn = build_control_instances(
        corpus,
        allowed_attrs=corpus.attributes.controllable_attrs(),
        **kw,
    )
    hld = build_control_instances(
        corpus,
        allowed_attrs=held.controllable_attrs(),
        attributes=held,
        **kw,
    )
    by_in = {inst.user_id: inst for inst in inn}
    by_held = {inst.user_id: inst for inst in hld}
    common = sorted(set(by_in) & set(by_held))
    if max_users is not None and len(common) > max_users:
        rng = np.random.default_rng(seed)
        pick = sorted(rng.choice(len(common), size=max_users, replace=False).tolist())
        common = [common[i] for i in pick]
    return [by_in[uid] for uid in common], [by_held[uid] for uid in common]
