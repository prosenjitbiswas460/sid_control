"""Constraint-aware SID pruning: how much catalog can a coarse decision settle?

Part A asks whether a constraint *can* be written as banned prefixes. This
module asks a systems question instead: taking the constraint as exact, how
much of the identifier space can be settled by a decision on a whole subtree,
rather than item by item?

Descend the SID trie from the root. At a node ``v`` with ``A(v)`` allowed items
underneath:

PRUNE
    ``|A(v)| == 0``. Nothing feasible can be reached through ``v``, so the
    whole subtree goes with one decision and zero recall loss.
ACCEPT
    every item under ``v`` is allowed. The subtree needs no constraint work at
    all.
DESCEND
    ``v`` is mixed: the constraint does not align with this subtree, so the
    decision has to be pushed to the children.

Work is the number of nodes visited. The baseline is the item-by-item scan that
builds an allowed set directly, so ``work_ratio = visits / rows``. A tokenizer
whose subtrees are constraint-coherent settles the catalog near the root and
gets a small ratio; a tokenizer whose attributes are scattered has to descend
to the leaves and saves nothing.

Relaxing PRUNE to ``allowed share <= theta`` buys more pruning at the cost of
feasible items, which is the risk/compute curve: ``theta = 0`` is exact and is
precisely the allowed-item trie.

Everything here is catalog-side and model-free, so it runs on CPU and scales
with the number of distinct prefixes rather than the number of users.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from sidctl.attributes import AttributeTable
from sidctl.sid.tokenizer import SIDTokenizer

DEFAULT_THETAS = [0.0, 1e-4, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

PRUNE, ACCEPT, DESCEND = "prune", "accept", "descend"


@dataclass
class PruneResult:
    """Outcome of one constraint resolution over the SID trie."""

    attr: int
    attr_name: str
    theta: float
    num_items: int
    num_rows: int  # SID rows; > num_items for tiled identifiers
    num_allowed_items: int
    node_visits: int = 0
    surviving_rows: np.ndarray | None = None
    # per depth: decisions taken and SID rows settled by each decision
    by_depth: list[dict] = field(default_factory=list)
    feasible_recall: float = 1.0
    lost_items: int = 0
    tokenizer_depth: int = 0

    @property
    def work_ratio(self) -> float:
        """Node visits per SID row. Can exceed 1 when every leaf is visited."""
        return float(self.node_visits / max(self.num_rows, 1))

    @property
    def work_saved(self) -> float:
        """Fraction of SID rows whose fate was decided above the leaf.

        That is the systems quantity: one subtree decision instead of one
        check per item. Zero means the constraint is scattered to the leaves
        and the tree saved no item-level work; one means the whole catalog
        was settled by an internal node.
        """
        leaf = self.tokenizer_depth
        settled = sum(
            d["rows_prune"] + d["rows_accept"]
            for d in self.by_depth
            if d["depth"] < leaf
        )
        return float(settled / max(self.num_rows, 1))

    def settled_by(self, decision: str, max_depth: int | None = None) -> float:
        """Fraction of SID rows settled by ``decision`` at or above ``max_depth``."""
        rows = sum(
            d[f"rows_{decision}"]
            for d in self.by_depth
            if max_depth is None or d["depth"] <= max_depth
        )
        return float(rows / max(self.num_rows, 1))

    @property
    def settled_at_depth_1(self) -> float:
        return self.settled_by("prune", 1) + self.settled_by("accept", 1)

    def summary(self) -> dict:
        return {
            "attr": self.attr,
            "attr_name": self.attr_name,
            "theta": self.theta,
            "num_items": self.num_items,
            "num_rows": self.num_rows,
            "prevalence": float(
                1.0 - self.num_allowed_items / max(self.num_items, 1)
            ),
            "node_visits": self.node_visits,
            "work_ratio": self.work_ratio,
            "work_saved": self.work_saved,
            "rows_pruned": self.settled_by("prune"),
            "rows_accepted": self.settled_by("accept"),
            "settled_at_depth_1": self.settled_by("prune", 1)
            + self.settled_by("accept", 1),
            "feasible_recall": self.feasible_recall,
            "lost_items": self.lost_items,
            "by_depth": self.by_depth,
        }


def _rows(tokenizer: SIDTokenizer) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    items, sids = [], []
    for item_idx, sid in tokenizer._iter_item_sids():
        items.append(item_idx)
        sids.append(sid)
    return np.asarray(items, dtype=np.int64), sids


def resolve_constraint(
    tokenizer: SIDTokenizer,
    attributes: AttributeTable,
    attr: int,
    theta: float = 0.0,
) -> PruneResult:
    """Resolve "exclude ``attr``" by descending the SID trie.

    Forbidden items are never reachable through the surviving rows, whatever
    ``theta`` is: a leaf is a single SID, so the last decision is always exact.
    ``theta`` only controls how many *allowed* items get pruned along with a
    mostly-forbidden subtree.
    """
    item_of_row, sids = _rows(tokenizer)
    is_attr = attributes.matrix[:, attr].astype(bool)
    allowed_row = ~is_attr[item_of_row]
    depth_max = tokenizer.sid_length

    result = PruneResult(
        attr=attr,
        attr_name=attributes.names[attr],
        theta=float(theta),
        num_items=int(attributes.num_items),
        num_rows=len(sids),
        num_allowed_items=int((~is_attr).sum()),
        tokenizer_depth=depth_max,
    )

    survivors: list[int] = []
    tally = {
        d: {"depth": d, "prune": 0, "accept": 0, "descend": 0,
            "rows_prune": 0, "rows_accept": 0, "rows_descend": 0}
        for d in range(depth_max + 1)
    }

    # (depth, rows) frontier; depth is the length of the shared prefix.
    frontier: list[tuple[int, np.ndarray]] = [
        (0, np.arange(len(sids), dtype=np.int64))
    ]
    while frontier:
        depth, rows = frontier.pop()
        result.node_visits += 1
        n_allowed = int(allowed_row[rows].sum())
        share = n_allowed / len(rows)

        if share <= theta:
            decision = PRUNE
        elif n_allowed == len(rows):
            decision = ACCEPT
        elif depth >= depth_max:
            # Unreachable for unique SIDs; a collision would land here.
            decision = PRUNE if n_allowed == 0 else ACCEPT
        else:
            decision = DESCEND

        tally[depth][decision] += 1
        tally[depth][f"rows_{decision}"] += len(rows)

        if decision == ACCEPT:
            survivors.extend(rows[allowed_row[rows]].tolist())
        elif decision == DESCEND:
            children: dict[int, list[int]] = defaultdict(list)
            for r in rows.tolist():
                children[sids[r][depth]].append(r)
            for kids in children.values():
                frontier.append((depth + 1, np.asarray(kids, dtype=np.int64)))

    result.surviving_rows = np.asarray(sorted(survivors), dtype=np.int64)
    result.by_depth = [tally[d] for d in range(depth_max + 1) if
                       tally[d]["prune"] or tally[d]["accept"] or tally[d]["descend"]]

    reachable = np.zeros(attributes.num_items, dtype=bool)
    reachable[item_of_row[result.surviving_rows]] = True
    allowed_items = ~is_attr
    kept = int((reachable & allowed_items).sum())
    result.feasible_recall = float(kept / max(result.num_allowed_items, 1))
    result.lost_items = int(result.num_allowed_items - kept)
    return result


def prune_curve(
    tokenizer: SIDTokenizer,
    attributes: AttributeTable,
    attr: int,
    thetas: list[float] | None = None,
) -> list[dict]:
    """Risk/compute curve: one (work_saved, feasible_recall) point per theta."""
    return [
        resolve_constraint(tokenizer, attributes, attr, theta).summary()
        for theta in (thetas or DEFAULT_THETAS)
    ]


def analyze_pruning(
    tokenizer: SIDTokenizer,
    attributes: AttributeTable,
    attrs: list[int] | None = None,
    thetas: list[float] | None = None,
) -> dict:
    """Pruning analysis for one tokenizer, averaged over attributes."""
    attrs = attrs if attrs is not None else attributes.controllable_attrs()
    thetas = thetas or DEFAULT_THETAS

    per_attr = {
        attributes.names[a]: prune_curve(tokenizer, attributes, a, thetas)
        for a in attrs
    }

    by_theta = []
    for i, theta in enumerate(thetas):
        rows = [curve[i] for curve in per_attr.values()]
        by_theta.append(
            {
                "theta": theta,
                "work_saved": float(np.mean([r["work_saved"] for r in rows])),
                "work_ratio": float(np.mean([r["work_ratio"] for r in rows])),
                "node_visits": float(np.mean([r["node_visits"] for r in rows])),
                "rows_pruned": float(np.mean([r["rows_pruned"] for r in rows])),
                "rows_accepted": float(np.mean([r["rows_accepted"] for r in rows])),
                "settled_at_depth_1": float(
                    np.mean([r["settled_at_depth_1"] for r in rows])
                ),
                "feasible_recall": float(
                    np.mean([r["feasible_recall"] for r in rows])
                ),
                "worst_recall": float(np.min([r["feasible_recall"] for r in rows])),
            }
        )

    return {
        "tokenizer": tokenizer.stats(),
        "attrs_tested": [attributes.names[a] for a in attrs],
        "thetas": thetas,
        "aggregate_by_theta": by_theta,
        "per_attr": per_attr,
    }


def fig_pruning(runs: dict[str, dict], path, dataset: str = "") -> None:
    """Recall against computation saved, one line per tokenizer."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_c, ax_d) = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, run in runs.items():
        pts = run["aggregate_by_theta"]
        ax_c.plot(
            [p["work_saved"] for p in pts],
            [p["feasible_recall"] for p in pts],
            marker="o",
            label=name,
        )
        exact = pts[0]
        ax_d.bar(
            name,
            exact["settled_at_depth_1"],
            label=None,
        )
    ax_c.set_xlabel("catalog settled above the leaf")
    ax_c.set_ylabel("feasible-item recall")
    ax_c.set_title(f"risk/compute frontier{': ' + dataset if dataset else ''}")
    ax_c.legend(fontsize=8)
    ax_d.set_ylabel("catalog settled by a depth-1 decision")
    ax_d.set_ylim(0, 1)
    ax_d.set_title("exact pruning (theta = 0)")
    ax_d.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pruning_table(runs: dict[str, dict]) -> list[str]:
    """Markdown: exact pruning per tokenizer, then the theta sweep."""
    lines = [
        "| tokenizer | node visits | work saved | settled @depth1 | "
        "rows pruned | rows accepted |",
        "|---|---|---|---|---|---|",
    ]
    for name, run in runs.items():
        e = run["aggregate_by_theta"][0]
        lines.append(
            f"| {name} | {e['node_visits']:.0f} | {e['work_saved']:.3f} | "
            f"{e['settled_at_depth_1']:.3f} | {e['rows_pruned']:.3f} | "
            f"{e['rows_accepted']:.3f} |"
        )
    lines += ["", "| tokenizer | theta | work saved | recall | worst-attr recall |",
              "|---|---|---|---|---|"]
    for name, run in runs.items():
        for p in run["aggregate_by_theta"]:
            lines.append(
                f"| {name} | {p['theta']:g} | {p['work_saved']:.3f} | "
                f"{p['feasible_recall']:.4f} | {p['worst_recall']:.4f} |"
            )
    return lines + [""]
