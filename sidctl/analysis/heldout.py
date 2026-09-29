"""In-family vs held-out constraint measurements on a frozen SID table."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sidctl.analysis.policies import evaluate_tokenizer_policies
from sidctl.analysis.pruning import analyze_pruning
from sidctl.analysis.purity import prefix_purity, realizability_frontier
from sidctl.attributes import AttributeTable
from sidctl.sid.tokenizer import SIDTokenizer


def _mean(xs: list[float]) -> float:
    vals = [x for x in xs if np.isfinite(x)]
    return float(np.mean(vals)) if vals else float("nan")


def measure_family(
    tokenizer: SIDTokenizer,
    attributes: AttributeTable,
    level: int = 1,
    ignore_tile_channels: bool = False,
) -> dict:
    """Collateral, Pmaj, depth-1 settlement, visits — one attribute family.

    ``ignore_tile_channels`` is for held-out families: tile L0 codes index the
    *in-family* attribute, not brand/decade ids, so AND-channel masks would
    be a spurious integer alignment. The SID tree is still the one that was
    built (all tiles); only the control-channel shortcut is skipped.
    """
    prev = getattr(tokenizer, "_ignore_tile_channels", False)
    tokenizer._ignore_tile_channels = bool(ignore_tile_channels)
    try:
        return _measure_family(tokenizer, attributes, level)
    finally:
        tokenizer._ignore_tile_channels = prev


def _measure_family(
    tokenizer: SIDTokenizer,
    attributes: AttributeTable,
    level: int,
) -> dict:
    attrs = attributes.controllable_attrs()
    if not attrs:
        return {
            "n_attrs": 0,
            "mean_collateral_l1": float("nan"),
            "pmaj_leakage": float("nan"),
            "pmaj_coverage": float("nan"),
            "purity_l1": float("nan"),
            "settled_at_depth_1": float("nan"),
            "node_visits": float("nan"),
        }
    coll = [
        realizability_frontier(tokenizer, attributes, a, level).collateral_at(0.0)
        for a in attrs
    ]
    policies = evaluate_tokenizer_policies(
        tokenizer, attributes, attrs=attrs, levels=[level]
    )
    pmaj = policies["by_level"][str(level)]["aggregates"]["Pmaj"]
    prune = analyze_pruning(tokenizer, attributes, attrs=attrs, thetas=[0.0])
    exact = prune["aggregate_by_theta"][0]
    return {
        "n_attrs": len(attrs),
        "attr_names": [attributes.names[a] for a in attrs],
        "mean_collateral_l1": _mean(coll),
        "pmaj_leakage": pmaj["leakage"],
        "pmaj_coverage": pmaj["coverage"],
        "purity_l1": prefix_purity(tokenizer, attributes, level)["purity"],
        "settled_at_depth_1": exact["settled_at_depth_1"],
        "node_visits": exact["node_visits"],
        "rows_pruned": exact["rows_pruned"],
        "feasible_recall": exact["feasible_recall"],
        "policies": policies,
        "pruning": prune,
    }


def heldout_table(runs: dict[str, dict]) -> list[str]:
    """Markdown: in-family vs held-out, one row per tokenizer."""
    lines = [
        "| tokenizer | in coll@L1 | held coll@L1 | in settled@d1 | "
        "held settled@d1 | in visits | held visits |",
        "|---|---|---|---|---|---|---|",
    ]
    for tag, pair in runs.items():
        inn, hld = pair["in_family"], pair["held_out"]
        lines.append(
            f"| {tag} | {inn['mean_collateral_l1']:.3f} | "
            f"{hld['mean_collateral_l1']:.3f} | "
            f"{inn['settled_at_depth_1']:.3f} | "
            f"{hld['settled_at_depth_1']:.3f} | "
            f"{inn['node_visits']:.0f} | {hld['node_visits']:.0f} |"
        )
    return lines + [""]


def fig_heldout(runs: dict[str, dict], path: Path, dataset: str = "") -> None:
    """Grouped bars: depth-1 settlement, in-family vs held-out."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tags = list(runs)
    x = np.arange(len(tags))
    w = 0.36
    inn = [runs[t]["in_family"]["settled_at_depth_1"] for t in tags]
    hld = [runs[t]["held_out"]["settled_at_depth_1"] for t in tags]
    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(tags)), 4.2))
    ax.bar(x - w / 2, inn, w, label="in-family (SID cut)")
    ax.bar(x + w / 2, hld, w, label="held-out family")
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("catalog settled at depth 1")
    ax.set_title(f"same SIDs, two constraints{': ' + dataset if dataset else ''}")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
