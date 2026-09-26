"""Catalog-level control policies on a frozen SID table.

P0, Pτ, and Pmaj change *which prefixes are banned*, not the integer labels on
the codes. Permuting L1 token IDs leaves every metric in this module unchanged.

P0
    Zero-leak union: ban prefix ``p`` iff it contains at least one forbidden
    item. This is ``collateral@0leak``.
Pτ
    Ban ``p`` iff the share of forbidden items in ``p`` is at least ``τ``.
    Sweeping ``τ`` traces the leak–collateral curve of the same codebook.
Pmaj
    Assign each prefix its unique majority attribute (argmax occupancy; ties
    get no label). Ban prefixes whose majority label is the forbidden
    attribute.

The cheap experiment is P0 vs Pτ vs Pmaj on frozen Beauty L1 prefixes.
If Pmaj already sits near the category bound, the partition is fine and only
the interface was wrong. If P0 is bad *and* Pmaj still has large collateral
at low leak, the clusters themselves are mixed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sidctl.analysis.purity import DEFAULT_THRESHOLDS, _prefix_index, realizability_frontier
from sidctl.attributes import AttributeTable
from sidctl.control.masks import build_majority_mask, mask_effect
from sidctl.sid.tokenizer import SIDTokenizer

HEADLINE_POLICIES = ("P0", "Pmaj", "Ptau@0.5")


def _with_coverage(point: dict) -> dict:
    out = dict(point)
    leak = out.get("leakage")
    out["coverage"] = (
        float(1.0 - leak) if leak is not None and np.isfinite(leak) else float("nan")
    )
    return out


def _mean_finite(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if key in r and np.isfinite(r[key])]
    return float(np.mean(vals)) if vals else float("nan")


def majority_assignment_stats(
    tokenizer: SIDTokenizer,
    attributes: AttributeTable,
    level: int,
) -> dict:
    """How often an L-depth prefix has a unique majority attribute."""
    if tokenizer.is_tiled:
        return {
            "control_semantics": "tiled_and",
            "note": "majority is computed per control channel, not on the mixed tree",
        }

    ids, n_prefixes = _prefix_index(tokenizer, level)
    counts = np.bincount(ids, minlength=n_prefixes).astype(np.float64)
    occupancy = np.zeros((n_prefixes, attributes.num_attrs), dtype=np.float64)
    for a in range(attributes.num_attrs):
        occupancy[:, a] = np.bincount(
            ids,
            weights=attributes.matrix[:, a].astype(np.float64),
            minlength=n_prefixes,
        )
    peak = occupancy.max(axis=1)
    n_winners = (occupancy == peak[:, None]).sum(axis=1)
    unique = (n_winners == 1) & (peak > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(counts > 0, peak / counts, 0.0)

    nonempty = counts > 0
    return {
        "control_semantics": "single_sid",
        "num_prefixes": int(n_prefixes),
        "unique_majority_rate": float(unique.mean()) if n_prefixes else float("nan"),
        "mean_majority_share": (
            float(share[nonempty].mean()) if nonempty.any() else float("nan")
        ),
        "mean_majority_share_unique": (
            float(share[unique].mean()) if unique.any() else float("nan")
        ),
    }


def evaluate_attr_policies(
    tokenizer: SIDTokenizer,
    attributes: AttributeTable,
    attr: int,
    level: int,
    thresholds: list[float] | None = None,
) -> dict:
    """P0 / Pτ sweep / Pmaj for one attribute at one prefix depth."""
    thresholds = list(thresholds or DEFAULT_THRESHOLDS)
    if 0.0 not in thresholds:
        thresholds = [0.0] + thresholds
    if 0.5 not in thresholds:
        thresholds = sorted(thresholds + [0.5])

    frontier = realizability_frontier(
        tokenizer, attributes, attr, level, thresholds=thresholds
    )
    ptau = [_with_coverage(p) for p in frontier.points]
    by_tau = {float(p["tau"]): p for p in ptau}

    pmaj = _with_coverage(
        mask_effect(
            tokenizer,
            attributes,
            attr,
            build_majority_mask(tokenizer, attributes, attr, level),
            level,
        )
    )
    pmaj["policy"] = "Pmaj"

    p0 = dict(by_tau[0.0])
    p0["policy"] = "P0"
    ptau_05 = dict(by_tau[0.5])
    ptau_05["policy"] = "Ptau@0.5"

    return {
        "attr": attr,
        "attr_name": attributes.names[attr],
        "level": level,
        "prevalence": frontier.prevalence,
        "p0": p0,
        "pmaj": pmaj,
        "ptau_05": ptau_05,
        "ptau": ptau,
    }


def _aggregate_policy(per_attr: list[dict], key: str) -> dict:
    rows = [row[key] for row in per_attr]
    return {
        "leakage": _mean_finite(rows, "leakage"),
        "collateral": _mean_finite(rows, "collateral"),
        "coverage": _mean_finite(rows, "coverage"),
        "catalog_retained": _mean_finite(rows, "catalog_retained"),
        "prefix_ban_rate": _mean_finite(rows, "prefix_ban_rate"),
        "banned_prefixes": _mean_finite(rows, "banned_prefixes"),
        "n_attrs": len(rows),
    }


def evaluate_tokenizer_policies(
    tokenizer: SIDTokenizer,
    attributes: AttributeTable,
    attrs: list[int] | None = None,
    levels: list[int] | None = None,
    thresholds: list[float] | None = None,
) -> dict:
    """Full P0 / Pτ / Pmaj report for one frozen tokenizer."""
    attrs = attrs if attrs is not None else attributes.controllable_attrs()
    levels = levels or [1]
    by_level: dict[str, dict] = {}
    for level in levels:
        if not 1 <= level < tokenizer.sid_length:
            continue
        per_attr = [
            evaluate_attr_policies(
                tokenizer, attributes, a, level, thresholds=thresholds
            )
            for a in attrs
        ]
        by_level[str(level)] = {
            "majority_assignment": majority_assignment_stats(
                tokenizer, attributes, level
            ),
            "aggregates": {
                "P0": _aggregate_policy(per_attr, "p0"),
                "Pmaj": _aggregate_policy(per_attr, "pmaj"),
                "Ptau@0.5": _aggregate_policy(per_attr, "ptau_05"),
            },
            "per_attr": per_attr,
        }
    return {
        "tokenizer": tokenizer.stats(),
        "attrs_tested": [attributes.names[a] for a in attrs],
        "by_level": by_level,
    }


def table_rows(tag: str, report: dict) -> list[dict]:
    """Flatten mean leak / coll / coverage for the headline policies."""
    rows = []
    for level_s, block in report["by_level"].items():
        assignment = block.get("majority_assignment", {})
        for policy in HEADLINE_POLICIES:
            agg = block["aggregates"][policy]
            rows.append(
                {
                    "tokenizer": tag,
                    "level": int(level_s),
                    "policy": policy,
                    "leakage": round(agg["leakage"], 4),
                    "collateral": round(agg["collateral"], 4),
                    "coverage": round(agg["coverage"], 4),
                    "retained": round(agg["catalog_retained"], 4),
                    "uniq_maj": (
                        round(assignment["unique_majority_rate"], 4)
                        if "unique_majority_rate" in assignment
                        else "n/a"
                    ),
                }
            )
    return rows


def _pick_tag(
    tags: list[str],
    preferred: tuple[str, ...],
    fallback_prefix: str,
) -> str | None:
    for name in preferred:
        if name in tags:
            return name
    for tag in tags:
        if tag.startswith(fallback_prefix):
            return tag
    return None


def diagnose_l1(reports: dict[str, dict]) -> str:
    """One-paragraph reading of the Beauty-style L1 falsification test."""
    tags = [t for t, r in reports.items() if "1" in r.get("by_level", {})]
    if not tags:
        return "no L1 results"

    rq = _pick_tag(tags, ("rq_title", "rq_title_side"), "rq_")
    cat = _pick_tag(tags, ("category_title", "category_title_side"), "category_")
    rnd = _pick_tag(tags, ("random_title", "random_title_side"), "random_")
    if rq is None:
        return "no rq tokenizer at L1"

    def agg(tag: str, policy: str) -> dict:
        return reports[tag]["by_level"]["1"]["aggregates"][policy]

    rq_p0 = agg(rq, "P0")
    rq_maj = agg(rq, "Pmaj")
    rq_t = agg(rq, "Ptau@0.5")
    lines = [
        f"L1 diagnosis ({rq}):",
        f"  P0       leak={rq_p0['leakage']:.4f}  coll={rq_p0['collateral']:.4f}  "
        f"coverage={rq_p0['coverage']:.4f}",
        f"  Pmaj     leak={rq_maj['leakage']:.4f}  coll={rq_maj['collateral']:.4f}  "
        f"coverage={rq_maj['coverage']:.4f}",
        f"  Pτ@0.5   leak={rq_t['leakage']:.4f}  coll={rq_t['collateral']:.4f}  "
        f"coverage={rq_t['coverage']:.4f}",
    ]
    if cat is not None:
        cat_p0 = agg(cat, "P0")
        cat_maj = agg(cat, "Pmaj")
        lines.append(
            f"  {cat} P0/Pmaj coll={cat_p0['collateral']:.4f}/"
            f"{cat_maj['collateral']:.4f}"
        )
    if rnd is not None:
        rnd_p0 = agg(rnd, "P0")
        lines.append(f"  {rnd} P0 coll={rnd_p0['collateral']:.4f}")

    cat_coll = agg(cat, "P0")["collateral"] if cat is not None else 0.0
    if rq_maj["collateral"] <= cat_coll + 0.05 and rq_maj["leakage"] <= 0.05:
        reading = (
            "Pmaj is already near the category bound at low leak: the L1 "
            "partition is fine; P0 was the wrong interface. No new tokenizer."
        )
    elif rq_p0["collateral"] >= 0.3 and rq_maj["collateral"] >= 0.3:
        reading = (
            "P0 is expensive and Pmaj still destroys a large share of acceptable "
            "catalog at this leak level: prefixes are mixed. A tokenizer "
            "intervention is justified."
        )
    else:
        reading = (
            "Pmaj trades leakage for collateral relative to P0. Read the "
            "leak–coll points before deciding whether to retrain the tokenizer."
        )
    lines.append(reading)
    return "\n".join(lines)


def fig_policies(
    reports: dict[str, dict],
    out: Path,
    dataset: str,
    level: int = 1,
) -> Path | None:
    """Leak vs collateral: Pτ curve plus P0 / Pmaj / Pτ@0.5 markers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tags = [t for t, r in reports.items() if str(level) in r.get("by_level", {})]
    if not tags:
        return None

    fig, axes = plt.subplots(
        1, len(tags), figsize=(4.1 * len(tags), 3.8), sharey=True, squeeze=False
    )
    grid = np.linspace(0, 1, 41)
    for ax, tag in zip(axes[0], tags):
        per_attr = reports[tag]["by_level"][str(level)]["per_attr"]
        stacked = []
        p0, pmaj, p05 = [], [], []
        for row in per_attr:
            pts = row["ptau"]
            leak = np.array([q["leakage"] for q in pts], dtype=float)
            coll = np.array([q["collateral"] for q in pts], dtype=float)
            order = np.argsort(leak)
            stacked.append(np.interp(grid, leak[order], coll[order]))
            p0.append((row["p0"]["leakage"], row["p0"]["collateral"]))
            pmaj.append((row["pmaj"]["leakage"], row["pmaj"]["collateral"]))
            p05.append((row["ptau_05"]["leakage"], row["ptau_05"]["collateral"]))
        if stacked:
            ax.plot(grid, np.mean(stacked, axis=0), c="0.45", lw=1.4, label="Pτ sweep")

        def _mean_xy(pairs: list[tuple[float, float]]) -> tuple[float, float]:
            xs, ys = zip(*pairs)
            return float(np.mean(xs)), float(np.mean(ys))

        x, y = _mean_xy(p0)
        ax.scatter([x], [y], s=70, zorder=3, label="P0")
        x, y = _mean_xy(pmaj)
        ax.scatter([x], [y], s=70, marker="s", zorder=3, label="Pmaj")
        x, y = _mean_xy(p05)
        ax.scatter([x], [y], s=70, marker="D", zorder=3, label="Pτ@0.5")
        ax.set_title(tag, fontsize=9)
        ax.set_xlabel("leakage")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("collateral")
    axes[0][-1].legend(fontsize=7, loc="upper right")
    fig.suptitle(
        f"Frozen-prefix control policies (depth {level}) -- {dataset}", fontsize=11
    )
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    path = out / ("fig4_policies.png" if level == 1 else f"fig4_policies_l{level}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
