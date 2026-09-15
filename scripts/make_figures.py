#!/usr/bin/env python
"""Figures for the paper.

fig1_frontier      leakage vs collateral, one curve per prefix depth, per tokenizer
fig2_depth_cost    collateral at zero leakage against prefix depth
fig3_control_cost  NDCG retention vs violation rate for each decoder (needs Part B)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sidctl.utils import load_config  # noqa: E402


def fig_frontier(detail: dict, out: Path, dataset: str) -> None:
    tags = list(detail)
    fig, axes = plt.subplots(
        1, len(tags), figsize=(4.2 * len(tags), 3.6), sharey=True, squeeze=False
    )
    for ax, tag in zip(axes[0], tags):
        res = detail[tag]
        levels = sorted({p["level"] for p in res["frontier_points"]})
        for level in levels:
            curves = [
                p["points"] for p in res["frontier_points"] if p["level"] == level
            ]
            # Average the frontier over attributes on a common leakage grid.
            grid = np.linspace(0, 1, 41)
            stacked = []
            for pts in curves:
                leak = np.array([q["leakage"] for q in pts])
                coll = np.array([q["collateral"] for q in pts])
                order = np.argsort(leak)
                stacked.append(np.interp(grid, leak[order], coll[order]))
            if stacked:
                ax.plot(grid, np.mean(stacked, axis=0), label=f"depth {level}")
        ax.set_title(tag, fontsize=9)
        ax.set_xlabel("leakage (unwanted items surviving)")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("collateral (acceptable items destroyed)")
    axes[0][-1].legend(fontsize=8)
    fig.suptitle(f"Prefix-control realisability frontier -- {dataset}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig1_frontier.png", dpi=200)
    plt.close(fig)


def fig_depth_cost(table: list[dict], out: Path, dataset: str) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    tags = sorted({r["tokenizer"] for r in table})
    for tag in tags:
        rows = sorted(
            (r for r in table if r["tokenizer"] == tag), key=lambda r: r["level"]
        )
        ax.plot(
            [r["level"] for r in rows],
            [r["collateral@0leak"] for r in rows],
            marker="o",
            label=tag,
        )
    ax.set_xlabel("SID prefix depth used for the ban")
    ax.set_ylabel("collateral at zero leakage")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(sorted({r["level"] for r in table}))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"Cost of exact attribute control -- {dataset}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig2_depth_cost.png", dpi=200)
    plt.close(fig)


def fig_control_cost(per_decoder: dict, out: Path, dataset: str, tag: str) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for name, m in per_decoder.items():
        ax.scatter(m["violation_rate"], m["ndcg_retention"], s=60)
        ax.annotate(
            name,
            (m["violation_rate"], m["ndcg_retention"]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.axhline(1.0, ls="--", c="grey", lw=0.8)
    ax.axvline(0.0, ls="--", c="grey", lw=0.8)
    ax.set_xlabel("violation rate @K (lower is better)")
    ax.set_ylabel("NDCG retention vs unconstrained")
    ax.grid(alpha=0.3)
    ax.set_title(f"Control cost -- {dataset} / {tag}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / f"fig3_control_cost_{tag}.png", dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--figures", default="figures")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dataset = cfg["dataset"]
    res_dir = Path(args.results) / dataset
    out = Path(args.figures) / dataset
    out.mkdir(parents=True, exist_ok=True)

    part_a = res_dir / "part_a_prefix_control.json"
    if part_a.exists():
        data = json.loads(part_a.read_text())
        fig_frontier(data["detail"], out, dataset)
        fig_depth_cost(data["table"], out, dataset)
        print(f"wrote {out}/fig1_frontier.png and fig2_depth_cost.png")
    else:
        print(f"skipping Part A figures, {part_a} not found")

    for spec in cfg["tokenizers"]:
        tag = f"{spec['kind']}_{spec['text_source']}"
        path = res_dir / tag / "control_eval.json"
        if path.exists():
            data = json.loads(path.read_text())
            fig_control_cost(data["per_decoder"], out, dataset, tag)
            print(f"wrote {out}/fig3_control_cost_{tag}.png")


if __name__ == "__main__":
    main()
