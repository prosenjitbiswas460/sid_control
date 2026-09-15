#!/usr/bin/env python
"""Collect Part A and Part B results into the two tables the paper needs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sidctl.utils import load_config, save_json  # noqa: E402


def table(rows: list[dict], note: str = "") -> str:
    if not rows:
        return "(no results)"
    cols = list(rows[0])
    w = {c: max(len(c), *(len(f"{r[c]}") for r in rows)) for c in cols}
    lines = ["  ".join(c.ljust(w[c]) for c in cols)]
    lines.append("-" * len(lines[0]))
    for r in rows:
        lines.append("  ".join(f"{r[c]}".ljust(w[c]) for c in cols))
    if note:
        lines.append("")
        lines.append(note)
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dataset = cfg["dataset"]
    res_dir = Path(args.results) / dataset

    out_lines = [f"# Results: {dataset}", ""]

    part_a = res_dir / "part_a_prefix_control.json"
    if part_a.exists():
        data = json.loads(part_a.read_text())
        out_lines += [
            "## Table 1 -- Is an attribute constraint prefix-realisable?",
            "",
            table(
                data["table"],
                "collateral@0leak: share of acceptable catalog destroyed to fully\n"
                "honour the constraint by banning SID prefixes at that depth.",
            ),
            "",
            f"attributes tested: {', '.join(data['attrs_tested'])}",
            "",
        ]

    rows_b = []
    for spec in cfg["tokenizers"]:
        tag = f"{spec['kind']}_{spec['text_source']}"
        path = res_dir / tag / "control_eval.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for name, m in data["per_decoder"].items():
            rows_b.append(
                {
                    "tokenizer": tag,
                    "decoder": name,
                    "NDCG@10": round(m["ndcg"], 4),
                    "retain": round(m["ndcg_retention"], 3),
                    "viol@10": round(m["violation_rate"], 4),
                    "anyviol": round(m["any_violation"], 3),
                    "short": round(m["short_list_rate"], 3),
                    "fill": round(m["fill_rate"], 3),
                }
            )
    if rows_b:
        out_lines += [
            "## Table 2 -- What does enforcing the constraint cost at decoding time?",
            "",
            table(
                rows_b,
                "Constraints are compatible with the held-out target, so an ideal\n"
                "control surface would show retain=1.000, viol=0.000, short=0.000.",
            ),
            "",
        ]

    report = "\n".join(out_lines)
    print(report)
    (res_dir / "summary.md").write_text(report)
    save_json({"table1": json.loads(part_a.read_text())["table"] if part_a.exists() else [],
               "table2": rows_b}, res_dir / "summary.json")
    print(f"\nwrote {res_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
