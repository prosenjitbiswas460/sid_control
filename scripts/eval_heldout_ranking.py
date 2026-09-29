#!/usr/bin/env python
"""Same frozen checkpoints, two constraint families, two decoders.

Scores the users who are eligible under both the in-family attribute table
and the held-out table (brand or decade). Decoders are unconstrained beam
and the exact allowed-item trie. Writes

    results/<dataset>/heldout_ranking.md
    results/<dataset>/heldout_ranking.json
    figures/<dataset>/fig8_heldout_ranking.png

Does not rewrite control_eval.json, pruning.json, corpus.pkl, tokenizers,
or checkpoints. Skips sliced and any tokenizer whose checkpoint is missing.

    python scripts/eval_heldout_ranking.py --config configs/amazon_beauty.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sidctl.analysis.heldout import (  # noqa: E402
    fig_heldout_ranking,
    heldout_ranking_table,
)
from sidctl.attributes import AttributeTable  # noqa: E402
from sidctl.control.decoders import select_decoders  # noqa: E402
from sidctl.control.protocol import (  # noqa: E402
    instance_stats,
    pair_control_instances,
)
from sidctl.data import build_vocab  # noqa: E402
from sidctl.data.corpus import Corpus  # noqa: E402
from sidctl.eval import evaluate_control  # noqa: E402
from sidctl.sid import SIDTokenizer  # noqa: E402
from sidctl.train import Trainer  # noqa: E402
from sidctl.utils import load_config, pick_device, save_json, set_seed  # noqa: E402

_KEEP = ("rq_", "category_", "tiled_", "random_")
_SPECS = select_decoders(["unconstrained", "allowed_trie"])
_KEEP_METRICS = (
    "n",
    "ndcg",
    "hit_rate",
    "violation_rate",
    "any_violation",
    "ndcg_retention",
    "fill_rate",
    "short_list_rate",
    "latency_ms",
)


def _discover_tags(art: Path, ckpt_root: Path, cfg: dict, requested: list[str] | None) -> list[str]:
    tags: list[str] = []
    seen: set[str] = set()
    for spec in cfg.get("tokenizers", []):
        if spec["kind"] == "sliced":
            continue
        tag = f"{spec['kind']}_{spec['text_source']}"
        if tag not in seen:
            tags.append(tag)
            seen.add(tag)
    if ckpt_root.exists():
        for path in sorted(ckpt_root.glob("*/best.pt")):
            tag = path.parent.name
            if tag in seen or tag.startswith("sliced"):
                continue
            if tag.startswith(_KEEP):
                tags.append(tag)
                seen.add(tag)
    if requested:
        allow = set(requested)
        tags = [t for t in tags if t in allow]
    return tags


def _slim(results: dict) -> dict:
    return {
        "num_instances": results["num_instances"],
        "topk": results["topk"],
        "beam_size": results["beam_size"],
        "per_decoder": {
            name: {k: block[k] for k in _KEEP_METRICS if k in block}
            for name, block in results["per_decoder"].items()
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--checkpoints", default="checkpoints")
    ap.add_argument("--results", default="results")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--checkpoint-name", default="best.pt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-users", type=int, default=None)
    ap.add_argument("--beam-size", type=int, default=None)
    ap.add_argument("--tokenizers", nargs="+", default=None)
    ap.add_argument("--family", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = pick_device(args.device or cfg.get("device"))
    dataset = cfg["dataset"]
    art = Path(args.artifacts) / dataset
    corpus = Corpus.load(art / "corpus.pkl")
    family = args.family or ("decade" if dataset == "ml-1m" else "brand")
    held_path = art / "heldout" / family / "attributes.npz"
    if not held_path.exists():
        raise SystemExit(
            f"missing {held_path}; run "
            f"python scripts/build_heldout_attributes.py --config {args.config}"
        )
    held = AttributeTable.load(art / "heldout" / family / "attributes")
    if held.num_items != corpus.num_items:
        raise SystemExit(
            f"held-out table has {held.num_items} items, corpus has {corpus.num_items}"
        )

    ccfg = cfg.get("control", {})
    max_users = args.max_users if args.max_users is not None else ccfg.get("max_users")
    beam = args.beam_size or ccfg.get("beam_size", 50)
    topk = ccfg.get("topk", 10)
    in_name = str(corpus.meta.get("attribute_family", "in_family"))
    inn, hld = pair_control_instances(
        corpus,
        held,
        min_attr_count=ccfg.get("min_attr_count", 3),
        min_history=ccfg.get("min_history", 3),
        max_users=max_users,
        split="test",
        seed=cfg.get("seed", 42),
    )
    if not inn:
        raise SystemExit(
            f"no users eligible under both {in_name} and {family}"
        )
    print(
        f"dataset={dataset}  paired_users={len(inn)}  "
        f"in_family={in_name} ({corpus.attributes.num_attrs} attrs)  "
        f"held_out={family} ({held.num_attrs} attrs)  "
        f"beam={beam}  topk={topk}  device={device}"
    )
    print("in-family constraints:", instance_stats(inn)["top_attributes"])
    print("held-out constraints:", instance_stats(hld)["top_attributes"])

    ckpt_root = Path(args.checkpoints) / dataset
    runs: dict[str, dict] = {}
    for tag in _discover_tags(art, ckpt_root, cfg, args.tokenizers):
        tok_path = art / tag / "tokenizer.pkl"
        ckpt_path = ckpt_root / tag / args.checkpoint_name
        if not tok_path.exists():
            print(f"[{tag}] missing tokenizer, skip")
            continue
        if not ckpt_path.exists():
            print(f"[{tag}] missing checkpoint, skip")
            continue
        tok = SIDTokenizer.load(tok_path)
        vocab = build_vocab(tok)
        model, _ = Trainer.load_checkpoint(ckpt_path, device="cpu")
        print(f"[{tag}] in-family ...")
        inn_res = evaluate_control(
            model=model,
            tokenizer=tok,
            vocab=vocab,
            corpus=corpus,
            instances=inn,
            specs=_SPECS,
            beam_size=beam,
            topk=topk,
            max_history_len=cfg.get("train", {}).get("max_history_len", 20),
            device=device,
            attributes=corpus.attributes,
        )
        print(f"[{tag}] held-out {family} ...")
        hld_res = evaluate_control(
            model=model,
            tokenizer=tok,
            vocab=vocab,
            corpus=corpus,
            instances=hld,
            specs=_SPECS,
            beam_size=beam,
            topk=topk,
            max_history_len=cfg.get("train", {}).get("max_history_len", 20),
            device=device,
            attributes=held,
        )
        runs[tag] = {"in_family": _slim(inn_res), "held_out": _slim(hld_res)}
        for key, label in (("in_family", "in"), ("held_out", "held")):
            block = runs[tag][key]["per_decoder"]
            unc, trie = block["unconstrained"], block["allowed_trie"]
            print(
                f"  {label:4s}  unc={unc['ndcg']:.4f}  trie={trie['ndcg']:.4f}  "
                f"retain={trie['ndcg_retention']:.3f}  "
                f"viol unc={unc['violation_rate']:.3f} trie={trie['violation_rate']:.3f}"
            )
        del model

    if not runs:
        raise SystemExit("no checkpoints evaluated")

    out = Path(args.results) / dataset
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "in_family": in_name,
        "held_out": family,
        "num_users": len(inn),
        "beam_size": beam,
        "topk": topk,
        "decoders": ["unconstrained", "allowed_trie"],
        "runs": runs,
    }
    save_json(payload, out / "heldout_ranking.json")
    md = (
        f"# Held-out ranking: {dataset}\n\n"
        f"Same {len(inn)} users. In-family = `{in_name}`. Held-out = `{family}`. "
        f"Frozen checkpoints. Unconstrained beam vs exact allowed trie. "
        f"NDCG@{topk}. retain = trie NDCG / unconstrained NDCG within that family.\n\n"
        + "\n".join(heldout_ranking_table(runs))
    )
    (out / "heldout_ranking.md").write_text(md)
    print("\n" + md)
    fig_path = Path(args.figures) / dataset / "fig8_heldout_ranking.png"
    fig_heldout_ranking(
        runs, fig_path, dataset=dataset, in_name=in_name, held_name=family
    )
    print(f"wrote {out / 'heldout_ranking.json'}")
    print(f"wrote {out / 'heldout_ranking.md'}")
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
