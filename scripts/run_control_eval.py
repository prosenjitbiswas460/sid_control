#!/usr/bin/env python
"""Part B: cost of enforcing an attribute constraint at decoding time."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sidctl.control import DEFAULT_DECODERS, build_control_instances, instance_stats  # noqa: E402
from sidctl.data import build_vocab  # noqa: E402
from sidctl.data.corpus import Corpus  # noqa: E402
from sidctl.eval import evaluate_control  # noqa: E402
from sidctl.sid import SIDTokenizer  # noqa: E402
from sidctl.train import Trainer  # noqa: E402
from sidctl.utils import load_config, pick_device, save_json, set_seed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--checkpoints", default="checkpoints")
    ap.add_argument("--results", default="results")
    ap.add_argument("--checkpoint-name", default="best.pt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-users", type=int, default=None)
    ap.add_argument("--beam-size", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = pick_device(args.device or cfg.get("device"))

    dataset = cfg["dataset"]
    art = Path(args.artifacts) / dataset
    corpus = Corpus.load(art / "corpus.pkl")
    tok = SIDTokenizer.load(art / args.tokenizer / "tokenizer.pkl")
    vocab = build_vocab(tok)

    ckpt_path = Path(args.checkpoints) / dataset / args.tokenizer / args.checkpoint_name
    if not ckpt_path.exists():
        raise SystemExit(f"missing checkpoint {ckpt_path}; run train.py first")
    model, _ = Trainer.load_checkpoint(ckpt_path, device="cpu")

    ccfg = cfg.get("control", {})
    instances = build_control_instances(
        corpus,
        allowed_attrs=corpus.attributes.controllable_attrs(),
        min_attr_count=ccfg.get("min_attr_count", 3),
        min_history=ccfg.get("min_history", 3),
        max_users=args.max_users or ccfg.get("max_users"),
        split="test",
        seed=cfg.get("seed", 42),
    )
    stats = instance_stats(instances)
    print("control instances:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    if not instances:
        raise SystemExit("no eligible control instances")

    results = evaluate_control(
        model=model,
        tokenizer=tok,
        vocab=vocab,
        corpus=corpus,
        instances=instances,
        specs=DEFAULT_DECODERS,
        beam_size=args.beam_size or ccfg.get("beam_size", 50),
        topk=ccfg.get("topk", 10),
        max_history_len=cfg.get("train", {}).get("max_history_len", 20),
        device=device,
    )
    results["instance_stats"] = stats
    results["tokenizer"] = args.tokenizer

    _print(results)
    out = Path(args.results) / dataset / args.tokenizer
    rows = results.pop("rows")
    save_json(results, out / "control_eval.json")
    save_json(rows, out / "control_eval_rows.json")
    print(f"\nwrote {out / 'control_eval.json'}")


def _print(results: dict) -> None:
    k = results["topk"]
    print(
        f"\n{'decoder':<24s} {'NDCG@'+str(k):>9s} {'retain':>7s} "
        f"{'viol@'+str(k):>8s} {'anyviol':>8s} {'short':>7s} {'fill':>6s}"
    )
    print("-" * 72)
    for name, m in results["per_decoder"].items():
        print(
            f"{name:<24s} {m['ndcg']:>9.4f} {m['ndcg_retention']:>7.3f} "
            f"{m['violation_rate']:>8.4f} {m['any_violation']:>8.3f} "
            f"{m['short_list_rate']:>7.3f} {m['fill_rate']:>6.3f}"
        )
    print(
        "\nEvery constraint is compatible with the held-out target, so an ideal\n"
        "control surface would show retain=1.000, viol=0.000, short=0.000."
    )


if __name__ == "__main__":
    main()
