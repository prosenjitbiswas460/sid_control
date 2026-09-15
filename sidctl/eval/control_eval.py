"""Part B: what does enforcing an attribute constraint cost at decoding time?

Because every constraint is compatible with the held-out target (see
``sidctl.control.protocol``), a control mechanism that were perfectly selective
would match the unconstrained NDCG exactly. The reported ``ndcg_retention`` is
therefore a direct measure of collateral damage.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from sidctl.control.decoders import DecoderSpec, decode
from sidctl.control.masks import MaskCache
from sidctl.control.protocol import ControlInstance
from sidctl.data.corpus import Corpus
from sidctl.data.encode import encode_history, to_tensors
from sidctl.data.vocab import Vocab
from sidctl.eval.metrics import hit_at_k, mean, ndcg_single
from sidctl.models.tiger import TigerGR
from sidctl.sid.tokenizer import SIDTokenizer


@torch.no_grad()
def evaluate_control(
    model: TigerGR,
    tokenizer: SIDTokenizer,
    vocab: Vocab,
    corpus: Corpus,
    instances: list[ControlInstance],
    specs: list[DecoderSpec],
    beam_size: int = 50,
    topk: int = 10,
    max_history_len: int = 20,
    device: str | None = None,
) -> dict:
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device_t).eval()

    mask_cache = MaskCache(tokenizer, corpus.attributes)
    attr_matrix = corpus.attributes.matrix

    rows: list[dict] = []
    for inst in tqdm(instances, desc="control eval"):
        tokens = encode_history(
            inst.history, tokenizer, vocab, max_history_len=max_history_len
        )
        if not tokens:
            continue
        input_ids, attn = to_tensors(tokens, device=device_t)

        for spec in specs:
            res = decode(
                model,
                tokenizer,
                input_ids,
                attn,
                spec,
                attr=inst.attr,
                mask_cache=mask_cache,
                beam_size=beam_size,
                topk=topk,
            )
            violations = sum(
                1 for it in res.items[:topk] if attr_matrix[it, inst.attr]
            )
            rows.append(
                {
                    "user_id": inst.user_id,
                    "attr_name": inst.attr_name,
                    "reason": inst.reason,
                    "decoder": spec.name,
                    "ndcg": ndcg_single(res.items, inst.target, topk),
                    "hit": hit_at_k(res.items, inst.target, topk),
                    "violation_rate": violations / topk,
                    "any_violation": float(violations > 0),
                    "short_list": float(res.n_returned < topk),
                    "fill_rate": res.n_returned / topk,
                    "beam_size": res.beam_size,
                }
            )

    return {
        "topk": topk,
        "beam_size": beam_size,
        "num_instances": len({r["user_id"] for r in rows}),
        "per_decoder": _aggregate(rows, specs, topk),
        "per_decoder_by_reason": _aggregate_by(rows, specs, topk, "reason"),
        "rows": rows,
    }


def _summarize(subset: list[dict]) -> dict:
    return {
        "n": len(subset),
        "ndcg": mean([r["ndcg"] for r in subset]),
        "hit_rate": mean([r["hit"] for r in subset]),
        "violation_rate": mean([r["violation_rate"] for r in subset]),
        "any_violation": mean([r["any_violation"] for r in subset]),
        "short_list_rate": mean([r["short_list"] for r in subset]),
        "fill_rate": mean([r["fill_rate"] for r in subset]),
    }


def _aggregate(rows: list[dict], specs: list[DecoderSpec], topk: int) -> dict:
    by_decoder = defaultdict(list)
    for r in rows:
        by_decoder[r["decoder"]].append(r)

    out = {}
    base = _summarize(by_decoder.get("unconstrained", []))
    for spec in specs:
        summary = _summarize(by_decoder.get(spec.name, []))
        base_ndcg = base.get("ndcg", float("nan"))
        summary["ndcg_retention"] = (
            float(summary["ndcg"] / base_ndcg)
            if base_ndcg and np.isfinite(base_ndcg) and base_ndcg > 0
            else float("nan")
        )
        summary["description"] = spec.description
        out[spec.name] = summary
    return out


def _aggregate_by(
    rows: list[dict], specs: list[DecoderSpec], topk: int, key: str
) -> dict:
    groups = sorted({r[key] for r in rows})
    return {
        g: _aggregate([r for r in rows if r[key] == g], specs, topk) for g in groups
    }
