"""TIGER-style encoder-decoder over Semantic ID tokens."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

# T5 geometry, inlined so no HuggingFace download is needed.
T5_BASE_CONFIG = {
    "d_ff": 1024,
    "d_kv": 64,
    "d_model": 256,
    "decoder_start_token_id": 0,
    "dropout_rate": 0.1,
    "eos_token_id": 1,
    "feed_forward_proj": "relu",
    "initializer_factor": 1.0,
    "is_encoder_decoder": True,
    "layer_norm_epsilon": 1e-6,
    "num_decoder_layers": 4,
    "num_heads": 6,
    "num_layers": 4,
    "pad_token_id": 0,
    "relative_attention_max_distance": 128,
    "relative_attention_num_buckets": 32,
    "tie_word_embeddings": True,
    "use_cache": True,
}

NEG_INF = float("-inf")


class TigerGR(nn.Module):
    """
    Seq2seq model mapping a user's SID history to the next item's SID.

    The vocabulary is a flat concatenation of per-level code sets, so
    ``level_offsets[l] + code`` is the token id of ``code`` at position ``l``.
    Token id 0 is padding.
    """

    def __init__(
        self,
        vocab_size: int,
        sid_length: int,
        level_sizes: list[int],
        pos_token_id: int = 0,
        neg_token_id: int = 0,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.sid_length = sid_length
        self.level_sizes = list(level_sizes)
        self.pos_token_id = pos_token_id
        self.neg_token_id = neg_token_id
        self.vocab_size = vocab_size

        offsets, running = [], 1
        for size in self.level_sizes:
            offsets.append(running)
            running += size
        self.level_offsets = offsets

        cfg = dict(T5_BASE_CONFIG)
        cfg.update(
            vocab_size=vocab_size,
            d_model=d_model,
            d_ff=d_ff,
            num_layers=num_layers,
            num_decoder_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout,
            d_kv=max(d_model // num_heads, 16),
        )
        self.model = T5ForConditionalGeneration(T5Config(**cfg))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict:
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        return {"loss": out.loss, "logits": out.logits}

    def sid_to_token_ids(self, sid: tuple[int, ...]) -> list[int]:
        return [self.level_offsets[l] + c for l, c in enumerate(sid)]

    # ---------------------------------------------------------------- decoding

    @torch.no_grad()
    def beam_search(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prefix_trie: list[dict[tuple, list[int]]],
        beam_size: int = 20,
        banned_prefixes: set[tuple] | None = None,
        max_return: int | None = None,
    ) -> list[tuple[tuple[int, ...], float]]:
        """
        Trie-constrained beam search for a single user history.

        Every returned sequence is a valid catalog SID. ``banned_prefixes`` is a
        set of equal-length prefixes removed from the trie; this is how an
        attribute constraint is applied at the identifier level, so the beam
        never spends budget on a banned branch.

        Returns ``(sid, logprob)`` pairs in descending score order.
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
        if input_ids.size(0) != 1:
            raise ValueError("beam_search expects a single example")

        device = input_ids.device
        ban_len = len(next(iter(banned_prefixes))) if banned_prefixes else 0

        encoder_out = self.model.get_encoder()(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        start_id = self.model.config.decoder_start_token_id
        beams: list[tuple[tuple[int, ...], float]] = [((), 0.0)]

        for level in range(self.sid_length):
            offset = self.level_offsets[level]

            allowed: list[list[int]] = []
            for prefix, _ in beams:
                codes = prefix_trie[level].get(prefix, [])
                if banned_prefixes and level + 1 == ban_len:
                    codes = [c for c in codes if (prefix + (c,)) not in banned_prefixes]
                allowed.append(codes)
            n_allowed = sum(len(c) for c in allowed)
            if n_allowed == 0:
                return []

            n_beams = len(beams)
            dec_in = torch.tensor(
                [
                    [start_id] + self.sid_to_token_ids(prefix)
                    for prefix, _ in beams
                ],
                dtype=torch.long,
                device=device,
            )
            # repeat rather than expand: the encoder states are consumed by
            # cross-attention, and a contiguous copy avoids any stride surprises.
            expanded = BaseModelOutput(
                last_hidden_state=encoder_out.last_hidden_state.repeat(n_beams, 1, 1)
            )
            logits = self.model(
                encoder_outputs=expanded,
                attention_mask=attention_mask.repeat(n_beams, 1),
                decoder_input_ids=dec_in,
                return_dict=True,
            ).logits[:, -1, :]
            logprobs = torch.log_softmax(logits.float(), dim=-1)

            mask = torch.zeros_like(logprobs, dtype=torch.bool)
            for b, codes in enumerate(allowed):
                if codes:
                    idx = torch.as_tensor(codes, device=device, dtype=torch.long) + offset
                    mask[b, idx] = True

            prior = torch.tensor([s for _, s in beams], dtype=torch.float, device=device)
            scores = logprobs.masked_fill(~mask, NEG_INF) + prior[:, None]

            vocab = scores.size(1)
            k = min(beam_size, n_allowed)
            top_scores, top_idx = scores.reshape(-1).topk(k)

            parents = beams
            beams = []
            for score, flat_idx in zip(top_scores.tolist(), top_idx.tolist()):
                if score == NEG_INF:
                    continue
                beam_i, token_id = divmod(flat_idx, vocab)
                beams.append((parents[beam_i][0] + (token_id - offset,), score))
            if not beams:
                return []

        beams.sort(key=lambda x: x[1], reverse=True)
        return beams[: max_return or beam_size]

    @torch.no_grad()
    def generate_sid(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prefix_trie: list[dict[tuple, list[int]]],
        sid_length: int | None = None,
        beam_size: int = 20,
        device: torch.device | None = None,
    ) -> list[tuple[int, ...]]:
        """Backwards-compatible wrapper returning SIDs without scores."""
        return [
            sid
            for sid, _ in self.beam_search(
                input_ids, attention_mask, prefix_trie, beam_size=beam_size
            )
        ]
