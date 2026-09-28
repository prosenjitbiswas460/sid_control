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

    def _next_logprobs(
        self,
        encoder_out: BaseModelOutput,
        attention_mask: torch.Tensor,
        prefixes: list[tuple[int, ...]],
    ) -> torch.Tensor:
        """Full-vocabulary log-probabilities of the next code after each prefix."""
        device = attention_mask.device
        start_id = self.model.config.decoder_start_token_id
        dec_in = torch.tensor(
            [[start_id] + self.sid_to_token_ids(p) for p in prefixes],
            dtype=torch.long,
            device=device,
        )
        n = len(prefixes)
        # repeat rather than expand: the encoder states are consumed by
        # cross-attention, and a contiguous copy avoids any stride surprises.
        expanded = BaseModelOutput(
            last_hidden_state=encoder_out.last_hidden_state.repeat(n, 1, 1)
        )
        logits = self.model(
            encoder_outputs=expanded,
            attention_mask=attention_mask.repeat(n, 1),
            decoder_input_ids=dec_in,
            return_dict=True,
        ).logits[:, -1, :]
        return torch.log_softmax(logits.float(), dim=-1)

    @torch.no_grad()
    def score_sids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sid_tokens: torch.Tensor,
        chunk_size: int = 4096,
    ) -> torch.Tensor:
        """
        Exact sequence log-probability of every SID in ``sid_tokens``.

        ``sid_tokens`` is ``[N, sid_length]`` of vocabulary ids (level offsets
        already applied). Scores use the same full-vocabulary log-softmax as
        :meth:`beam_search`, so an oracle ranking and a beam ranking are
        directly comparable.
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
        encoder_out = self.model.get_encoder()(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        start_id = self.model.config.decoder_start_token_id
        out = torch.empty(sid_tokens.size(0), device=sid_tokens.device)
        for s in range(0, sid_tokens.size(0), chunk_size):
            tok = sid_tokens[s : s + chunk_size]
            n = tok.size(0)
            start = torch.full((n, 1), start_id, dtype=torch.long, device=tok.device)
            dec_in = torch.cat([start, tok[:, :-1]], dim=1)
            logits = self.model(
                encoder_outputs=BaseModelOutput(
                    last_hidden_state=encoder_out.last_hidden_state.expand(n, -1, -1)
                ),
                attention_mask=attention_mask.expand(n, -1),
                decoder_input_ids=dec_in,
                return_dict=True,
            ).logits
            lp = torch.log_softmax(logits.float(), dim=-1)
            out[s : s + n] = lp.gather(-1, tok.unsqueeze(-1)).squeeze(-1).sum(-1)
        return out

    @torch.no_grad()
    def beam_search(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prefix_trie: list[dict[tuple, list[int]]],
        beam_size: int = 20,
        banned_prefixes: set[tuple] | None = None,
        max_return: int | None = None,
        prefix_bonus: list[dict[tuple, float]] | None = None,
        bonus_weight: float = 0.0,
        lookahead: int = 0,
    ) -> list[tuple[tuple[int, ...], float]]:
        """
        Trie-constrained beam search for a single user history.

        Every returned sequence is a valid catalog SID. ``banned_prefixes`` is a
        set of equal-length prefixes removed from the trie; this is how an
        attribute constraint is applied at the identifier level, so the beam
        never spends budget on a banned branch.

        ``prefix_bonus`` and ``lookahead`` change only which prefixes survive
        pruning; returned scores and the final order are the model's own
        log-probabilities.

        * ``prefix_bonus[level][prefix]`` is added (times ``bonus_weight``) to
          the pruning score of a depth ``level + 1`` prefix. PACD passes the
          log share of allowed items under the prefix.
        * ``lookahead = M > 0`` pre-selects ``M * beam_size`` candidates, then
          prunes by true score plus the model's log-mass on the candidate's
          children that remain in ``prefix_trie``. One extra forward per level.

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
        use_bonus = prefix_bonus is not None and bonus_weight != 0.0
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

            logprobs = self._next_logprobs(
                encoder_out, attention_mask, [p for p, _ in beams]
            )

            mask = torch.zeros_like(logprobs, dtype=torch.bool)
            bonus = torch.zeros_like(logprobs) if use_bonus else None
            for b, codes in enumerate(allowed):
                if codes:
                    idx = torch.as_tensor(codes, device=device, dtype=torch.long) + offset
                    mask[b, idx] = True
                    if use_bonus:
                        table = prefix_bonus[level]
                        prefix = beams[b][0]
                        vals = [table.get(prefix + (c,), 0.0) for c in codes]
                        bonus[b, idx] = torch.tensor(vals, device=device) * bonus_weight

            prior = torch.tensor([s for _, s in beams], dtype=torch.float, device=device)
            scores = logprobs.masked_fill(~mask, NEG_INF) + prior[:, None]
            prune = scores + bonus if use_bonus else scores

            vocab = scores.size(1)
            flat_true = scores.reshape(-1)
            flat_prune = prune.reshape(-1)
            k = min(beam_size, n_allowed)
            last = level + 1 == self.sid_length

            if lookahead > 0 and not last:
                n_cand = min(lookahead * beam_size, n_allowed)
                _, cand_idx = flat_prune.topk(n_cand)
                cand_idx = cand_idx[flat_true[cand_idx] > NEG_INF]
                cand = []
                for flat_idx in cand_idx.tolist():
                    beam_i, token_id = divmod(flat_idx, vocab)
                    cand.append(beams[beam_i][0] + (token_id - offset,))
                nxt = self._next_logprobs(encoder_out, attention_mask, cand)
                next_off = self.level_offsets[level + 1]
                nmask = torch.zeros_like(nxt, dtype=torch.bool)
                for j, p in enumerate(cand):
                    codes = prefix_trie[level + 1].get(p, [])
                    if codes:
                        idx = torch.as_tensor(codes, device=device, dtype=torch.long)
                        nmask[j, idx + next_off] = True
                mass = torch.logsumexp(nxt.masked_fill(~nmask, NEG_INF), dim=-1)
                sel = flat_prune[cand_idx] + mass
                keep = sel.topk(min(k, sel.numel())).indices
                top_idx = cand_idx[keep]
            else:
                _, top_idx = flat_prune.topk(k)

            parents = beams
            beams = []
            for flat_idx in top_idx.tolist():
                score = float(flat_true[flat_idx])
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
