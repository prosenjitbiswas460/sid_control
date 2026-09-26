# Is Semantic-ID Control an Illusion?

Prefix-realisability of attribute constraints in generative recommendation.
Target venue: **ECIR 2027 short paper**.

## The question

Every Semantic ID (SID) paper repeats that the first code is coarse semantics and
later codes are fine detail. If that were true, user control would be nearly
free: to honour *"show me less horror"*, ban the level-1 codes that carry horror
items and decode as usual. No retraining, no reranker, no embeddings.

Two 2026 results make this doubtful. SIDInspector reports that learned level-1
prefixes recover only 0.154 of co-occurrence neighbourhoods versus 0.447 for a
deterministic category prefix. Wang et al. find SIDs keep broad organisation but
lose fine local structure. So the coarse-to-fine story may be far too weak to
carry a control interface.

**Claim under test.** Attribute constraints are *not prefix-realisable* in
RQ-based Semantic IDs: enforcing one by prefix masking destroys a large fraction
of unrelated, perfectly acceptable catalog, and the cost falls with masking
depth only as the computational benefit disappears.

Both outcomes are publishable. If prefix control works, the paper reports a free
controllability mechanism. If it does not, the paper reports that hierarchical
SIDs do not give controllability for free, which is the more interesting result
and speaks directly to the governable-personalization agenda.

## Two measurements

**Part A (no GPU, minutes).** Given a SID table and an item-attribute table,
sweep a masking threshold and record two quantities per attribute and depth:

- **leakage** -- share of unwanted items that survive the mask (the violation)
- **collateral** -- share of acceptable items destroyed by the mask (the price)

A perfect control surface reaches zero leakage at zero collateral. The headline
number is `collateral@0leak`: what fraction of the acceptable catalog you must
delete to *fully* honour the request. Tokenizers bracket the answer:

| tokenizer | role |
|---|---|
| `rq_title` | the honest system under test, no attribute leakage into the SID |
| `rq_title_attr` | what feeding the attribute into the tokenizer buys you |
| `category_title` | single-label upper bound: level 0 *is* the dominant attribute |
| `tiled_title` | Proposal 2: one control tile per attribute; AND ban (multi-label fix) |
| `random_title` | lower bound: prefixes carry no attribute information |

**Part B (needs one trained model per tokenizer).** Budget-matched decoders,
each enforcing the same constraint in a different place:

| decoder | mechanism |
|---|---|
| `unconstrained` | no control; accuracy reference |
| `prefix_mask_l1/l2/l3` | ban SID prefixes at that depth (cheap, blunt) |
| `post_filter` | exact item-level ban on the generated list (starves the beam) |
| `post_filter_2x` | same, given twice the beam budget |
| `prefix_l1_plus_filter` | ban majority-attribute prefixes, filter the rest |

The protocol's key design choice: **every constraint is compatible with the
held-out target**, i.e. the item the user actually consumed next does not carry
the banned attribute. A perfect control surface would therefore lose no accuracy
at all, so the reported `retain` (NDCG retention) is a direct measure of
collateral damage rather than a conflict with the user's real preference.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Datasets

| config | dataset | attribute | download |
|---|---|---|---|
| `configs/ml1m.yaml` | MovieLens-1M | 18 genres | ~6 MB, automatic |
| `configs/amazon_beauty.yaml` | Amazon Beauty 5-core (2014) | category taxonomy | ~170 MB |
| `configs/amazon_sports.yaml` | Amazon Sports & Outdoors 5-core | category taxonomy | ~250 MB |

Amazon 2014 is the standard TIGER benchmark, so accuracy numbers stay comparable
to published SID recommenders. Set `attribute_family: brand` in the Amazon
configs for the non-semantic contrast: a text-derived SID has some chance of
organising by category and almost none of organising by brand.

Datasets download on first use, or fetch them up front on a server:

```bash
./scripts/download_data.sh ml1m
./scripts/download_data.sh beauty sports
```

## Running

Smoke test first (seconds, CPU):

```bash
python -m pytest tests/test_smoke.py -q
```

## Locked experiment: P0 vs Pτ vs Pmaj (Beauty first)

Keep the frozen codebook. Change only the prefix-ban policy. No GPU, no
retraining, no token-ID remapping (that cannot move leak or collateral).

| policy | ban prefix `p` when |
|---|---|
| P0 | `p` contains any forbidden item (zero leak, current coll@0leak) |
| Pτ | `P(forbidden \| p) ≥ τ` (sweep; headline point τ=0.5) |
| Pmaj | unique majority attribute of `p` is the forbidden one |

Needs `artifacts/<dataset>/corpus.pkl` and `artifacts/<dataset>/<tag>/tokenizer.pkl`.

```bash
# Beauty — the clean disjoint-category bound
python scripts/eval_control_policies.py --config configs/amazon_beauty.yaml --levels 1

# same command on the other catalogs after their tokenizers exist
python scripts/eval_control_policies.py --config configs/amazon_sports.yaml --levels 1
python scripts/eval_control_policies.py --config configs/ml1m.yaml --levels 1
```

Writes `results/<dataset>/control_policies.json`, `control_policies.md`, and
`figures/<dataset>/fig4_policies.png`.

If tokenizers are missing:

```bash
./scripts/download_data.sh beauty
python scripts/prepare_data.py --config configs/amazon_beauty.yaml
python scripts/eval_control_policies.py --config configs/amazon_beauty.yaml --levels 1
```

## Copy to a server

Code only, if the server already has `data/` and `artifacts/` from Part A:

```bash
# from this repo root (code only; server already has data/ and artifacts/)
rsync -avz \
  --exclude '.venv' --exclude '.git' --exclude '__pycache__' \
  --exclude '.pytest_cache' --exclude 'data' --exclude 'artifacts' \
  --exclude 'results' --exclude 'figures' --exclude 'logs' --exclude 'checkpoints' \
  ./ \
  USER@SERVER:~/negative_evidence_gr/
```

If the server has neither data nor tokenizers, drop the `--exclude 'data'` and
`--exclude 'artifacts'` flags, or rebuild on the server with `download_data.sh`
+ `prepare_data.py`.

On the server:

```bash
cd ~/negative_evidence_gr
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/test_smoke.py -q
python scripts/eval_control_policies.py --config configs/amazon_beauty.yaml --levels 1
```

## Running the rest of the pipeline

Part A alone answers the go/no-go question and needs no GPU:

```bash
PART_A_ONLY=1 ./scripts/run_all.sh configs/ml1m.yaml
```

MovieLens Part B for the tiled tokenizer only:

```bash
TRAIN_TOKENIZERS="tiled_title" ./scripts/run_all.sh configs/ml1m.yaml
```

Full pipeline for one dataset:

```bash
./scripts/run_all.sh configs/ml1m.yaml
```

Or step by step:

```bash
python scripts/prepare_data.py            --config configs/ml1m.yaml
python scripts/analyze_prefix_control.py  --config configs/ml1m.yaml
python scripts/eval_control_policies.py   --config configs/ml1m.yaml --levels 1
python scripts/train.py                   --config configs/ml1m.yaml --tokenizer rq_title
python scripts/run_control_eval.py        --config configs/ml1m.yaml --tokenizer rq_title
python scripts/make_figures.py            --config configs/ml1m.yaml
python scripts/summarize.py               --config configs/ml1m.yaml
```

Part B for Proposal 2 (tiled identifiers on MovieLens):

```bash
python scripts/train.py            --config configs/ml1m.yaml --tokenizer tiled_title
python scripts/run_control_eval.py --config configs/ml1m.yaml --tokenizer tiled_title
python scripts/summarize.py        --config configs/ml1m.yaml
```

## Reading the output

`results/<dataset>/summary.md` holds the two paper tables.

Table 1 is Part A. **`collateral@0leak` near 0 means prefix control is free;
near 1 means it is a fiction.** Compare `rq_*` against `category_*` (upper
bound) and `random_*` (floor); where `rq` falls between them is the paper.

Table 1b is P0 vs Pτ vs Pmaj on the **same** frozen prefixes. P0 is coll@0leak.
Pmaj bans prefixes whose majority attribute is the forbidden one. Pτ@0.5 bans
if `P(forbidden|prefix) ≥ 0.5`. If Pmaj already sits near `category` at low
leak, the partition is fine and only the interface was wrong. If P0 is expensive
**and** Pmaj still has large collateral, the clusters are mixed.

Table 2 is Part B. Since constraints are target-compatible, an ideal control
surface shows `retain=1.000`, `viol=0.000`, `short=0.000`. Watch for the two
distinct failure modes: prefix masking loses `retain`, post-filtering loses
`fill` and gains `short`.

Figures land in `figures/<dataset>/`:

- `fig1_frontier.png` -- leakage vs collateral, one curve per depth
- `fig2_depth_cost.png` -- the depth/precision trade-off, the paper's core figure
- `fig3_control_cost_<tok>.png` -- NDCG retention vs violation rate per decoder
- `fig4_policies.png` -- P0 / Pmaj / Pτ@0.5 on the frozen L1 codebook

## Layout

```
sidctl/
  attributes/   item x attribute incidence table (the control ground truth)
  data/         MovieLens + Amazon loaders, corpus container, leave-one-out splits
  sid/          RQ-KMeans and tokenizer variants (rq, category, tiled, random)
  analysis/     Part A: prefix purity, realisability frontier, P0/Pτ/Pmaj
  control/      prefix masks, constraint-selection protocol, decoders
  models/       TIGER-style T5 with trie-constrained, ban-aware beam search
  train/        training loop
  eval/         ranking metrics and the Part B control evaluation
scripts/        prepare_data, analyze_prefix_control, eval_control_policies,
                train, run_control_eval, make_figures, summarize, run_all.sh,
                download_data.sh
configs/        smoke, ml1m, amazon_beauty, amazon_sports
```

## Notes on correctness

Two bugs from the earlier iteration of this repo are fixed here, and both matter
for the results:

- Beam search enumerated the entire trie and scored full SIDs with one forward
  pass each, rather than pruning per level. It is now a real constrained beam
  search with per-level top-B pruning, which is what makes the budget-matched
  decoder comparison meaningful.
- T5 labels were prefixed with `-100` for the encoder length, a decoder-only
  convention. Labels are now the target SID alone, so the decoder inputs seen
  during training match those used at inference.
