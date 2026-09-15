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
delete to *fully* honour the request. Four tokenizers bracket the answer:

| tokenizer | role |
|---|---|
| `rq_title` | the honest system under test, no attribute leakage into the SID |
| `rq_title_attr` | what feeding the attribute into the tokenizer buys you |
| `category_title` | upper bound: level 0 *is* the attribute by construction |
| `shuffled_title` | **the null model**: real RQ codes with the item assignment permuted |
| `random_title` | uniform codes; *not* granularity-matched, see below |

The `shuffled` null is essential. `collateral@0leak` is confounded by prefix
granularity: uniform random codes give almost one prefix per item, so a "prefix
ban" degenerates into an item blacklist and scores near-zero collateral while
providing no control at all. Permuting the real RQ table preserves the prefix
size distribution exactly, so the `rq` vs `shuffled` gap isolates semantic
alignment. For the same reason, purity is reported alongside **AMI**
(chance-corrected mutual information): raw purity is inflated by attribute base
rates, and NMI grows with the number of prefixes.

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

## Findings so far

Part A is complete on two datasets, and the answer is **conditional, not a flat
negative**. Prefix controllability is governed by whether the attribute is a
*partition* of the catalog, not by the SID hierarchy.

All numbers are level-1 (`title`-only tokenizer, so text richness is matched
across datasets), against the granularity-matched `shuffled` null whose prefix
size distribution is identical to `rq` by construction.

| dataset | attribute | labels/item | multi-label | upper-bound collateral | rq collateral | null collateral | rq gap | rq AMI |
|---|---|---|---|---|---|---|---|---|
| Amazon Beauty (12086) | 6 categories | **1.000** | 0.0% | 0.000 | 0.533 | 0.962 | **0.429** | 0.241 |
| Amazon Sports (17833) | 14 categories | **1.099** | 4.8% | 0.088 | 0.486 | 0.840 | **0.354** | 0.212 |
| MovieLens-1M (3416) | 18 genres | **1.707** | 51.5% | 0.238 | 0.654 | 0.698 | **0.044** | 0.041 |

1. **Label overlap, not the SID hierarchy, sets the cost of prefix control.**
   The upper bound -- a tokenizer whose level-1 code *is* the attribute -- tracks
   labels/item almost exactly: 1.000 to 0.000 collateral, 1.099 to 0.088, 1.707
   to 0.238. Overlap imposes a floor that no tokenizer can beat, because banning
   one label necessarily bans the co-occurring labels of the same items.
2. **RQ prefixes are attribute-aligned only when the attribute is a
   partition.** On Beauty and Sports the gap over the null is large (0.43,
   0.35). On MovieLens it collapses to 0.044 with AMI 0.041, i.e. RQ level-1
   codes are statistically indistinguishable from a permutation that destroys
   all semantics, and suppressing one genre destroys 65% of the acceptable
   catalog.
3. **It is not a text-richness artifact.** These rows use titles only. Adding
   Amazon descriptions moves Beauty from 0.533 to 0.470 collateral -- a small
   effect next to the cross-dataset spread.
4. **Purity and NMI would have hidden this.** On MovieLens `rq_title` scores
   0.468 purity against 0.438 for the null, because Drama alone covers 39% of the
   catalog. Chance correction is what exposes the gap as near-zero, and raw NMI
   additionally inflates with prefix count.
5. **Attribute supervision helps where the content signal is weak.** Putting
   attribute names into the tokenizer text lifts level-1 AMI from 0.041 to 0.297
   on MovieLens and from 0.212 to 0.362 on Sports.

Part B (decoding cost) still needs a trained model per tokenizer; see below.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The existing `.venv` was created under the repo's former name, so its console
scripts have stale shebangs. Use `.venv/bin/python -m pip ...` and
`.venv/bin/python -m pytest ...`, or recreate the environment.

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

Part A alone answers the go/no-go question and needs no GPU:

```bash
PART_A_ONLY=1 ./scripts/run_all.sh configs/ml1m.yaml
```

Full pipeline for one dataset:

```bash
./scripts/run_all.sh configs/ml1m.yaml
```

Or step by step:

```bash
python scripts/prepare_data.py          --config configs/ml1m.yaml
python scripts/analyze_prefix_control.py --config configs/ml1m.yaml
python scripts/train.py                 --config configs/ml1m.yaml --tokenizer rq_title
python scripts/run_control_eval.py      --config configs/ml1m.yaml --tokenizer rq_title
python scripts/make_figures.py          --config configs/ml1m.yaml
python scripts/summarize.py             --config configs/ml1m.yaml
```

## Reading the output

`results/<dataset>/summary.md` holds the two paper tables.

Table 1 is Part A. **`collateral@0leak` near 0 means prefix control is free;
near 1 means it is a fiction.** Compare `rq_*` against `category_*` (upper
bound) and `random_*` (floor); where `rq` falls between them is the paper.

Table 2 is Part B. Since constraints are target-compatible, an ideal control
surface shows `retain=1.000`, `viol=0.000`, `short=0.000`. Watch for the two
distinct failure modes: prefix masking loses `retain`, post-filtering loses
`fill` and gains `short`.

Figures land in `figures/<dataset>/`:

- `fig1_frontier.png` -- leakage vs collateral, one curve per depth
- `fig2_depth_cost.png` -- the depth/precision trade-off, the paper's core figure
- `fig3_control_cost_<tok>.png` -- NDCG retention vs violation rate per decoder

## Layout

```
sidctl/
  attributes/   item x attribute incidence table (the control ground truth)
  data/         MovieLens + Amazon loaders, corpus container, leave-one-out splits
  sid/          RQ-KMeans and the three tokenizer variants
  analysis/     Part A: prefix purity and the realisability frontier
  control/      prefix masks, constraint-selection protocol, decoders
  models/       TIGER-style T5 with trie-constrained, ban-aware beam search
  train/        training loop
  eval/         ranking metrics and the Part B control evaluation
scripts/        prepare_data, analyze_prefix_control, train, run_control_eval,
                make_figures, summarize, run_all.sh, download_data.sh
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
