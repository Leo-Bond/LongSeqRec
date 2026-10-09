# LongSeqRec

Lightweight long-sequence recommendation experiments for a Mac or a single
consumer GPU. The project compares Mean Pooling, DIN, SASRec, SIM, TWIN and DARE in one
controlled training pipeline, and includes a synthetic dataset with persistent
and short-term user interests.

Models are decomposed into Event Encoder, General Search Unit, Exact Search Unit
and CTR Tower components. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for
the model-to-paper mapping and implementation boundaries.

## Quick start

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run_experiments.py --config configs/smoke.json
.venv/bin/python run_experiments.py --config configs/local.json
```

Use `--device auto` to choose CUDA, Apple MPS, or CPU. Results are written as
JSON, CSV, and a Markdown report under the configured output directory.

## Models and controlled ablations

- `mean`: candidate-independent masked mean over the full history.
- `din`: candidate-aware attention over only the recent history.
- `sasrec`: causal self-attention over the complete behavior sequence.
- `sim`: category-based general search over the long history, followed by DIN.
- `twin`: multi-head target attention with inherent/cross feature splitting;
  CP-GSU and ESU share Q/K projections, cross-feature biases and relevance scores.
- `dare`: TWIN with independent matching and representation embeddings.
- `dare_shared`: DARE with the two tables tied, isolating embedding decoupling.
- `dare_no_long`: DARE restricted to the recent window, isolating long history.
- `dare_no_time`: DARE without time-gap embeddings.

This is a single-machine research implementation rather than a copy of private
industrial serving systems. SIM uses hard category search. TWIN implements the
paper's consistency-preserved multi-head target attention with the available
KuaiRand fields: item/category as inherent features and time gap as a cross
feature. DARE separates full attention and representation event encoders.

## Data

The default synthetic generator deliberately gives every user stable long-term
interests plus a drifting short-term interest. It is useful for checking whether
the implementation can recover the expected ablation trend before using real
logs. Splits are chronological within each user and every history is strictly
prior to its target.

For external data, create an `.npz` file with one interaction per row:

```text
user_id   integer user ID
item_id   integer item ID
category_id integer category ID
timestamp integer or float ordering key
label     optional binary label; omitted means every row is a positive event
```

Then set `data.kind` to `npz` and `data.path` in a config. IDs are remapped to
contiguous values during preprocessing. Positive-only logs use sampled negative
targets, optionally drawing same-category hard negatives. Labeled impression
logs retain their labels and build history from earlier positive events.

### KuaiRand-1K

Convert the official extracted dataset and run the single-seed 4060 config:

```bash
.venv/bin/python prepare_kuairand.py \
  --root /path/to/KuaiRand-1K \
  --output data/kuairand_1k.npz \
  --positive-field long_view
.venv/bin/python run_experiments.py --config configs/kuairand_4060.json
```

The converter streams the large CSV files, derives a coarse category from the
first video tag, and writes an audit JSON with coverage and label statistics.
Only earlier positive events enter a sample's history, preventing impression
and future-information leakage.

For all 1,000 users on an 18GB Apple Silicon Mac, use dynamic history indexing:

```bash
.venv/bin/python run_experiments.py \
  --config configs/kuairand_mac_full.json \
  --device mps
```

The default full-Mac study runs one epoch over the five core comparisons. Nine
models over several epochs is a multi-day laptop workload rather than a memory
requirement.

## Fair-comparison rules

All models use the same embedding dimension, classifier shape, samples, seed,
optimizer and early-stopping metric. Reports include mean and standard deviation
for the configured runs. Parameter counts are shown because DARE intentionally adds a second
item embedding table.
