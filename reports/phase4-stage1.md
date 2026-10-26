# Phase 4, stage 1 — EOT target extraction

Date: 2026-10-25.

## Pipeline

- Teacher: Ornith 1.5 9B Q5_K_M, all layers on the RTX 3070
  (`--ngl 99`), `ctx=4096`, `batch=64`.
- Documents are tokenized independently and fed as
  `[BOS] + doc tokens`, one segment per document (`--doc-starts`). The
  dumper clears memory at every document start, so no cross-document
  attention leaks into the targets. Documents longer than 4096 tokens are
  split into context-bounded segments.
- Logits are streamed over stdout and converted immediately with the DFS
  byte trie. EOT rows and explicit `next_symbols` are stored; raw logits
  are never persisted.
- Conversion uses a float32 cumulative sum, an observed-token probability
  floor for rare prefixes, and a non-negative clamp followed by row
  renormalization. This avoids both zero-prefix-mass NaNs and negative
  fp16 rounding artifacts.
- Marginalize-It is derived at train time by shifting to the next row at
  token-end positions; it is not stored separately.

## Stage 1 result

| property | value |
|---|---:|
| documents | 18 |
| teacher tokens | 132,396 |
| content bytes | 325,065 |
| student positions | 457,461 |
| byte / EOT / BOS symbols | 325,065 / 132,378 / 18 |
| next-byte / next-EOT targets | 325,065 / 132,396 |
| shards | 2 |
| target size | 0.238 GB |
| extraction time | 490.1 s |
| teacher tokens/s | 270.2 |

The selected documents overshoot the 100k-token stage limit because
selection is document-granular; the full documents were kept to preserve
their context.

## Validation

A full scan of all 457,461 positions (`scripts/validate_shards.py`):

- no non-finite rows;
- no negative target values;
- every row sums to 1 within 1e-2;
- the stored `next_symbols` always has positive target mass;
- document ids are non-decreasing;
- manifest counts match the data exactly.

## Cost projection

The teacher alone runs at 890 tokens/s; the end-to-end rate is 270 tokens/s
because logit transfer and CPU conversion dominate. At this rate:

- 1 M teacher tokens: ~1.0 h;
- the full 6.59 M-token corpus: ~6.8 h;
- dense targets: ~13.4 GB for the full corpus.

The 100k shard is sufficient for student learning and objective
comparisons; the 1M and full stages are throughput runs, not correctness
risks.

## Artifacts

- `/data/ornith-byte/targets/train-100k/` — shards, manifests, report.
- `scripts/validate_shards.py` — full-scan structural checks.
- `src/byte_agent/{shards,extract}.py` — writer/reader and extraction loop.

## Correction (2026-10-26): Python float32 precision, C++ converter

The stage-1 implementation used a float32 cumulative sum in Python. On a
float64 reference, that path can misallocate rare-prefix mass by up to
**0.789** in a target row. The error is largest where a token is a prefix of
other tokens and the interval is tiny relative to the total mass. Row sums
stay 1 and the observed path is floored, so the data was structurally valid
but not exact.

The C++ converter uses a double cumulative sum and matches the float64
reference to **1.3e-7**. It reads the dumper's raw records plus an explicit
`observed[]` array, converts against the exported trie, and writes only
target rows (no raw logits through Python). A 1k-token end-to-end shard
passes all counts and invariants.

Consequences:

- The 100k shards and the training numbers in `phase5-6.md` are
  preliminary; they used the superseded float32 converter.
- The Python 1M run was stopped at 35% and its partial shards discarded.
- The 1M run is being repeated with `--cpp-targets --ngl 32 --threads 8`.
- `scripts/export_trie.py`, `scripts/build_converter.sh`,
  `scripts/test_converter.py` and `src/byte_agent/teacher.py:stream_targets`
  implement the new path.
