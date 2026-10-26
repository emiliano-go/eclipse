# Phase 5–6 report — dense student, overfit, three-way comparison

Date: 2026-10-26.

> **Precision note:** the 100k numbers below used the superseded Python
> float32 converter, which can misallocate rare-prefix targets by up to
> 0.789. They are kept only as a preliminary scaling point. The corrected
> 1M results use the C++ double-cumsum converter and are the primary
> result.

## Student

47,941,440 parameters: d=576, 12 layers, 9 heads (head_dim 64), SwiGLU
ffn=1536, pre-RMSNorm, QK-norm, RoPE, tied embedding/head, 260-symbol
vocabulary, 1024-byte context, KV cache. AdamW, cosine LR, AMP fp16,
batch 8, no gradient checkpointing. Peak training VRAM: 4.97 GiB.

## Overfit tests (alignment proof)

| targets | steps | BPB | EOT KL | top-1 | time |
|---|---:|---:|---:|---:|---:|
| 100k, preliminary Python converter | 300 | 0.331 | 0.013 | 0.950 | 85 s |
| 1M, corrected C++ converter | 300 | 0.342 | 0.0156 | 0.9445 | 83 s |

Both memorize the target stream. The corrected-target run confirms that
the C++ shards, symbols, `next_symbols`, and target rows are aligned.

## Preliminary 100k comparison

402 train / 44 validation windows, 500 steps, batch 8, 32,499 validation
bytes. Targets came from the superseded float32 converter.

| objective | val BPB | EOT-incl BPB | teacher KL | top-1 |
|---|---:|---:|---:|---:|
| EOT | 1.200 | 1.295 | 0.526 | 0.828 |
| Marginalize-It | 1.166 | 6.868 | 0.693 | 0.566 |
| byte CE | 1.196 | 1.305 | 0.672 | 0.843 |

## Corrected 1M comparison (primary)

3,527 train / 392 validation windows, 2,000 steps, batch 8, 1024-byte
context, 300,653 validation bytes, identical seed and data order.

| objective | val BPB | EOT-incl BPB | teacher KL | next top-1 | time |
|---|---:|---:|---:|---:|---:|
| EOT | 0.860 | **0.921** | **0.348** | 0.862 | 834.0 s |
| Marginalize-It | **0.771** | 6.418 | 0.404 | 0.637 | 753.0 s |
| byte CE | 0.872 | 0.937 | 0.488 | **0.865** | 639.9 s |

### Interpretation

- **EOT now beats byte CE** on both content BPB (0.860 vs 0.872) and
  teacher KL (0.348 vs 0.488), while CE has a marginal top-1 lead
  (0.865 vs 0.862). EOT also beats CE on EOT-inclusive BPB.
- **Marginalize-It still has the best content-only BPB** (0.771), but it
  has no EOT mass: its EOT-inclusive BPB is 6.418 and its top-1 is 0.637.
  It is optimizing a continuous byte stream, not the teacher's token
  distribution.
- EOT has the lowest KL to the teacher, i.e. it reproduces the teacher's
  probability distribution most faithfully. That is the property the rig
  was built to preserve.
- Scaling from 100k to 1M improved all three objectives by roughly
  1.4x in BPB. The planned gate `EOT <= MI <= CE` on content BPB is still
  not met: MI leads content BPB by 0.089, while EOT leads KL by 0.056 and
  is the only method with usable boundary behavior. The paper's
  asymptotic EOT advantage remains untested at this budget.

## Limitations

- 1M of the 6.59M available teacher tokens (~15%), 2,000 steps (~4
  epochs), one seed.
- Validation is 300,653 bytes; small BPB differences are not strongly
  established.
- Raw-text corpus and an instruction-tuned thinking teacher; no chat
  framing yet.
- No token-student control yet.
- MI is derived from EOT rows; exact for content bytes, but it cannot
  express token ends.

## Artifacts

- `/data/ornith-byte/runs/overfit-1m/` — corrected-target overfit.
- `/data/ornith-byte/runs/1m/{eot,mi,ce}/` — checkpoints, final weights,
  JSONL logs.
- `/data/ornith-byte/runs/100k-{eot,mi,ce}/` — preliminary runs.
- `src/byte_agent/{model,train}.py`; tests for the model and losses.

## Next

1. Full 6.59M-token extraction: ~4.25 h, ~13.8 GB at the measured rate.
2. Add the token-student control at 1M.
3. Q8 teacher precision audit.
4. Only then consider BLT / DeepSeek architecture branches.
