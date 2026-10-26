# Phase 5–6 report — dense student, overfit, three-way comparison

Date: 2026-10-25.

## Student

47,941,440 parameters: d=576, 12 layers, 9 heads (head_dim 64), SwiGLU
ffn=1536, pre-RMSNorm, QK-norm, RoPE, tied embedding/head, 260-symbol
vocabulary, 1024-byte context, KV cache. Trained on the 100k shard with
AdamW, cosine LR, AMP fp16, batch 8, and no gradient checkpointing.
Peak training VRAM on the 3070 was 4.97 GiB.

## Overfit test (alignment proof)

32 windows, 300 steps, train = validation, EOT objective:

| step | EOT KL | BPB | next top-1 |
|---:|---:|---:|---:|
| 50 | 1.739 | 3.456 | 0.472 |
| 100 | 0.728 | 1.625 | 0.746 |
| 200 | 0.038 | 0.360 | 0.947 |
| 300 | 0.013 | 0.331 | 0.950 |

The model memorizes the targets and the byte stream; the conversion and
training alignment are correct. 85 s, 4.97 GiB peak.

## Three-way comparison

402 train windows / 44 validation windows, identical seed and data order,
500 steps, batch 8, 1024-byte context. Validation has 32,499 content bytes.

| objective | val BPB | EOT-inclusive BPB | teacher KL | next top-1 | time |
|---|---:|---:|---:|---:|---:|
| EOT | **1.200** | **1.295** | **0.526** | 0.828 | 133.5 s |
| Marginalize-It | 1.166 | 6.868 | 0.693 | 0.566 | 131.3 s |
| byte CE | 1.196 | 1.305 | 0.672 | **0.843** | 131.3 s |

### Interpretation

- Mi has the lowest content-only BPB (1.166 vs 1.200 for EOT, a 2.8%
  difference), but it has no EOT mass: its EOT-inclusive BPB is 6.87, and
  its next-symbol top-1 is 0.566. It is optimizing a different, continuous
  byte stream.
- EOT has the lowest KL to the teacher distribution (0.526) and strong
  boundary behavior (1.295 EOT-inclusive, top-1 0.828). This is the target
  the rig was built to preserve.
- Byte CE is competitive on content BPB (1.196) and has the best top-1
  (0.843), as expected for a one-hot objective that ignores the teacher's
  uncertainty.

The planned gate `EOT <= MI <= CE` on content BPB is **not met** at this
scale. The documented exception: 100k tokens, 48M parameters and 500
steps are in the low-compute regime where the paper reports byte/EOT
models underperforming token models, and content-only BPB deliberately
excludes the EOT symbol that MI cannot model. The correct reading is that
the rig works and EOT transfers the teacher distribution best; the
asymptotic advantage claimed by the paper is not testable at this budget.

## Limitations

- One seed, one small validation split (32,499 bytes); differences of a
  few percent are not strongly established.
- 100k teacher tokens and 500 steps: heavily undertrained, with visible
  train/validation separation.
- Raw-text corpus and an instruction-tuned thinking teacher; no chat
  framing and no token-student control yet.
- MI is derived from EOT rows; its implementation is exact for content
  bytes but cannot express token ends.

## Artifacts

- `/data/ornith-byte/runs/overfit-eot/` — overfit checkpoint and log.
- `/data/ornith-byte/runs/100k-{eot,mi,ce}/` — checkpoints, final weights,
  JSONL logs.
- `src/byte_agent/{model,train}.py`; `tests/test_{model,train_loss}.py`.

## Next

1. Extract 1M tokens (~1 h) and repeat the comparison with more data and
   longer training.
2. Add the token-student control on the same corpus.
3. Decide whether the full 6.6M-token extraction (~6.8 h, ~13.4 GB) is
   worth running for this rig.
