# Phase 7 — full-data training with EOT + 10% CE

Date: 2026-10-26; prompt-format check added 2026-10-27.

## Setup

- Objective: `0.9 * EOT KL + 0.1 * byte CE` (1M sweep winner).
- Data: `train-full`, 1,055 documents, 6,591,664 teacher tokens,
  26,185,473 positions.
- 10,000 steps x 8 x 1024 = 81.92M positions, ~3.1 epochs.
- Batch 8, seq 1024, LR 3e-4 cosine with 5% warmup, seed 1337.
- Eval every 1,000 steps on `val-full`; checkpoints every 2,000.
- Run: `/data/ornith-byte/runs/full-eot-ce-10`
  (job `bgj-11o5m1zd`).
- Training time: 22:28-23:18 (~50 min), 27.3k tok/s at the end,
  peak 4.97 GiB VRAM. Total job including document eval: 54.5 min.

## Final document-held-out result

36 documents, 420,313 bytes scored, 557,056 positions.

| metric | value |
|---|---:|
| BPB | **0.8181** |
| EOT-inclusive BPB | 0.8790 |
| teacher KL | 0.3170 |
| observed next-symbol top-1 | 0.8649 |
| valid UTF-8 (20 prompts) | 1.00 |
| EOT rate | 0.217 |
| greedy byte accuracy | 0.087 |
| greedy prefix | 7.25 bytes |
| generation speed | 83.5 B/s |

Per domain:

| domain | bytes | BPB | EOT-incl BPB | KL | top-1 |
|---|---:|---:|---:|---:|---:|
| documentation | 360,211 | 0.764 | 0.822 | 0.284 | 0.874 |
| Python source | 60,102 | 1.143 | 1.220 | 0.525 | 0.807 |

## Comparison with the 1M sweep winner

Same objective, same held-out set, 10x more training positions:

| run | BPB | EOT-incl | KL | top-1 | valid UTF-8 | greedy prefix |
|---|---:|---:|---:|---:|---:|---:|
| 1M, 2k steps | 1.2755 | 1.3785 | 0.5751 | 0.8006 | 0.90 | 6.35 |
| full, 10k steps | **0.8181** | **0.8790** | **0.3170** | **0.8649** | **1.00** | **7.25** |

- BPB: -36%
- teacher KL: -45%
- top-1: +6.4 points
- generation validity: 0.90 -> 1.00
- greedy prefix: 6.35 -> 7.25 bytes

The gain is dominated by data/scale, not by a change of objective. The
1M ordering still stands: EOT+CE 0.1 was the best distribution-matching
arm, and it scales.

## Prompt-format check

Training inputs are teacher-token boundaries: `BOS, bytes, EOT, bytes, EOT`.
The original generation test prompted with raw bytes and no EOTs, which is
out of distribution. A corrected prompt builder now tokenizes the held-out
prefix, inserts EOTs after complete tokens, and leaves a token cut by the
64-byte budget without a trailing EOT.

| prompt format | greedy prefix | greedy byte acc | EOT rate | B/s |
|---|---:|---:|---:|---:|
| EOT boundaries (training format) | 7.25 | 0.0870 | 0.217 | 83.5 |
| raw bytes (old test) | 7.30 | 0.0861 | 0.218 | 82.7 |

The two are materially identical. The prompt mismatch was real, but it does
not explain the weak generation; that result stands.

## Validation curve

| step | val BPB | top-1 |
|---:|---:|---:|
| 1,000 | 1.491 | 0.763 |
| 2,000 | 1.059 | 0.827 |
| 3,000 | 0.955 | 0.844 |
| 4,000 | 0.904 | 0.850 |
| 5,000 | 0.863 | 0.856 |
| 6,000 | 0.835 | 0.860 |
| 7,000 | 0.831 | 0.863 |
| 8,000 | 0.818 | 0.864 |
| 9,000 | **0.805** | **0.866** |
| 10,000 | 0.818 | 0.865 |

The last 2,000 steps are within validation noise; the 9,000-step point was
marginally better, but only steps 8,000 and 10,000 were saved. Ten
thousand steps is adequate for this feasibility run.

## Conclusion

- The byte student learns the teacher **much** better with 3.1 epochs than
  with the 1M screening set: BPB 0.82, KL 0.32, top-1 86.5%.
- **EOT + 10% CE remains the right objective** from the sweep. The soft
  teacher distribution supplies most of the transfer; the small CE term
  sharpens the observed symbol.
- Generation is still weak: a 7.25-byte greedy match, with valid UTF-8 but
  not coherent text. This proves the rig, not a usable model.
- Python source remains harder than documentation (1.14 vs 0.76 BPB), but
  the gap narrowed substantially with full data.
- No gpu-unlock was used; the measured bottleneck was not tensor issue,
  and the reboot cost exceeded the expected saving. The deferred protocol
  is in `reports/plan-training-efficiency.md`.

## Known limitations

- The 36 validation documents were used for objective selection and then for
  the final report. They are held out from gradient training, but they are not
  an untouched final test set.
- Teacher context is 4,096 tokens; the student sees 1,024 byte/EOT symbols,
  roughly 770 raw bytes at 2.97 bytes/token. The reported KL is under the
  student's shorter context, not a context-matched comparison.
- `TargetDataset` builds windows within shards and does not use document IDs
  to reset attention; a window can cross a document boundary. The teacher
  resets context per document.
- The reported top-1 is agreement with the observed next symbol, not with the
  teacher's argmax.
- FP16 target storage underflowed 50 observed symbols in the full shards
  (1.91e-06 of positions). Targets are now renormalized on load; this changes
  the eval KL only at the 1e-5 level.
- The 1M -> full comparison changes both training positions and optimization
  steps (2k -> 10k); it does not isolate data from steps, and it does not test
  model size or a token-student architecture.
- The corpus is 20 MB of Python, documentation and licenses, trained as raw
  text with no chat framing. The teacher is instruction-tuned, so this is not
  evidence of assistant behavior. Special tokens are collapsed into EOT and
  there is no trained EOS target.
- There is no token-student baseline and no matched generation-speed,
  prefill, or VRAM comparison against a token student or quantized Ornith.
  The 83.5 B/s figure is the current Python single-symbol generation loop, not
  a fair systems claim.

## Remaining optional work

- A second seed to measure run-to-run noise.
- A token-student control at the same parameter count.
- Longer training only if the validation curve shows a real post-9k gain;
  the current 10k result is flat within noise.
- A 24 GB GPU path: 100-300M student, on-the-fly byte targets, 2k-4k context.
  This is a scale upgrade, not a fix for the generation gap; see
  `reports/plan-training-efficiency.md`.

## Artifacts

- `/data/ornith-byte/runs/full-eot-ce-10/final.pt`
- `/data/ornith-byte/runs/full-eot-ce-10/doc_eval.json` — EOT-prompt eval
- `/data/ornith-byte/runs/full-eot-ce-10/doc_eval_eotprompt.json`
- `/data/ornith-byte/runs/full-eot-ce-10/doc_eval_rawprompt.json`
- `/data/ornith-byte/runs/full-eot-ce-10/train.jsonl`
- `/data/ornith-byte/runs/1m-mix/` — all eight sweep arms and doc evals.
