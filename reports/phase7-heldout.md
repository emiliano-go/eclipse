# Phase 7 — document-held-out evaluation (1M checkpoints)

Date: 2026-10-26.

## Why this run matters

The 1M comparison used a random 10% of windows from the same 231 training
documents. Adjacent windows leak context. This evaluation uses the corpus
`val` split: **36 documents never used for training**.

- 136,881 teacher tokens
- 420,670 content bytes
- 557,551 target positions (557,056 covered by full windows; 99.9%)
- 420,313 content bytes scored

## Overall results

| objective | BPB | EOT-incl BPB | teacher KL | top-1 | valid UTF-8 | EOT rate | greedy bytes | greedy acc | B/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EOT | 1.362 | **1.473** | **0.617** | **0.790** | 1.00 | 0.202 | 2.5 | 0.061 | 93 |
| MI | **1.228** | 6.562 | 3.287 | 0.583 | 1.00 | 0.000 | 2.9 | 0.100 | 125 |
| byte CE | 1.483 | 1.615 | 0.826 | 0.787 | 1.00 | 0.184 | **14.2** | **0.151** | 106 |

## Per-domain results

| domain | bytes | EOT BPB | EOT KL | MI BPB | MI KL | CE BPB | CE KL |
|---|---:|---:|---:|---:|---:|---:|---:|
| documentation | 360,211 | 1.280 | 0.564 | 1.145 | 3.329 | 1.389 | 0.786 |
| Python source | 60,102 | 1.850 | 0.950 | 1.728 | 3.018 | 2.047 | 1.080 |

The val split happened to contain documentation and Python documents only;
no license document landed in it.

## Findings

1. **The window-level validation was optimistic.** EOT went from 0.860 BPB
   on in-document windows to **1.362** on held-out documents. The previous
   absolute numbers should not be used as generalization results.
2. **The content-BPB ordering survives**: MI < EOT < CE on held-out
   documents. MI leads by 0.134 BPB over EOT, a 9.8% relative advantage.
3. **EOT beats byte CE on both BPB and teacher KL** (1.362 vs 1.483;
   0.617 vs 0.826). This is the first held-out evidence that preserving the
   teacher distribution is better than one-hot training.
4. **MI still cannot model token ends.** Its EOT-inclusive BPB is 6.562,
   its EOT rate is zero, and its full-distribution KL is 3.287 because it
   assigns near-zero mass to EOT. It is a content-byte diagnostic, not a
   deployable objective.
5. **CE generates better exact continuations.** Greedy continuation
   prefix: CE 14.2 bytes, MI 2.9, EOT 2.5. All models produce valid UTF-8.
   This is expected: CE directly optimizes the observed symbol, while EOT
   spends probability on the teacher's uncertainty and boundaries.
6. **Python documents are harder than documentation** for every objective
   (EOT 1.850 vs 1.280 BPB), despite the corpus being Python-heavy. The
   held-out Python documents are genuinely different from the training
   files.
7. The models remain weak for generation (greedy prefix 2–14 bytes). More
   data and training are needed before drawing assistant-quality
   conclusions.

## Implication for the mixed loss

The held-out results sharpen the case for **EOT + CE**:

- EOT supplies the lowest teacher KL, valid EOT mass, and the better
  likelihood;
- CE supplies the better greedy continuation and top-1;
- MI is useful as a content-BPB ablation but not as the final objective.

The 1M mixed-loss sweep is the next test: EOT+CE at 0.1/0.25/0.5 CE,
MI+CE, a three-way point, T=2, and an EOT→CE two-stage schedule.

## Artifacts

- `/data/ornith-byte/targets/val-full/` — held-out target shards (0.29 GB).
- `/data/ornith-byte/runs/1m/{eot,mi,ce}/final.pt` and `doc_eval.json`.
- Copies of the evaluation JSONs are in `reports/doc_eval_1m_*.json`.
- `scripts/eval_model.py` — document/per-domain/generation evaluator.

## Status

The full train extraction was restarted detached after the previous job
was lost by the background-job server at 4.7 GB (36% of the corpus). It is
running as `bgj-eq46ahcx`, log at
`~/.local/share/opencode-bgj/runs/bgj-eq46ahcx.log`. Expected ~4.25 h at
the idle-machine rate. The mixed-loss sweep follows after extraction.
