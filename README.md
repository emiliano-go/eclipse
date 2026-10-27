# Eclipse

Byte-level distillation rig for **Ornith 1.5 9B**.

This repository is a feasibility study, not a trained product. The goal is to
prove that the teacher's token distributions can be converted to exact
byte/End-Of-Token targets, that a small byte student learns from them, and that
EOT beats Marginalize-It and byte cross-entropy in a controlled comparison.

## Conclusions

- The rig works end to end. A 48M byte student learns the 9B teacher's
  next-byte distribution: held-out BPB **0.818**, teacher KL **0.317**,
  top-1 **0.865**, valid UTF-8 **1.00** on 36 unseen documents.
- **EOT + 10% CE is the best objective** in the controlled 1M comparison.
  Pure EOT reached 1.362 BPB / 0.617 KL; pure CE 1.483 / 0.826; MI wins
  content-only BPB by discarding EOT mass and is not deployable.
- Scale is the dominant lever: moving from 1M to 26.2M training positions cut
  BPB 36% and KL 45%. The validation curve is flat after ~8k steps.
- **This is not a usable language model.** A 7.3-byte greedy prefix with 86.5%
  teacher-forced top-1 shows that free-running generation still compounds
  error. The result proves the pipeline, not the model.
- Next work is a 24 GB GPU / larger-student path and a generation-focused
  objective; both are recorded in `reports/plan-training-efficiency.md`.

## Scope

- Teacher: `ornith-ai/Ornith-1.5-9B` (Qwen3.5 hybrid, 9B, BF16 available).
- Targets: Q5_K_M bulk, Q8_0 precision audit.
- Student: ~48M dense byte Transformer, 1024-byte context, 260-symbol vocab.
- Corpus: ~20 MB of normalized text and code, staged extraction.
- Hardware: RTX 3070 8 GB. Models, shards and checkpoints live in
  `/data/ornith-byte`; this repository holds code only.

Out of scope: BLT, MLA, MTP, MoE, FP8, serving, instruction tuning, and any
claim of general capability.

## Method

Teacher logits are streamed from `llama.cpp` and converted against a
byte-trie of the 248k-token vocabulary in DFS order, so each prefix subtree is a
contiguous range. EOT targets are exact: byte distributions come from child
subtree masses, and the terminal node receives `p(token) / mass(prefix)` on the
EOT symbol. Marginalize-It is derived from the same 260-dimensional row by
zeroing EOT and renormalizing. Raw logits are never persisted.

## Layout

```
src/byte_agent/   tokenizer audit, trie, corpus, teacher, shards, model, train, eval
tools/            dump_logits.cpp (llama.cpp example)
scripts/          fetch_models.sh, build_llama.sh
configs/          phase_a.yaml, student_48m.yaml
tests/            unit and property tests
/data/ornith-byte/{venv,models,tools,corpus,targets,runs}
```

## Status

Phase 0 complete. Tokenizer audit and 200k-character HF/llama.cpp parity are
exact; the logits dumper is deterministic. Ornith Q5_K_M produced 32,768
teacher tokens at 890 tokens/s with peak 7,687 MiB VRAM at ctx 4096. Real
logits convert to EOT targets with unit row sums.

Phase 1 is complete: a 20.0 MB NFC-normalized code/documentation corpus with
no train/validation overlap, 6,592,718 train tokens and 136,916 validation
tokens. The project decision is raw text without chat framing; the known
consequence is that samples may inherit the teacher's thinking-mode framing.
See `reports/phase0.md` and `reports/phase1.md`.

The extraction pipeline is implemented: document-aware teacher chunks, a
streaming logits dumper, exact EOT/Marginalize-It conversion, and 260-symbol
fp16 shards with explicit `next_symbols`. Stage 1 (100k tokens) produced
457,461 validated positions in 8.4 minutes.

The 48M student overfits 32 windows to BPB 0.331 (EOT KL 0.013) in 85 s.
On 500-step matched runs: EOT BPB 1.200 / KL 0.526 / top-1 0.828;
Marginalize-It BPB 1.166 / KL 0.693 / top-1 0.566; byte-CE BPB 1.196 /
KL 0.672 / top-1 0.843. EOT transfers the teacher distribution best; MI
wins content-only BPB because it has no EOT mass. See
`reports/phase5-6.md`.

A precision audit found the Python converter's float32 cumulative sum can
misallocate rare-prefix targets (up to 0.789 vs a float64 reference). The
new C++ converter (`tools/convert_targets.cpp`) matches the reference to
1.3e-7. The 1M C++ extraction is complete: 1,000,497 teacher tokens -> 4,014,129
positions, 2.087 GB, 431 tokens/s, 38.7 min. The full shard validates
(non-finite 0, negatives 0, unit row sums, exact counts); 5 positions
underflow in fp16. The earlier 100k training numbers are preliminary.

Corrected 1M comparison (2,000 steps, 300,653 validation bytes):

| objective | val BPB | EOT-incl BPB | teacher KL | top-1 |
|---|---:|---:|---:|---:|
| EOT | 0.860 | 0.921 | 0.348 | 0.862 |
| MI | 0.771 | 6.418 | 0.404 | 0.637 |
| CE | 0.872 | 0.937 | 0.488 | 0.865 |

EOT beats CE on BPB and KL and is the only method with usable token
boundaries. MI still wins content-only BPB because it has no EOT mass.
The corrected 1M overfit test reaches BPB 0.342 / KL 0.0156. See
`reports/phase5-6.md`.

### Held-out documents (36 docs, 420,313 bytes scored)

| objective | BPB | EOT-incl BPB | teacher KL | top-1 | greedy prefix |
|---|---:|---:|---:|---:|---:|
| EOT | 1.362 | **1.473** | **0.617** | **0.790** | 2.5 |
| MI | **1.228** | 6.562 | 3.287 | 0.583 | 2.9 |
| byte CE | 1.483 | 1.615 | 0.826 | 0.787 | **14.2** |

The old window-level 0.860 EOT BPB was in-document and optimistic. On
truly held-out documents the ordering MI < EOT < CE on content BPB
survives, EOT beats CE on BPB/KL, and CE generates better exact
continuations. MI still has no EOT mass. See `reports/phase7-heldout.md`.

### Full-data training (EOT + 10% CE)

The 1M sweep selected `0.9 * EOT KL + 0.1 * byte CE`. Training on the full
26.2M-position set for 10,000 steps (~3.1 epochs) produced:

| metric | 1M, 2k steps | full, 10k steps |
|---|---:|---:|
| BPB | 1.276 | **0.818** |
| EOT-inclusive BPB | 1.378 | **0.879** |
| teacher KL | 0.575 | **0.317** |
| top-1 | 0.801 | **0.865** |
| valid UTF-8 | 0.90 | **1.00** |
| greedy prefix | 6.35 | **7.3** |

Documentation reaches BPB 0.764 / KL 0.284; Python 1.143 / 0.525. The
validation curve is flat after ~8k steps. Generation is still weak (7.3-byte
greedy match), so this proves the rig rather than a usable model. No
gpu-unlock was used; the deferred speed protocol is in
`reports/plan-training-efficiency.md`. See `reports/phase7-full.md`.

Next: optional second seed or token-student control, or the 24 GB GPU path
(100-300M student, on-the-fly targets) recorded in
`reports/plan-training-efficiency.md`.

The plan is stored by the `rbg` plan tool under task `ornith-byte-rig`.

## License

MIT. See [LICENSE](LICENSE).
