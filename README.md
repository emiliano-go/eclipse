# Eclipse

Byte-level distillation rig for **Ornith 1.5 9B**.

This repository is a feasibility study, not a trained product. The goal is to
prove that the teacher's token distributions can be converted to exact
byte/End-Of-Token targets, that a small byte student learns from them, and that
EOT beats Marginalize-It and byte cross-entropy in a controlled comparison.

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

Next: the 2,000-step three-way comparison on the corrected 1M targets,
then the full-corpus decision.

The plan is stored by the `rbg` plan tool under task `ornith-byte-rig`.
