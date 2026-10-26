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

Ornith behaves as an instruction-tuned thinking model on raw text, so the
Phase 1 corpus needs an explicit decision about chat framing. See
`reports/phase0.md`. Next: Phase 1 corpus and Phase 2 conversion hardening.

The plan is stored by the `rbg` plan tool under task `ornith-byte-rig`.
