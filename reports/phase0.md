# Phase 0 report — environment, teacher access, parity

Date: 2026-10-25. Hardware: RTX 3070 8 GB (sm_86), Ryzen 7 5800X,
31 GB RAM. CUDA driver 615.78 / UMD 13.4.

## Environment

- Project venv: `/data/ornith-byte/venv`, Python 3.12.
- `torch 2.11.0+cu128`, CUDA 12.8, `torch.cuda.is_available() == True`,
  compute capability `(8, 6)`. Measured 35–37 TFLOP/s fp16 matmul
  after clocks ramp.
- Tests: `pytest tests` passes (7 trie/property tests).
- `uv` installs the self-contained PyTorch cu128 wheel plus the
  `nvidia-*-cu12` runtime packages.

## Pinned artifacts

Teacher `ornith-ai/Ornith-1.5-9B` @ `489cb979…`; GGUF
`ornith-ai/Ornith-1.5-9B-GGUF` @ `abdd624b…`.

| file | bytes | sha256 |
|---|---:|---|
| config.json | 2,910 | `1f1b3751…930a313` |
| tokenizer.json | 12,807,982 | `5f9e4d49…bb81cb42` |
| tokenizer_config.json | 16,710 | `316230d6…a0a64ac8` |
| Ornith-1.5-9B-Q5_K_M.gguf | 6,642,544,576 | `e4d9634a…0fb4247d` |

`Ornith-1.5-9B-Q8_0.gguf` (9,786,060,384 B) is partially downloaded for
the precision audit; it is not needed for the Phase 0 gate.

A development stand-in, `Qwen/Qwen3.5-0.8B` Q8_0, was downloaded to
exercise the pipeline before the 9B model arrived. Its `tokenizer.json`
is byte-identical to Ornith's (`5f9e4d49…`), and its GGUF architecture is
`qwen35` with vocab 248,320.

Toolchain: llama.cpp `d854d98bae598388ecea1ed84396587e5edc326d`,
built with `GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=86`, and the
`dump_logits` example.

## Tokenizer audit

| property | value |
|---|---|
| ids | 248,077 (248,044 base + 33 added) |
| specials | 21 (text, vision, audio/TTS markers) |
| byte tokens | 248,056 |
| distinct byte strings | 248,056 (no duplicates) |
| token length (vocab) | min 1, max 128, mean 7.43 |
| NFC | idempotent |
| 10 MB round-trip | exact |

Non-special added tokens (for example `<tool_call>` and FIM markers) are
literal UTF-8 and are included in the trie. Special tokens have no bytes
and are mapped to boundary (EOT) mass.

## Token-id parity

`dump_logits --tokenize` versus HF `tokenizers` on 200,000 characters
(52,023 tokens):

| model | tokens | mismatches | round-trip |
|---|---:|---:|---|
| Qwen3.5-0.8B Q8_0 | 52,023 | 0 | exact |
| Ornith Q5_K_M | 52,023 | 0 | exact |

On 338 characters the same check also passed with 106 tokens. The dumper
is deterministic: two independent dumps of the same ids produced
byte-identical files.

## Teacher throughput and memory

`dump_logits`, Ornith Q5_K_M, ctx 4096, batch 128, all layers on GPU:

- 32,768 tokens in 36.8 s = **890 tokens/s**.
- Peak VRAM: **7,687 MiB / 8,192 MiB** (tight; batch 64 is the safe
  production setting).
- Chunks of `ctx` tokens reset memory at chunk boundaries but not between
  sub-batches, so each token attends to the preceding tokens of its chunk.
  Verified: with `ctx=16` the logits at position 16 differ by ~18 from the
  full-context run, while `batch` size only changes numerics
  (argmax identical; max |delta| < 0.7).

Extrapolation for the 20 MB corpus (~5.6 M teacher tokens): **~1.75 h**
at 890 tok/s, plus conversion time.

## End-to-end conversion on real teacher logits

512 Ornith tokens (2,094 content bytes) -> 2,605 target rows:

- every row sums to 1.0 (max error 1.2e-7);
- no missing observed-byte mass;
- teacher BPB on content bytes: 0.6027 (the sample is Python source);
- mean EOT probability at token ends: 0.9518.

## Findings that change later phases

1. **Ornith is instruction-tuned and enters a thinking mode.** Raw
   completion of `"The capital of France is"` produced a reasoning trace,
   not plain continuation. Phase B should either use the chat template
   with control symbols or explicitly document that raw-text distillation
   transfers an instruct model's distribution, including its framing.
2. **Marginalize-It needs a shift at token boundaries.** At an EOT row
   with no byte children, dropping EOT leaves no mass. The correct MI
   target for the last byte of a token is the *next* row's root byte
   distribution, not a renormalization of the current EOT row. The EOT
   shard stores both rows sequentially, so the training loader can shift;
   rows at document end are skipped.
3. **VRAM is the binding constraint**, not throughput. Batch 128 leaves
   ~0.5 GB headroom at ctx 4096; use batch 64 for long unattended runs.
4. The teacher of record for bulk targets is Q5_K_M; the Q8 audit remains
   pending.

## Gate status

Phase 0 gate is met: qwen3_5 loads, token-id parity is exact, the dumper
is deterministic, conversion invariants hold on real logits, and the 9B
teacher throughput is measured. Phase 1 (corpus) and Phase 2 (trie
hardening) can proceed.

## Protocol v2 (used by extraction)

- Dumper header is six `uint32` fields: magic, n_tokens, n_vocab,
  first_pos, flags, doc_id.
- `--doc-starts` resets memory at every document boundary;
  `--ngl` controls layer offload.
- Determinism was re-checked after the protocol change: two independent
  dumps of 106 tokens produced byte-identical files.
- Conversion now uses a float32 cumulative sum plus the observed token's
  probability as a prefix-mass floor, clamps negative fp16 artifacts, and
  stores an explicit `next_symbols` array. The stage-1 scan found no
  non-finite rows, no negatives, and unit row sums.

## Protocol v3 (C++ converter)

The dumper now emits `int32 observed[n]` after the input ids (the next
token for each position, `-1` at a document end). The standalone
`convert_targets` reads that field, converts against the exported trie,
and writes target records. The Python raw-logits parser was updated in
lockstep. This removed the Python conversion bottleneck and fixed the
float32 cumulative-sum precision issue: the C++ converter matches a
float64 reference within 1.3e-7.
