# Phase 1 report — byte corpus

Date: 2026-10-25.

## Decision: raw text, no chat framing

Ornith is an instruction-tuned thinking model. On a raw Python prefix it
emits `[Start thinking]` and analyses the prompt instead of continuing the
code. Per the project decision, the rig proceeds with **raw text and code,
without the chat template or control symbols**. The student will therefore
learn the teacher's next-token distribution on raw documents, including its
tendency to frame text as a request. This is acceptable for validating the
distillation pipeline and comparing EOT / Marginalize-It / byte-CE; it is
not the final corpus for a usable assistant. Chat framing with
`enable_thinking=false` is deferred.

## Sources

Local, already-present corpora only; no additional downloads:

- `/usr/lib/python3.14` — Python standard library source.
- `/usr/share/doc` — package documentation and READMEs.
- `/usr/share/licenses` — license and legal text.

Files larger than 2 MB were skipped. Text was decoded as strict UTF-8,
NFC-normalized, stripped of teacher special-token literals, exact-deduped by
SHA-256, and near-deduped with a 16-hash MinHash signature plus Jaccard
verification (> 0.9). The split is document-level and targets 2% of bytes.

## Result

| property | value |
|---|---:|
| content bytes | 20,014,479 |
| documents | 1,091 |
| train documents | 1,055 |
| validation documents | 36 |
| train bytes | 19,595,917 |
| validation bytes | 420,740 (2.10%) |
| train/val SHA-256 overlap | 0 |
| rejected exact / near / short | 8 / 11 / 38 |

Teacher tokenization (Ornith tokenizer):

| split | bytes | tokens | bytes/token |
|---|---:|---:|---:|
| train | 19,595,917 | 6,592,718 | 2.972 |
| validation | 420,740 | 136,916 | 3.073 |

The corpus is code- and documentation-heavy, as expected from the local
sources. At the measured 890 teacher tokens/s, extraction is projected at
**~2.1 h** for train plus ~3 min for validation. Dense 260-symbol fp16
targets are projected at **~14 GB** (about 520 B per student position,
including one EOT position per token).

## Artifacts

- `/data/ornith-byte/corpus/{train,val}.{txt,bin}` — normalized streams.
- `/data/ornith-byte/corpus/index.json` — document hashes, sources, split.
- `/data/ornith-byte/corpus/stats.json` — builder counts.
- Copies of the index and stats are in `reports/`.

## Gate status

Phase 1 gate met: no train/validation overlap, NFC-idempotent input,
~20 MB content, token counts and projected extraction cost recorded.
