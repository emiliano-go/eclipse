# Training efficiency and scaling plan (deferred)

Status: the speed work below is **deferred**. The 1M sweep and the full-data
training are complete; this file records the deferred speed experiments and
the next GPU/scale step. Do not reboot for gpu-unlock without a fresh decision.

## Evidence gathered

- Training already uses FP16 autocast with FP32 accumulation — the path
  gpu-unlock accelerates.
- Measured GPU during the sweep: 100% utilisation, 174–215 W of a 220 W
  limit, 1740–1905 MHz of 2100 max, 74–77 °C. Effective throughput is
  ~6–7 TFLOPS, ~15% of the 3070's dense FP16 tensor peak.
- The workload is power-limiting at times, but the low tensor utilisation
  means issue rate is not the only bottleneck. gpu-unlock's README reports
  1.35× for FP16 on a 3090 Ti at 8192³; small GEMMs will gain less.
- Full training is 10,000 steps × 8192 tokens = 81.92M tokens. At the
  measured ~19.6k tok/s (training-only) that is ~70 min. A 10% speedup
  saves ~7 min; installing gpu-unlock and rebooting costs ~20 min. It is
  net-negative for this run.
- Evaluation is the largest removable overhead: ~15 s per eval on
  val-full. Every-100 eval over 10k steps costs ~25 min; every-1000 costs
  ~2.5 min.
- Boot is Omarchy **UKI** (`/boot/EFI/Linux/omarchy_linux.efi`), cmdline
  comes from `/etc/default/limine`, Secure Boot is disabled, there is no
  SSH, and the single GPU also drives the display. gpu-unlock failure
  recovery would be physical.

## Recommended fast path

1. **Sweep decision.** Keep `eot_ce_10`, `eot_ce_25`, `eot_ce_50`.
   Compare document-held-out BPB/KL/top-1 and generation; choose the
   Pareto point. Prune MI+CE, three-way, T=2, EOT→CE two-stage unless a
   specific hypothesis needs them. Saving: ~1.5–1.8 h.
2. **10-minute benchmark.** 200 steps on `train-1m`, fixed seed, no eval.
   Test one at a time:
   - `torch.compile` (save the unwrapped state dict; keys may carry
     `_orig_mod.`).
   - `AdamW(fused=True)`.
   - `pin_memory=True`, `num_workers=2`, `persistent_workers=True`.
   - `sudo nvidia-smi -pl 250`; revert to 220 W if >82 °C or no gain.
   Keep only >5% stable gains.
3. **Full training.** Winner, from scratch, on `train-full`:
   ```bash
   PYTHONPATH=src /data/ornith-byte/venv/bin/python -m byte_agent.train \
     --targets /data/ornith-byte/targets/train-full \
     --val-targets /data/ornith-byte/targets/val-full \
     --out /data/ornith-byte/runs/full-<winner> \
     --objective mix --alpha-kl <kl> --alpha-ce <ce> \
     --steps 10000 --batch 8 --seq-len 1024 \
     --eval-every 1000 --save-every 2000 --seed 1337
   ```
   Then full document eval:
   ```bash
   PYTHONPATH=src /data/ornith-byte/venv/bin/python scripts/eval_model.py \
     --checkpoint /data/ornith-byte/runs/full-<winner>/final.pt \
     --val-targets /data/ornith-byte/targets/val-full \
     --out /data/ornith-byte/runs/full-<winner>/doc_eval.json
   ```
4. **Optional gpu-unlock A/B — only for future long runs.** Recommended
   skip for the current ~70 min run. If pursued later:
   - Baseline 200 steps; record tokens/s, power, clocks.
   - Back up `/etc/default/limine` and the current UKI; create a bootable
     fallback entry without the flag.
   - Install with the repo's Arch target (`make install-arch` or
     `make install`), append `gpu_unlock.targets=all` to
     `KERNEL_CMDLINE[default]`, regenerate the UKI/initramfs, reboot.
   - Verify `journalctl -k -b | grep gpu_unlock`; rerun the identical
     200-step benchmark.
   - Keep only if >10% faster and stable; otherwise remove the flag and
     package, regenerate, reboot. Do this only when a run is long enough
     (>3 h) for the saving to exceed the install cost.

## ETA

Training itself, current code with eval every 1000:

- **~70–80 min** (81.92M tokens at ~19.6k tok/s + 10 evals + checkpoints).
- **~55–65 min** if the benchmark finds a 10–20% stable gain.

If the pruned sweep ends ~18:15 and training starts then:

- current code: finish **~19:30**;
- optimized: finish **~19:10**.

If the full 8-arm sweep is allowed to finish (~20:00), training finishes
~21:15. The sweep is detached; it will keep running through all arms
unless stopped after `eot_ce_50`.

## Acceptance

- Full run completes in ≤80 min without optimization, ≤65 min with
  optimizations.
- Final document-held-out BPB ≤ the best 1M sweep arm (currently 1.2755
  for `eot_ce_10`), validation UTF-8 rate 1.0.
- No boot/GPU changes unless gpu-unlock is explicitly approved, and no
  changes that cannot be reverted.


## Next step: 24 GB GPU and a larger student

The current 48M model is data-limited: held-out BPB plateaued at 0.805-0.818
after ~8k steps, and it was trained on only 20 MB / 26M positions. A bigger
model on the same data will overfit. The useful upgrade is a 24 GB RTX 3090,
primarily because it changes the data pipeline, not just the model size.

### What a 3090 changes

- **Teacher and student coexist.** The 9B Q5 teacher is 6.6 GB (Q8 9.8 GB);
  a 100M student training is ~6-8 GB. Both fit on 24 GB, so extraction and
  training stop being mutually exclusive.
- **On-the-fly targets.** Stream the teacher's 260-dim byte targets to the
  student and skip the 13.6 GB of targets per 20 MB of text. This removes the
  disk wall; teacher throughput becomes the limit.
- **Bigger student and context.** 100-300M parameters with gradient
  checkpointing, 2k-4k byte context. The 8 GB 3070 stops at 48-100M.
- **Q8 teacher.** The 9.8 GB Q8 model fits alongside a smaller student, so the
  precision audit becomes practical.
- The 3090 is roughly 2x the 3070's bandwidth and FP16 tensor throughput.

### What it does not fix

- **Generation.** The 7.3-byte greedy prefix is compounding error; a bigger
  one-step model may not produce coherent free-running text. Scheduled
  sampling or a sequence-level objective is still required.
- **Data scale.** The teacher generates ~1-2k tokens/s. A 100 MB corpus is
  ~6-12 h; 1 GB is days. On-the-fly removes storage, not teacher time.
- **Context and RAM.** 24 GB gives a few thousand bytes of context. The
  current 31 GB RAM already shows 27 GB used and 20 GB swap; 64 GB is advisable.

### Hardware constraints on this machine

- Board: ASRock B550M Pro SE. One PCIe 4.0 x16 plus a second PCIe 3.0 x16
  (likely x4 electrical) and an x1 slot. Verify the manual and physical
  clearance; x4 is fine for teacher inference and target streaming.
- PSU is the gating unknown: 3090 (350 W) + 3070 (220 W) + 5800X + rest is
  ~750-800 W. Use a quality 850 W minimum, 1000 W for 3090 transients; check
  the 8-pin cables.
- CPU currently reads 86.5 C; a 350 W card in a micro-ATX case needs airflow.
- A **single 3090 replacing the 3070** is simpler than two GPUs: no second
  slot/PSU/clearance issues, and the teacher plus a 100M student still fit on
  24 GB. Keep the 3070 only if you need concurrent teacher/student.

### Phases

1. **Validate hardware:** PSU model/wattage, second-slot electrical width,
   case clearance, CPU cooling, RAM capacity.
2. **Capacity ablation on existing data:** train 100M and ~200M on the current
   26M positions, 10k steps, same document eval. Cost ~2-4 h. If held-out BPB
   and generation do not improve, capacity is not the lever.
3. **Streaming targets:** teacher on one GPU, student on the other; reuse
   `dump_logits` and `convert_targets` in a queue instead of writing shards.
   Or overlap offline extraction on one GPU with training on the other.
4. **Domain-scale run:** expand the corpus to ~100 MB, stream targets, train
   100-300M at 2k-byte context, evaluate held-out BPB/KL/generation.
5. **Generation experiment:** scheduled sampling or sequence-level loss on the
   best checkpoint.

Decision rule: do not buy/train the larger model until the capacity ablation
shows a held-out gain, and do not scale architecture before data.
