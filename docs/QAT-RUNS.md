# QAT Distillation of the Fast-AR: Experimental Runs

Technical report; companion to the method description in
[QUANT.md §6](QUANT.md). Hardware: NVIDIA DGX Spark (GB10, `sm_121`).
Trainer: `tools/qat_fastar.py` (torch 2.11, single GPU). Corpus
generation: `tools/qat_corpus.py` against the serving engine with
`S2P_DUMP_FRAMES`. This report is updated as runs complete; open items
are marked as pending rather than omitted.

## 1. Scope

Experimental conditions and results for the runs that train the 0.41 B
fast-AR decoder to group-wise INT4. The metric throughout is held-out
per-step argmax agreement with the BF16 teacher, teacher-forced and
free-running; the bar is the per-channel INT8 deployment quantizer at
0.9650 / 0.8951 (measured on the v1 holdout under identical protocol).
Agreement is a proxy: near-tied codebook decisions are perceptually
equivalent, so the audio gate chain (HF envelope, EOS probe, critical
listening) decides acceptance, not the proxy alone.

## 2. Experimental setup

### 2.1 Corpus

- **v1** — 60,222 frames (≈ 46.6 min of audio): 360 takes, 10 texts
  × 12 registry voices × 3 seeds, seven languages.
- **v2** — extends v1 by appending (the dump hook opens its file in
  append mode, so v1 remains a valid prefix and cached teacher
  trajectories are reused; only the tail is recomputed). Design: all 33
  registry voices, 22 texts across the seven languages, ~30 % long-form
  (three ~80 s multilingual passages), 590 sampled (voice, text, seed)
  combinations, targeting ~270 k frames total. The two demonstration
  texts used for final listening takes are excluded from v2, so
  acceptance material is not drawn from training material. (v1 did
  contain the short multilingual demonstration text; recorded here for
  completeness. The long demonstration text was never in any corpus.)

### 2.2 Training configuration

Batch 256 frames, AdamW, cosine LR schedule, KL loss over the nine
residual steps; DAgger mixing (0.5 of each batch forced with student
free-run prefixes) after a configurable fraction of the schedule.
Holdout: 8,192 frames. For warm-started runs, `--holdout-from` draws the
holdout exclusively from frames the initializing run never trained on.
Best-by-holdout and last checkpoints are written at every evaluation
interval, so a run interrupted at any point yields its best artifact.

Two trainer optimizations were required to make DAgger affordable
(details in the commit history): a chunked 8-candidate MSE scale search
(memory O(8·N·K) instead of O(32·N·K·32), first-minimum tie order
preserved — validated by exact reproduction of a prior evaluation), and
a no-grad quantized-weight cache in the fake-quant linear (a 9-step
greedy rollout would otherwise re-quantize every tensor nine times).
Plain steps are interactive; DAgger steps remain ~4× more expensive —
the sequential rollout is inherent.

### 2.3 Run orchestration

Multi-hour runs proceed unattended: corpus generation, container
shutdown, training, checkpoint patching (`tools/apply_qat_patch.py`),
redeployment, and the gate battery are chained as self-triggering stages
keyed on filesystem and container conditions, with log monitors on
evaluation milestones and error signatures. Runs are therefore
reproducible end-to-end from scripts rather than interactive sessions.

One orchestration defect occurred and was repaired mid-run: the trainer
container writes its exported patch as root with mode 0600, so the
host-side stage guarding the warm-start copy failed silently and
selected its cold-start fallback. The permission was corrected through a
container, the copy re-established, and the post-corpus stages were
replaced by a corrected orchestrator before training started — run 3
therefore warm-starts as designed. Lesson recorded: stage guards must
distinguish "file missing" from "file unreadable".

### 2.4 Thermal conditions

Sustained mixed load (batch inference for corpus generation, then dense
training) is logged at 60 s resolution: GPU temperature, package power,
SM clock, hottest ACPI zone (trip point 104 °C). Full-night summary
(717 samples, ~12 h): GPU maximum 86 °C, hottest zone maximum 97 °C
(coinciding with a load transition that doubled package power,
37 → 87 W), SM clock at its full 2444 MHz throughout the loaded phases
— zero thermal-throttling episodes. The only clock-drop event in the
trace (208 MHz at GPU 57 °C) is the idle transition at training exit,
not throttling.

## 3. Runs

| run | steps | corpus | init | LR | DAgger from | result (free-run) |
| --- | ---: | --- | --- | --- | --- | --- |
| 1 | aborted | v1 | cold | 2e-5 | 50 % | — (trainer too slow; fixed, see §2.2) |
| 2 | 2000 | v1 | cold | 3e-5 | 50 % | see table below |
| 3 | 3000 | v2 | run 2 | 1.5e-5 → 1e-6 | 35 % | **0.6711** (best, step 2500) |

### 3.1 Run 2 (v1 corpus, cold start)

Held-out agreement (teacher-forced / free-run):

| checkpoint | teacher-forced | free-run |
| --- | ---: | ---: |
| INT8 deployment bar | 0.9650 | 0.8951 |
| pre-QAT (untrained INT4-g32) | 0.7025 | 0.4550 |
| step 500 | 0.7918 | 0.5556 |
| step 1000 | 0.8196 | 0.6029 |
| step 1500 (DAgger active) | 0.8414 | 0.6380 |
| step 2000 (final) | 0.8609 | 0.6690 |

The per-interval gain decays (+10.1, +4.7, +3.5, +3.1 points free-run
per 500 steps) while the trainer accumulates epochs over the 52 k-frame
training split — the signature of a data ceiling rather than a schedule
ceiling. This motivated run 3's design: grow the corpus ~4.5× and
continue from run 2's weights instead of extending run 2's schedule.
Run 2 closes at 75 % of the INT8 bar's free-run agreement (0.6690 vs
0.8951); its artifact proceeds through the audio gate battery
regardless, since the audibility threshold on this proxy is unknown.

**Run-2 artifact, audio gates (all-INT4 deployment via
`apply_qat_patch.py` + `S2P_INT4_ALL=1`):**

- *HF envelope*, two 148 s takes on the held-out long text: no muffling
  collapse. deeper-male rises 0.051 → 0.076 (first → last 10 s mean)
  with content-driven variation; neutral-female flat 0.014 → 0.012.
- *EOS probe*, 24 utterance pairs vs INT8: 0 flags; mean length 3.4 s
  vs 3.4 s; both long takes terminate at 148 s, matching the INT8
  reference length.
- Wall RTF during these gates is not meaningful (corpus generation
  shared the GPU by design); the clean measurement follows with the
  final artifact.
- Listening verdict: pending.

The catastrophic failure mode of untrained INT4 (progressive muffling
from ~10 s) is absent at 0.6690 agreement — the audio gates pass well
below the INT8 bar on the proxy metric.

### 3.2 Run 3 (v2 corpus, warm start)

Corpus v2 final size: 308,424 frames (590 v2 takes, 205 min of audio,
generated in 185 min of wall time). Warm start from run 2's exported
patch; holdout 8,192 frames drawn only from v2 frames
(`--holdout-from 60222`); evaluation and best/last checkpoints every
500 steps; cosine floor 1e-6; no wall-clock limit.

| checkpoint | teacher-forced | free-run |
| --- | ---: | ---: |
| INT8 deployment bar (v2 holdout) | 0.9645 | 0.8935 |
| warm start (run 2's weights) | 0.8139 | 0.5959 |
| step 500 | 0.8168 | 0.5999 |
| step 1000 | 0.8271 | 0.6163 |
| step 1500 | 0.8356 | 0.6274 |
| step 2000 | 0.8527 | 0.6540 |
| step 2500 (**best**) | 0.8613 | 0.6711 |
| step 3000 (final) | 0.8615 | 0.6698 |

Two observations. First, run 2's weights score 0.5959 on the v2
holdout against 0.6690 on their own — the v2 holdout contains voices
and texts run 2 never saw, so this quantifies the earlier run's
generalization gap and validates drawing the holdout from unseen
material. Second, the best checkpoint precedes the schedule end
(0.6711 at step 2500 vs 0.6698 at 3000); best-by-holdout selection,
not the final export, supplies the deployed artifact.

## 4. Results

The best artifact (step 2500, free-run 0.6711 = 75 % of the INT8 bar)
was deployed automatically via checkpoint patch and served with
`S2P_INT4=1 S2P_INT4_ALL=1` — the full all-INT4 weight stream
(4.37 GB/frame).

**Audio gates — all pass:**

- *HF envelope*, 144/148 s takes on the held-out long text: no
  muffling collapse; trajectories fluctuate with content (deeper-male
  median 0.046, buckets 0.041–0.135; neutral-female flat
  0.013 → 0.012).
- *EOS probe*, 24 utterance pairs vs INT8 on an otherwise idle GPU:
  0 flags, length ratios 0.88–1.38, mean 3.5 s vs 3.4 s.
- Long takes terminate at 144.1 s / 148.0 s — the INT8 reference class.

**Wall RTF (idle GPU, streamed serving):**

| take | audio | wall | RTF |
| --- | ---: | ---: | ---: |
| long-form, deeper-male voice | 144.1 s | 89.8 s | **0.62** |
| long-form, neutral-female voice | 148.0 s | 92.6 s | **0.63** |
| short multilingual, 3 voices | 48.1–48.8 s | 29.2–30.0 s | **0.60–0.62** |
| zero-shot, ~22 s | 21.6 s | 13.0 s | **0.60** (TTFA 0.14 s) |
| voice reference (warm cache), ~22 s | 22.0 s | 13.4 s | **0.61** (TTFA 0.20 s) |

The INT8-fast-AR configuration measured 0.75–0.76 on the same paths;
the all-INT4 stream delivers the predicted further ~20 % (weight bytes
6.17 → 4.37 GB/frame).

**Listening verdict: ACCEPTED (2026-08-03).** I listened to the five
final takes (two 144/148 s long-form, three 48 s short multilingual)
and accepted the artifact without reservation. The acceptance
calibrates the proxy metric: at 0.6711 free-run agreement — 75 % of
the INT8 bar — the artifact is perceptually clean, confirming that
near-tie codebook divergence well below the bar is inaudible. The
all-INT4 configuration is therefore the serving configuration, and the
README headline numbers advance to 0.60–0.63.

## 5. Reproduction

```sh
# corpus (server running with S2P_DUMP_FRAMES=/path/frames.bin)
python3 tools/qat_corpus.py

# training (torch container; model dir and dump mounted)
python3 tools/qat_fastar.py --model /model --dump /data/frames.bin \
    --out /data/qat --steps 3000 --batch 256 --lr 1.5e-5 --lr-min 1e-6 \
    --eval-every 500 --dagger-start 0.35 \
    --init /data/prev_qat_patch.safetensors --holdout-from 60222

# deployment
python3 tools/apply_qat_patch.py --model /model \
    --patch /data/qat/qat_patch_best.safetensors --out /model-qat
# serve with S2P_INT8=1 S2P_INT4=1 S2P_INT4_ALL=1 against /model-qat
```
