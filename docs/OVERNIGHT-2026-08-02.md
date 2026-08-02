# The overnight run — an engineering journal

*2026-08-02 → 2026-08-03. This document is the as-it-happened record of an
autonomous overnight training-and-deployment run: the project owner went to
sleep mid-experiment and the agent operating this repository designed,
armed, supervised and gated the entire pipeline alone. QUANT.md and
DAC-KERNELS.md hold the measurements; this file holds the story. Times are
CEST, taken from the machine logs.*

## Where the system stood at nightfall

By Saturday evening the native server had reached the numbers the project
had been chasing for weeks: wall RTF 0.75–0.76 (zero-shot, 60 s voice
reference, and ~147 s chunked long-form alike), time-to-first-audio 0.23 s
with the per-voice KV-prefix cache, and a weight ladder of INT4-g32
backbone + INT8 fast-AR/lm-head whose parity sat in the INT8 class
(docs/QUANT.md). One stream was still unconquered: the 0.41 B fast-AR
decoder — read nine times per frame, the largest single bandwidth consumer
— provably cannot go to 4 bits untrained (argmax agreement collapses
8/9 → 2/9). The QAT self-distillation attacking exactly that was mid-run:
2000 steps, with held-out free-run agreement at 0.4550 → 0.5556 → 0.6029
by step 1000, visibly flattening.

## The decision (≈22:30)

The owner asked the obvious question: *there is no deadline — why not just
let it train until it is actually perfect, all night if needed?* The honest
answer, which shaped everything below: simply running longer would not do
it. The gain curve was flattening while epochs piled up on a 52k-frame
training split — the classic signature of a data ceiling, not a schedule
ceiling. And "perfect" needed defining: the BF16 teacher's own INT8
deployment quantizer only agrees with the teacher on 89.5 % of free-run
decisions, because near-tied codebook picks are perceptually equivalent.
So the target is INT8-class held-out agreement *or* passing the audio gate
battery — and the owner's ear stays the final gate either way.

Two levers, applied together overnight:

1. **Corpus v2** — grow the training dump ~4.5× (60k → ~270k frames):
   all 33 registry voices, 22 texts across the seven languages, ~30 %
   long-form. The dump hook appends, so the old frames stay a valid
   prefix and cached teacher trajectories are reused. The two demo texts
   used for the final listening takes were deliberately excluded — the
   deliverables must not be generated from training material.
2. **Warm-started long run** — continue from run 2's weights instead of
   restarting; hold out only frames the earlier run never saw; write
   best-by-holdout and last checkpoints at every eval so a wall-clock
   cutoff loses nothing.

## The pipeline (armed 22:55–23:05)

Five stages, each a self-waiting script on the box — the chain triggers
itself on filesystem and container conditions, so it survives operator
disconnects entirely:

| stage | trigger | does | log |
| --- | --- | --- | --- |
| run 2 (already running) | — | 2000-step QAT, DAgger second half | qat_run2.log |
| night2.sh | qat-train container exits | preserve run-2 patch → corpus v2 (~590 takes) → stop all serving containers → launch warm-started 3000-step run | night2.log, corpus2.log, qat_night.log |
| phase_b.sh | run-2 patch file appears | patch a checkpoint copy, serve on :8012 with S2P_INT4_ALL=1, envelope + EOS gates | phase_b.log, gate-v1/ |
| phase_e.sh | TRAIN2-EXIT in night2.log | pick best checkpoint (best → final → run-2), deploy final server on :8011 (no dump), full gate battery, the three final listening takes | phase_e.log, final/ |
| cutoff guard | 06:45 wall clock | warn; hard-stop the trainer at 07:00 if still running — lossless thanks to the checkpoints | monitor event |

Monitors watch every log for eval milestones, corpus progress and error
signatures; each event wakes the agent, which verifies state and course-
corrects. Phase B runs deliberately *concurrent* with corpus generation —
its envelope/EOS gates are content-based and tolerate GPU contention; its
RTF prints are marked informational for exactly that reason.

## Thermal telemetry

A 12-hour sustained mixed load (batch inference for corpus generation,
then dense training) on a GB10 desktop box is itself worth recording.
A 60 s CSV logger (`/tmp/thermal.log`: GPU temp, power, SM clock, hottest
ACPI zone) runs all night with alert thresholds at GPU ≥ 88 °C / zone
≥ 95 °C. At arming time, ~2.5 h into continuous training: GPU 76–77 °C at
a full 2444 MHz SM clock (no throttling), hottest board zone 85–88 °C.

## Timeline

- **22:46** — run 2 at step 1200, DAgger phase, measured ~9.6 s/step
  (rollouts are ~4× the plain-step cost; expected and accepted).
- **22:55** — night2.sh armed (waits for the trainer container to exit).
- **22:59** — phase_b.sh armed (waits for the run-2 patch file).
- **23:01** — phase_e.sh armed (waits for TRAIN2-EXIT); 07:00 cutoff
  guard armed.
- **23:05** — trainer gained warm start, best/last checkpoints,
  incremental teacher cache (commit `ec886d2`); overnight protocol and
  corpus generator documented and pushed (`13a9424`).
- **23:15** — thermal logger running: GPU 76 °C / 2444 MHz, zones ≤ 88 °C.
- *(appended as the night unfolds)*

## Results

*Recorded after the run completes.*
