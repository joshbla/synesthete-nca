# Local Compute Envelope

## Target Machine

The baseline must train and render on this machine:

```text
Machine:        MacBook Pro
SoC:            Apple M5
CPU:            10 cores (4 performance, 6 efficiency)
GPU:            10 integrated cores
Memory:         16 GB unified
Graphics API:   Metal 4
ML backend:     PyTorch 2.11 MPS, confirmed with the actual NCA graph
```

Unified memory is shared by macOS, applications, CPU tensors, GPU tensors,
framework caches, command buffers, and the model. It is not equivalent to 16 GB
of dedicated VRAM.

There is no fixed safe allocation independent of the rest of the system. For
planning, the first implementation should target a measured peak well below the
physical limit, roughly in the 8-10 GB range on an otherwise quiet machine.
That range is a safety target, not a claim about what MPS will always provide.
The benchmark protocol later in this document replaces it with actual numbers.

## What Determines Training Memory

### Parameters are not the expected bottleneck

For fp32 training with AdamW, a useful static estimate is:

```text
weight                4 bytes per parameter
gradient              4 bytes per parameter
Adam first moment      4 bytes per parameter
Adam second moment     4 bytes per parameter
--------------------------------------------
static training state approximately 16 bytes per parameter
```

This excludes activations, temporary operator buffers, framework overhead, and
possible master copies under mixed precision.

The planned NCA update network should have tens of thousands of parameters, not
millions. Even 100,000 parameters would require only about 1.6 MB for this
static fp32 training state. Parameter count is therefore almost irrelevant to
whether the baseline fits.

### Recurrent activations are the real cost

One fp32 NCA state at the proposed size is:

```text
96 x 96 x 16 x 4 bytes = 589,824 bytes = 0.56 MiB
```

That is small, but training stores intermediate activations for every unrolled
NCA step. A typical update also materializes:

- the current state;
- four perception responses per channel;
- the update network's hidden features;
- audio-conditioned modulation values;
- the stochastic fire mask;
- tensors needed by autograd for the backward pass.

If perception produces 64 channels and the update network uses 96 hidden
channels, a single sample can use several MiB of saved activations per NCA step.
The approximate scaling is:

```text
activation memory proportional to
batch x height x width x hidden width x unrolled steps
```

The exact constant depends on implementation and MPS operator behavior. It must
be measured.

### Spatial scaling is quadratic

At fixed channel count:

```text
64 x 64   ->  4,096 cells
96 x 96   ->  9,216 cells  = 2.25x the 64 grid
128 x 128 -> 16,384 cells  = 1.78x the 96 grid
256 x 256 -> 65,536 cells  = 7.11x the 96 grid
```

Doubling side length approximately quadruples per-step state and convolution
work. Resolution should not be increased until the 96-pixel system has useful
dynamics.

### Temporal unrolling is linear in memory

Inference can roll the same state indefinitely while retaining only the current
state. Training through a rollout is different: ordinary backpropagation
through time (BPTT) stores the graph for each step.

If a loss is applied after 32 updates and gradients flow through all 32, peak
activation memory is approximately four times the same model unrolled for eight
updates. This makes rollout length, not parameter count, the likely first limit.

Possible mitigations, in preferred order after measurement:

1. Use shorter randomized training rollouts while testing long inference
   stability separately.
2. Lower batch size.
3. Use gradient checkpointing across groups of NCA steps.
4. Use truncated BPTT while carrying detached persistent states between
   optimization windows.
5. Reduce spatial resolution.
6. Reduce hidden or state channels.
7. Evaluate mixed precision only after the fp32 baseline is numerically stable.

Truncated BPTT changes the learning problem and should not be treated as a free
optimization. It limits how far backward the objective can assign credit.

## Compute Cost

### NCA inference

Inference applies a few small convolutions to one persistent grid. There is no
VAE decode per frame and no 15-100-step denoising process for every output
frame. If four to eight NCA updates produce one rendered frame, a one-minute
video at 15 fps requires:

```text
60 seconds x 15 frames/second x 4-8 updates/frame
= 3,600-7,200 small NCA updates
```

Published systems establish that comparable NCAs can render in real time on
consumer hardware. This makes real-time inference plausible, but the local
implementation still needs measurement. Framework and tensor-layout choices can
dominate a model this small.

### NCA training

A training step runs many NCA updates and then backpropagates through them. Its
cost grows roughly linearly with:

- batch size;
- number of unrolled updates;
- grid cells;
- update-network hidden width;
- number and cost of evaluation losses.

The baseline should avoid a heavy pretrained visual network in the inner loop.
VGG-style texture and optical-flow losses made prior NCA work effective, but on
this machine their activations could cost far more than the NCA itself. Initial
losses should use inexpensive spatial statistics, motion measurements, target
fields, or short differentiable trajectory comparisons. A perceptual network
should be introduced only to address a measured deficiency.

## Empirical Anchors From the Previous Projects

### Baby Synesthete

The artifact sequence indicates that its 2,156,099-parameter `64 x 64` pixel
DDPM trained for 15,000 optimizer steps in approximately 66 minutes. That is
about 0.26 seconds per step including periodic sampling overhead averaged over
the run.

This is a useful order-of-magnitude anchor, not a direct NCA prediction. The
baby U-Net performed one forward/backward denoising step per optimizer update;
the NCA will perform a recurrent rollout per optimizer update. Conversely, the
NCA has vastly fewer parameters and much narrower layers.

### Original Synesthete

The 10,232,990-value latent diffusion transformer and 1,514,442-value VAE fit
locally because the transformer operated on only 64 latent tokens. Memory was
manageable, but inference was dominated by sequential frame and diffusion-step
passes. At classifier-free guidance scale, a six-second clip could require up
to 2,700 transformer forwards.

The lesson is that model size and memory fit are poor proxies for iteration
speed. The new baseline is chosen to reduce the number and cost of serial
operations, not merely to reduce parameters.

## Data Pipeline

The initial NCA does not need a large stored dataset. Audio features can be
precomputed for a small set of synthetic probes and real clips. Seed states can
be generated cheaply. This should keep the GPU fed without reproducing the
original project's per-sample audio synthesis, FFT, visual rendering, and VAE
encoding pipeline.

Initial implementation guidance:

- Precompute audio features before training.
- Keep the complete probe set in memory when practical.
- Store feature schema and normalization metadata with every run.
- Generate stochastic state damage or seed variation in vectorized tensors.
- Do not introduce multiple data-loader workers until profiling shows a CPU
  bottleneck.
- Watch for MPS operations that silently execute inefficiently.

Because the NCA is small, Python dispatch and framework overhead may matter more
than raw arithmetic. Fusing the recurrent update into a compiled graph may later
be useful, but the first version should prioritize inspectability.

## Provisional Experiment Budgets

These are design targets, not measured promises.

### Smoke: 1-3 minutes

Purpose:

- construct the real model on MPS;
- execute forward, rollout, loss, backward, and optimizer steps;
- detect unsupported operations or CPU fallback;
- measure first-step compile time and steady-state step time;
- confirm bounded state and finite gradients.

The smoke run is not expected to produce an aesthetic result.

### Rapid comparison: no more than 10 minutes

Purpose:

- reject unstable update rules;
- compare short rollout lengths;
- test one conditioning or loss change;
- produce a small fixed-seed preview;
- determine whether the intended metric begins to move.

The number of optimizer steps cannot be specified honestly until the real graph
is benchmarked. A useful rapid run is defined by evidence, not a target step
count.

### Meaningful local run: 1-2 hours

Purpose:

- train one candidate long enough to assess coherent dynamics;
- render all fixed counterfactual probes;
- test long-rollout stability beyond the training horizon;
- compare multiple seeds;
- decide whether a specific next experiment is justified.

The first full run should checkpoint frequently enough that an interrupted or
memory-degraded process can resume without losing most of the budget.

## Benchmark Protocol

The first implementation milestone is a benchmark, not training quality.

### 1. Record environment

Record:

- macOS version;
- PyTorch version;
- MPS availability;
- default dtype;
- active memory pressure before launch;
- model parameter count and tensor shapes.

### 2. Warm up

At `96 x 96`, 16 state channels, batch 1, and one NCA update:

- time model construction;
- time the first compiled step separately;
- time at least 20 steady-state updates;
- record process memory and MPS allocated memory where available.

### 3. Scale one axis at a time

Measure the full training step for:

```text
rollout:  1, 4, 8, 16, 32
batch:    1, 2, 4, 8 as memory permits
grid:     64, 96, 128
```

Do not change more than one axis in each comparison. Record seconds per
optimizer step, peak memory, and whether memory returns to a stable level after
each step.

### 4. Test long inference

Roll one state for at least 10,000 NCA updates without gradients. Measure:

- updates per second;
- memory drift;
- NaN/Inf incidence;
- min, max, mean, and variance of every state channel;
- whether the visible channel dies, freezes, saturates, or explodes.

Long inference stability is not implied by short training stability.

### 5. Test precision

Only after fp32 is stable, compare bf16 or fp16 if MPS supports every required
operation. Record speed, peak memory, gradient behavior, and long-rollout state
drift. Mixed precision is accepted only if behavior remains equivalent under
the fixed probes.

### 6. Replace estimates

After a real 10-minute run, update this document with:

- measured steady-state seconds per optimizer step;
- optimizer steps completed in 10 minutes;
- measured peak memory;
- maximum practical batch and rollout;
- projected and later measured two-hour throughput;
- render speed at the selected updates per frame.

Until then, every runtime number for the NCA remains provisional.

## Measured Stage 0 Baseline

The full Stage 0 benchmark used batch 1, a `96 x 96 x 16` fp32 state, 96 hidden
update channels, circular perception, a 0.5 fire rate, a 0.1 update step, and a
simple terminal mean-square loss with AdamW. MPS fallback was disabled. These
measurements describe the NCA graph only; future data preparation, teacher
generation, richer losses, evaluation, checkpoint I/O, and rendering are not
included in the throughput projections.

The first cold NCA update observed during the smoke process took 0.391 seconds,
and the first cold forward/backward optimizer step took 0.817 seconds. After the
Metal operations had been exercised, steady inference took 0.000609 seconds per
update, or about 1,643 synchronized updates per second. This implies roughly
205-411 core-only frames per second at eight to four updates per rendered frame,
well above the proposed 15 fps rate before rendering and frame synchronization
are added.

The 10,000-update no-gradient trace completed in 3.20 seconds when synchronized
every 1,000 updates, or about 3,122 updates per second. Live MPS allocation was
623,616 bytes at both the start and end. Driver allocation increased once by
49,152 bytes and then remained constant through every later sample. Every
channel retained its initial finite statistics because the final projection is
initialized to exactly zero; this validates the intended identity recurrence
and bounded inference memory, not the stability of a learned rule.

Steady training timings begin after two optimizer steps: the recorded first-call
step and a synchronized memory-probe step. The projections therefore represent
warmed graph throughput and exclude first-call compilation. Bit-exact replay is
specific to the recorded hardware and pinned PyTorch environment; compatibility
metadata hard-fails, but exact arithmetic is not claimed across other runtimes.

### Training-shaped rollout measurements

| Rollout | Steady seconds/optimizer step | Highest synchronized MPS allocation | Projected steps in 10 min | Projected steps in 2 hr |
|---:|---:|---:|---:|---:|
| 1 | 0.00169 | 10.2 MiB | 354,792 | 4,257,500 |
| 4 | 0.00504 | 33.4 MiB | 119,042 | 1,428,505 |
| 8 | 0.00750 | 67.2 MiB | 80,027 | 960,321 |
| 16 | 0.01805 | 134.9 MiB | 33,234 | 398,813 |
| 32 | 0.03819 | 270.3 MiB | 15,712 | 188,545 |

PyTorch MPS does not expose a resettable peak-memory statistic. The table
therefore reports the highest synchronized `current_allocated_memory` snapshot
before, after forward, after backward, and after the optimizer step. The forward
snapshot was highest in every case, and allocation scaled approximately linearly
with rollout length as expected.

The measured graph leaves substantial runtime and memory headroom for the first
learning experiment. The projections should not be interpreted as expected
learning progress: Stage 1 must add a real teacher trajectory and objective, and
its throughput must be measured rather than inferred from parameter count.

## Measured Stage 1 Training

The retained Stage 1 deterministic diagnostic completed 2,000 optimizer steps
in 68 seconds, or about 29 steps per second, using batch 4 and randomized
rollouts of 2-8 updates. This includes CPU-resident teacher-window selection and
MPS forward/backward work, but not teacher-trajectory generation or artifact
rendering.

Increasing the budget to 8,000 steps with 4-16-update rollouts took 495 seconds,
or about 16 steps per second, and produced a substantially less stable rule.
Throughput was still well inside the local budget; experiment quality, not
compute, was the limiting factor. This is direct evidence that a longer run is
not currently justified as a route to stability.

## H100 Scaling Boundary

An H100 should not be used to rescue an unproven objective. It becomes
appropriate when local evidence shows all of the following:

1. The NCA is alive and stable.
2. Correct audio produces a stronger or more appropriate trajectory than the
   counterfactuals.
3. The behavior is aesthetically worth expanding.
4. A measured local bottleneck limits an experiment with a clear hypothesis,
   such as larger batches for diversity, longer credit assignment, larger grids,
   or a larger archive search.
5. The expected benefit of additional compute is stated before the rental.

More compute does not repair an objective that rewards the wrong behavior. It
only reaches that behavior faster.
