# Research Basis

## Purpose

The NCA direction is supported by existing evidence, but the complete proposed
system has not already been solved by one paper. This document separates:

- what the literature demonstrates;
- what design decision that evidence informs;
- what it does not establish;
- where this project still carries research risk.

The central unproven claim is that a small NCA can learn a useful audio-control
relationship while also developing autonomous, stable, aesthetically promising
dynamics. Existing work proves the components around that claim, not the claim
itself.

## Growing Neural Cellular Automata

**Source:** Mordvintsev et al., [Growing Neural Cellular
Automata](https://distill.pub/2020/growing-ca/)

### What it demonstrates

A shared local neural update rule can grow a target image from a seed, maintain
it, and regenerate damaged regions. A small cell state contains visible RGBA
channels and hidden communication channels. Cells perceive local gradients and
apply stochastic residual updates.

This establishes the core representation:

- global form can emerge from local computation;
- recurrence supplies persistent state;
- the same rule can operate at every cell;
- stochastic asynchronous updates can remain coherent;
- damage and long rollout can be meaningful tests.

### What it does not demonstrate

- audio conditioning;
- autonomous aesthetic novelty;
- dynamic texture as the primary objective;
- multiple behavioral regimes in one rule;
- stability under arbitrary external forcing.

The canonical task is supervised growth toward a known target. It proves that
the machinery works, not that a good creative objective is known.

### Decision informed

Use 16 persistent channels, fixed local perception, residual updates,
near-zero initial update projection, asynchronous cell masks, and explicit
long-rollout/damage evaluation.

## Self-Organizing and Ultra-Compact Texture NCA

**Source:** Mordvintsev and Niklasson,
[Texture Generation with Ultra-Compact Neural Cellular
Automata](https://arxiv.org/abs/2111.13545)

### What it demonstrates

NCA can act as compact procedural texture programs. The paper reduces learned
rules to 68, 150, 264, and 588 parameters, and reports one-byte quantization
without obvious quality loss for its examples. Simple and regular patterns can
be represented especially well. Some learned rules exhibit surprising temporal
behavior, such as colored dots oscillating through states while preserving an
overall target distribution.

This supports a key aesthetic premise: recurrence plus severe parameter sharing
can discover procedural behavior that is not an explicit sequence of rendering
commands.

### What it does not demonstrate

- one model covering a broad texture distribution;
- audio-driven control;
- high-fidelity natural images;
- guaranteed motion or long-term stability;
- that fewer parameters always produce better emergence.

The smallest models are trained per target image and favor regular patterns.
Their extraordinary compactness is evidence about representational possibility,
not a requirement that this project use only hundreds of parameters.

### Decision informed

Keep the rule in the tens-of-thousands range and let recurrence create visual
complexity. Do not increase model size merely to increase apparent capacity.

## DyNCA: Dynamic Texture Synthesis

**Sources:** Pajouheshgar et al.,
[DyNCA](https://arxiv.org/abs/2211.11417) and the
[project page](https://dynca.github.io/)

### What it demonstrates

DyNCA trains compact NCAs for dynamic texture synthesis and separates target
appearance from target motion. It adds positional encoding and, for larger
grids, multi-scale perception. Motion can be supervised from a vector field or
from video through optical-flow features.

Reported model sizes are approximately:

```text
DyNCA-S   6,000 parameters
DyNCA-L  10,000 parameters
```

The paper reports the following on one A100 40 GB GPU:

| Model | Grid | Training time | Synthesis per frame |
|---|---:|---:|---:|
| DyNCA-S | 128 x 128 | 2,320 s | 0.033 s |
| DyNCA-S | 256 x 256 | 3,980 s | 0.057 s |
| DyNCA-L | 128 x 128 | 2,370 s | 0.035 s |
| DyNCA-L | 256 x 256 | 4,380 s | 0.057 s |

The system uses 24 NCA updates per video frame for vector-field motion
experiments and 64 updates per frame when learning motion from dynamic texture
videos. Once trained, it can roll indefinitely and can render at grid sizes
different from its training size. It also supports post-training controls such
as speed, direction, local coordinate transforms, and damage/brush interaction.

### Important qualifications

The reported training times are A100 measurements, not expected M5 times. The
losses use pretrained visual features and, for video motion, a pretrained
optical-flow network. These networks can dominate local training cost even
though the NCA itself is tiny.

The paper's "infinitely long" claim means that the recurrent generator has no
fixed video length. It does not mean every learned rule remains aesthetically
or numerically ideal forever.

Reported limitations include incompatibility between some target appearance and
motion combinations, difficult motion-loss weighting, and reduced ability to
model videos that violate the temporally homogeneous dynamic-texture
assumption.

### What it does not demonstrate

- audio control;
- motion learned without a target vector field or video;
- broad visual novelty from one rule;
- local M5 training time;
- a cheap objective without pretrained visual networks.

### Decision informed

Dynamic texture is a realistic NCA output class, and 96-128-pixel grids with
very small rules are credible. Multi-scale perception and positional encoding
remain available if the baseline cannot produce global organization, but they
are not required initially.

## Emergent Dynamics in NCA

**Source:** Xu et al., [Emergent Dynamics in Neural Cellular
Automata](https://arxiv.org/abs/2404.06406)

### What it demonstrates

The study varies cell-state channel count `C` and update-network hidden width
`D` across 512 trained texture models. It measures spontaneous motion using
optical flow. Its main empirical result is that motion tends to disappear when
the update network is too narrow relative to state size. The authors recommend:

```text
D > C and, in practice, D / C > 2
```

Their dynamic example `C = 24, D = 96` contrasts with a comparatively static
`C = 96, D = 128` configuration.

### What it does not demonstrate

This is a correlation within a specific texture-training setup, not a universal
law. A high `D / C` ratio does not guarantee stable, useful, controllable, or
aesthetically interesting dynamics.

### Decision informed

Use `C = 16` and `D = 96` for a ratio of 6, safely above the observed threshold
without making the update network large. Measure whether this actually helps
rather than treating it as settled.

## Signal-Responsive Multi-Texture NCA

**Source:** Catrina et al., [Multi-Texture Synthesis through Signal
Responsive Neural Cellular Automata](https://arxiv.org/abs/2407.05991)

### What it demonstrates

The work reserves hidden cell-state channels as a genome signal. One, two, or
three binary genome channels identify 2, 4, or 8 target textures. One NCA can
therefore develop different textures depending on its internal signal. The
authors also demonstrate interpolation between genome values, grafting regions
with different identities, and regeneration adaptations.

Training uses `128 x 128` states, randomized rollouts of 32-96 NCA steps,
batches drawn from a persistent pool, and a pretrained VGG16 observer for
texture loss.

### What it does not demonstrate

Despite "signal-responsive" in the title, this is not an audio paper. The
signal is a persistent texture identity encoded in genome channels, not a
rapidly varying temporal control. The work does not establish synchronized
motion or stable control under continuously changing input.

### Decision informed

A small persistent genome is a credible way to let one rule cover multiple
material regimes. It should be added after one audio-controlled behavior works,
because a strong static genome can otherwise become another route for ignoring
audio.

## Artificial Dancing Intelligence

**Source:** Salcedo and Egozy, [Artificial Dancing Intelligence: Neural
Cellular Automata for Visual Performance of
Music](https://proceedings.mlr.press/v303/salcedo26a.html)

### What it demonstrates

This is the closest direct precedent for the proposed experience. It runs a
Growing NCA in a browser and modulates it with live audio RMS. The reported
system uses a `64 x 64 x 16` cell state and measured an average 31.2 frames per
second on a MacBook Air with Apple M2 and 8 GB RAM.

Inspection of its public implementation shows the audio path clearly:

1. Web Audio computes RMS.
2. The user selects one or more of the 16 state channels.
3. Before each NCA update, those selected state channels are multiplied by
   `RMS x user_multiplier`.
4. The ordinary trained NCA step advances the modified state.

The interface also controls NCA step size, color shift, seeding, and model
selection.

### Critical qualification

The audio relationship is not learned. The available models are trained Growing
NCAs for target forms such as a lizard or heart. RMS is applied at inference as
an external intervention. This proves real-time feasibility and shows that
audio can steer NCA behavior, but it does not solve learned audio conditioning.

The paper also reports that user settings and audio can push the system into
chaotic or explosive behavior that does not always fit the musical context.

### Decision informed

The M2 result is strong evidence that the target M5 can support baseline
inference, but the local implementation must still be benchmarked. Direct RMS
channel modulation should be reproduced as a Stage 2 control experiment. The
learned system should instead modulate update features and be trained/evaluated
for stability.

## Quality-Diversity Search Over NCA

**Source:** Earle et al., [Illuminating Diverse Neural Cellular Automata
for Level Generation](https://arxiv.org/abs/2109.05489)

### What it demonstrates

The paper uses CMA-ME to evolve archives of roughly 3,000-parameter NCA level
generators. The archive explicitly covers behavioral descriptors such as level
symmetry and path length. Compared with several one-pass generators, NCA filled
more of the measured level space and generalized better to new seeds in the
reported setting.

This supports a later idea: once valid NCA behavior and descriptors exist,
quality-diversity methods can retain multiple useful rules rather than optimize
one average candidate.

### Compute qualification

The reported experiments used 50,000 iterations on a 48-CPU node and took up to
three days, with repeated trials. This is not evidence that a full local
quality-diversity search will finish in ten minutes. Evolution was feasible
partly because the NCA had few parameters, but population evaluation still
required substantial compute.

The domain was discrete game-level generation with explicit validity and solver
criteria. Aesthetic dynamic texture has less reliable descriptors and is easier
to game.

### Decision informed

Defer MAP-Elites/CMA-ME. First validate one NCA and its metrics. Later search a
small genome or compact rule space, use measured parallel throughput, and keep
human review in the loop.

## Alternative Families Considered

### Latent neural differential equations

Latent neural ODE/SDE video models provide continuous dynamics with fewer
parameters than many recurrent generators. They offer a compact, smooth control
surface and may be useful if NCA cannot maintain global coordination.

They are not the first choice because a decoder still maps a global latent to a
frame, concentrating visual representation outside the dynamics. The NCA more
directly encodes the desired material metaphor.

### Neural fields and hypernetworks

Coordinate networks can represent continuous images and render at arbitrary
resolution. Hypernetworks can generate their weights from conditions. This
could later turn NCA state or audio into high-resolution fields.

The training data and stability burden is higher: a manifold of valid field
weights must be learned, and audio-conditioned weight generation is itself a
research project. Small per-pixel MLP dispatch may also be less efficient than
convolution on MPS.

### Modern video diffusion

Pretrained video diffusion can create much broader and more semantic output.
Adapters or controls could react to audio, especially on rented hardware.

That route does not meet the current objective. It imports an enormous visual
prior, requires much more memory and sequential compute, and risks producing a
music-video interpretation rather than a novel local material. It may be
revisited if the project later generalizes beyond cellular dynamics.

### Autoregressive visual tokens

A small transformer over learned visual tokens could model a distribution of
motion sequences. It would require a tokenizer and a training corpus of visual
sequences, reintroducing a multi-stage pipeline before the source of the visual
grammar exists.

## Evidence Gaps

The literature does not currently establish all of the following together:

- learned rather than hand-injected audio control;
- continuously changing multi-feature audio modulation;
- autonomous abstract dynamics rather than target imitation;
- stable long rollouts;
- multiple coherent regimes in one compact rule;
- training within the stated M5 time budget;
- objective novelty that agrees with human aesthetic judgment.

Those gaps define this project's experiments. The design is credible because
each component has evidence, but success remains uncertain because their
combination is the research contribution.

## Working Conclusions

1. NCA is an unusually strong fit for persistent organic motion under local
   compute constraints.
2. The planned `96 x 96 x 16` state and roughly 20,000-30,000-parameter rule
   are conservative relative to published systems.
3. Real-time local inference is plausible and should be easy to falsify early.
4. Learned audio causality is not solved by existing music-NCA work.
5. Objective design and long-horizon stability are more important risks than
   parameter memory.
6. Quality-diversity and H100 scaling are credible later tools, not substitutes
   for a valid baseline.
