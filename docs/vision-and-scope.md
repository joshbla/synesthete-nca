# Vision and Scope

## Purpose

Synesthete aims to create a visual process that feels grown rather than drawn:
an abstract material that persists through time, organizes itself through local
interactions, and changes its behavior in response to sound.

The intended experience is not simply that a loud sound produces a large image
or that a high frequency produces a particular color. The relationship should
be visible in the material's dynamics: how structures form, travel, merge,
dissolve, oscillate, settle, and transition between regimes.

This is a research project, not yet a product specification. The immediate goal
is to discover whether a small learned dynamical system can produce behavior
that is both aesthetically promising and causally controlled by audio within a
strict local-compute budget.

## Desired Output

The primary artifact is a short monochrome video paired with its input audio.
The video depicts one persistent material state evolving throughout the clip.
It should read as a continuous organism, medium, or artificial physical system,
not as a slideshow of frames.

The same trained model should support multiple runs. Variation can come from:

- the input audio;
- the initial cell state or random seed;
- a small persistent genome vector that alters the material's behavioral
  tendencies;
- stochastic asynchronous cell updates, when enabled under a recorded seed.

All of these inputs must be recorded so a run can be reproduced. Unrepeatable
difference is variance, not evidence of novelty.

## Why Organic Self-Organization

The broad space of audio visualization is too unconstrained for local training
from scratch. It includes typography, geometry, particles, landscapes,
cinematic scenes, semantic associations, color systems, editing grammar, and
many other unrelated visual languages. A small model cannot infer that entire
space from synthetic examples, and a large pretrained video model would replace
local experimentation with the cost and assumptions of its existing visual
prior.

An organic self-organizing material is a meaningful subset because it has a
strong inductive bias:

- every location follows the same local rule;
- complex global behavior must emerge from repeated local interaction;
- state persists, so temporal coherence is inherent rather than repaired later;
- the representation naturally supports growth, reaction, diffusion,
  regeneration, waves, fronts, and texture;
- model size can remain tiny even when the rendered grid is large.

This constraint narrows the visual language without predetermining the exact
visual result. It gives the system room to synthesize behavior while keeping the
search space compatible with the target laptop.

## What Novelty Means

Novelty is not defined as random output or as a new combination of authored
visualizer modules. It has three progressively stronger meanings here.

### 1. Trajectory novelty

From a familiar behavioral grammar, the model produces a spatial-temporal
trajectory that was not included as a training target. Different seeds and
audio clips lead to different coherent evolutions.

### 2. Regime novelty

The material supports qualitatively different modes such as settling,
branching, flowing, crystallizing, pulsing, or turbulent reorganization. These
modes arise from the learned rule and its state rather than from switching
between explicitly authored renderers.

### 3. Emergent grammar

The system develops persistent motifs or transitions that were not specified
as target primitives and are not recognizable copies of the supervision. This
is the long-range aspiration, not something the first local model can be
assumed to achieve.

A result is not novel merely because it is noisy, unstable, or different on
every execution. Coherence, persistence, and reproducibility are required.

## Why Start in Monochrome

Removing color cuts the visible output from three channels to one, but compute
savings are not the main reason. Color provides an easy conditioning shortcut.
Frequency-to-hue can make a system appear audio-reactive while its structure and
motion remain unchanged. The previous baby project used exactly this kind of
mapping and still failed to learn meaningful conditioning.

Monochrome forces the first experiments to express audio through luminance
structure and motion. It also makes evaluation clearer: changes in motion
energy, spatial organization, and regime are less likely to be hidden by an
attractive palette.

This is an experimental constraint, not a permanent prohibition. Color can be
reconsidered after the monochrome system demonstrates that audio controls its
dynamics rather than only its presentation. The same principle applies to
larger grids, richer audio encoders, and broader visual representations: each is
a possible later extension after the central causal result exists.

## In Scope for the First System

- A 2D persistent cell state rendered as a monochrome field.
- A learned local update rule shared by every cell.
- Low-level, time-aligned audio features as control signals.
- Short synthetic and real audio clips.
- Deterministic reruns from recorded model, seed, audio, and configuration.
- Local training and inference on the target Apple Silicon machine.
- Objective metrics for causality and stability, accompanied by human review.
- Multiple learned behavioral tendencies through a small genome vector, after
  a single stable behavior has been proven.

## Deferred, Not Rejected

These directions may become appropriate after the first system works:

- color as an additional expressive dimension;
- larger spatial grids and longer clips;
- richer learned audio representations;
- quality-diversity search over trained rules or genome values;
- neural fields or continuous coordinates for higher-resolution rendering;
- cloud training on H100-class hardware;
- a performance interface or real-time controls;
- a broader visual grammar beyond cellular materials.

They are deferred because introducing them now would obscure which mechanism
caused success or failure. Their eventual value should be evaluated from actual
limitations of the working baseline.

## Out of Scope for the Initial Research Program

- General-purpose text-to-video or audio-to-video generation.
- Semantic reconstruction of a sound source.
- Narrative editing, camera control, or cinematic scene composition.
- Training a foundation-scale video model from scratch.
- Presenting hand-written visualizer combinations as learned novelty.
- Treating aesthetic appeal alone as evidence of audio conditioning.

These goals may belong to a future successor architecture, but they do not
belong in the first NCA proof.

## Human Success Criteria

A skeptical viewer should be able to observe the following without knowing the
implementation:

1. The material stays coherent over time and does not reset each frame.
2. Silence produces a clear change in behavior.
3. Distinct audio passages produce distinct motion or organizational changes.
4. Replacing the audio while holding the starting state fixed changes the
   trajectory.
5. The response is not limited to global brightness or flicker.
6. Repeating the exact run reproduces the same result.
7. Different seeds or genome settings create related but meaningfully distinct
   behavior rather than a few fixed presets.

## Failure Criteria

The following outputs are failures even if they look attractive:

- the same trajectory occurs under correct, silent, and shuffled audio;
- audio only changes mean brightness;
- motion is structureless noise;
- the state dies, saturates, explodes, or freezes after a short rollout;
- every seed converges to the same attractor;
- frames flicker independently rather than showing persistent state;
- the system reproduces a target texture without developing useful dynamics;
- evaluation depends on changing several variables at once.

An NCA does not automatically prevent the audio-ignoring shortcut. It can learn
a compelling autonomous material and disregard the conditioning just as the
diffusion models did. The architecture makes persistent motion cheap; the
training objective and counterfactual tests must still make audio dependence
necessary.

## Staged Ambition

### Stage 1: Establish life

Find a small unconditioned NCA that remains bounded and exhibits coherent
nontrivial motion. This isolates the dynamical representation before audio is
introduced.

### Stage 2: Establish control

Demonstrate that simple hand-injected audio modulation changes the same NCA
state in predictable ways. This validates feature extraction, alignment,
rendering, and evaluation without claiming learned conditioning.

### Stage 3: Learn control

Train the update rule to use audio and pass correct-versus-shuffled,
correct-versus-silence, and feature-ablation counterfactuals from the same
starting state.

### Stage 4: Establish novelty

Add seed and genome variation, then determine whether the model supports
multiple coherent dynamical regimes without losing audio causality.

### Stage 5: Generalize and scale

Use real music, longer durations, larger grids, and only then consider H100
training or a different representation. Scaling is justified when a working
local model is limited by measured capacity or rollout cost, not merely because
its first output is imperfect.

## Research Standard

Every claimed improvement must be attached to a fixed comparison, recorded
configuration, and reproducible artifact. The project should prefer a narrow
negative result over an impressive output whose cause cannot be determined.
