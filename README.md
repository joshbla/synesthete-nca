# Synesthete

Synesthete is a local-first research project for generating an organic,
self-organizing visual material whose evolution is causally steered by audio.

The current direction uses a Neural Cellular Automaton (NCA): a grid of cells
repeatedly applies one small learned local rule. The visible image is one
channel of the persistent cell state. Motion is therefore the history of one
evolving system, not a sequence of independently generated frames.

This repository is the third attempt at the idea. It begins with documentation
because the earlier attempts showed that an implementation can appear visually
promising while failing the actual objective. Before model code is written, the
project records what failed, what the target machine can support, what the new
architecture is expected to accomplish, and how success will be falsified.

## Status

**Project environment established. No model has been implemented yet.**

The current work establishes:

- the desired visual and behavioral outcome;
- the postmortem of the two previous projects;
- the local compute envelope for the target Apple Silicon machine;
- the initial NCA architecture and audio-conditioning strategy;
- a staged experiment plan with explicit counterfactual evaluations;
- the research evidence behind the decision and the evidence still missing.

The repository now includes a minimal Python, PyTorch, test, and lint setup for
the Stage 0 runtime benchmarks. It does not yet include an NCA, audio feature
extractor, training pipeline, experiment configuration, or cloud resources.

## Setup

The project uses Python 3.13 and `uv`:

```bash
uv sync
uv run synesthete-check --require-mps
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

The runtime check reports the Python and PyTorch versions, fails when the local
MPS backend is unavailable, and executes a small synchronized matrix operation
on the GPU. This is an environment check only; the Stage 0 model benchmark
described in the experiment plan has not been implemented.

## The Core Bet

The earlier systems tried to learn an audio-to-frame or audio-to-video mapping
from examples of hand-written visualizers. That formulation required one model
to solve visual quality, temporal coherence, audio synchronization, style
diversity, and novelty at once. More importantly, its loss could become small
while the model ignored audio.

The new formulation narrows the problem:

> Learn a compact dynamical visual system whose local rules create coherent
> motion, then make audio materially alter the system's trajectory.

The output is initially monochrome and abstract. This removes easy shortcuts
such as frequency-to-hue mappings and avoids spending local compute on semantic
scene synthesis. Monochrome is a starting constraint, not a permanent claim
about the eventual limits of the project. Color, larger systems, and broader
visual representations can be reconsidered after motion-based audio causality
and useful novelty have been demonstrated.

## Initial System

The proposed baseline is deliberately small:

| Component | Initial decision |
|---|---|
| State grid | 96 x 96 |
| Cell state | 16 channels |
| Visible output | 1 luminance channel |
| Perception | Identity, Sobel X/Y, Laplacian |
| Update network | Per-cell 1x1 network, about 96 hidden units |
| Cell updates | Stochastic and asynchronous |
| NCA steps per video frame | 4-8 |
| Video rate | 15 or 24 fps |
| Audio conditioning | At most 10 low-level, time-aligned features |
| Style control | Deferred: small persistent genome vector |
| Local target | Rapid runs under 10 minutes; useful runs in 1-2 hours |

This is not yet a commitment to a particular loss. The difficult research
question is not whether an NCA can render quickly; existing work establishes
that it can. The difficult question is whether a locally trainable objective can
produce dynamics that are simultaneously:

- alive rather than static;
- structured rather than noise;
- stable over long rollouts;
- responsive to audio without reducing responsiveness to global brightness;
- varied enough to feel synthesized rather than preset;
- reproducible from the same seed and audio.

## What Success Means

A visually attractive output is not sufficient. The first autonomous-material
result, after the diagnostic stages, must pass all of these tests:

1. **Persistent dynamics:** the material remains coherent and active without
   being redrawn from scratch each frame.
2. **Audio causality:** from the same starting state, changing only the audio
   produces a meaningfully different trajectory.
3. **Legible silence:** silence or a dropout causes the material to settle,
   hold, reorganize, or otherwise respond differently.
4. **Temporal alignment:** audio events affect motion with a measurable and
   visually acceptable lag.
5. **More than brightness:** audio changes organization, flow, growth, or
   regime, not only luminance or surface flicker.
6. **Reproducible novelty:** different seeds or genome values produce distinct
   coherent behaviors, while the same seed and audio reproduce the same run.

The primary negative control is always the same-state counterfactual: correct
audio versus silent, shuffled, or feature-ablated audio. This is inherited from
the most important lesson of the previous projects: conditioning being present
does not mean the model uses it.

## What This Is Not

The first system is not intended to generate:

- arbitrary cinematic video;
- semantic scenes or recognizable objects;
- a collage of bars, waveforms, particles, or other known visualizers;
- a realistic interpretation of the physical source of a sound;
- a full music video from a prompt;
- proof of general audio-visual intelligence.

It is a learned visual material. Its novelty should come from the behavior of a
small recurrent rule, not from combining a large library of authored effects.

## Local-First Constraint

The target development machine is an Apple M5 MacBook Pro with 10 CPU cores,
10 GPU cores, and 16 GB of unified memory. A useful experiment must fit this
machine and return evidence on human timescales:

- 1-3 minutes for an end-to-end smoke check;
- no more than 10 minutes for rapid comparison runs;
- 1-2 hours for a meaningful local training run.

An H100 is a later scaling option, not a requirement for discovering whether
the central idea works. Cloud training becomes justified only after the local
system demonstrates causality, stability, and nontrivial behavior, and only
when a measured bottleneck explains why more compute should improve it.

## Documentation

- [Vision and Scope](docs/vision-and-scope.md): the desired experience,
  boundaries, definition of novelty, and staged ambition.
- [Previous Attempts](docs/previous-attempts.md): technical postmortem of the
  latent-diffusion and baby-diffusion projects.
- [Local Compute Envelope](docs/local-compute-envelope.md): memory and runtime
  constraints for the target Mac, plus the benchmark protocol.
- [Technical Direction](docs/technical-direction.md): proposed NCA state,
  update rule, audio injection, stability controls, and deferred alternatives.
- [Experiment Plan](docs/experiment-plan.md): ordered experiments, required
  artifacts, counterfactual evaluation, and stop/go gates.
- [Research Basis](docs/research-basis.md): what the relevant NCA research
  proves, what it does not prove, and how it informs this design.

## Development Principle

The project advances by evidence, not architectural accumulation. Each new
mechanism must answer a measured failure of the smallest prior system. A model
that looks interesting but fails its audio counterfactuals is not partial
success; it is evidence that the visual prior has again learned to ignore the
conditioning.
