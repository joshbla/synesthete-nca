# Experiment Plan

## Principle

The experiments are ordered to isolate failure. Each stage answers one question
and has a stop condition. Later stages do not begin because earlier output looks
promising; they begin because earlier output passed its fixed tests.

The sequence deliberately includes imitation and hand-controlled diagnostics
before novel synthesis. Those are not the final creative result. They establish
that the recurrent model, audio pipeline, and objective work before the project
asks them to discover a visual language.

## Standard Run Tiers

Every experiment should define one or more of these budgets.

### Smoke: 1-3 minutes

- Verify shapes, device placement, finite gradients, checkpointing, rendering,
  and exact replay.
- Use tiny step counts and short rollouts.
- Do not judge aesthetics.

### Rapid: no more than 10 minutes

- Compare one variable against a baseline.
- Render fixed probes.
- Reject instability or an objective that obviously rewards the wrong behavior.
- Do not run broad hyperparameter sweeps.

### Full local: 1-2 hours

- Train one justified candidate.
- Run the complete evaluation matrix.
- Test inference far beyond the training horizon.
- Produce artifacts sufficient for a human go/no-go decision.

## Required Experimental Discipline

Every comparison must hold constant:

- model initialization when relevant;
- initial cell state;
- stochastic fire-mask sequence;
- audio feature extraction;
- render mapping;
- evaluation duration;
- all configuration except the named independent variable.

If the test changes audio, it must not also change the state seed. If it changes
the model, both models must receive the same probe set. This sounds elementary,
but the previous projects frequently compared samples whose random diffusion
noise or visualizer identity also changed.

## Stage 0: Runtime and Reproducibility

### Question

Can the exact proposed graph train and roll out on MPS within the local budget,
and can a run be repeated bit-for-bit or within a documented deterministic
tolerance?

### Minimal configuration

- `96 x 96` grid;
- 16 state channels;
- 96 update hidden channels;
- fixed perception kernels;
- no audio or genome;
- rollout lengths 1, 4, 8, 16, and 32;
- batch sizes measured independently;
- simple scalar loss only for exercising backward.

### Work

Follow the benchmark protocol in `local-compute-envelope.md`. Add checkpoint
save/load and video rendering before any real training.

### Required artifacts

- benchmark table with seconds per step and peak memory;
- 10,000-step no-gradient stability trace from an untrained near-identity rule;
- two exact-repeat renders with matching hashes or a documented source of
  nondeterminism;
- checkpoint round-trip test;
- device/fallback log.

### Gate

- No NaN or Inf.
- No silent CPU fallback in the core update.
- Memory remains bounded during long inference.
- Checkpoint reload reproduces the same state trajectory.
- A useful rapid-run and full-run configuration can be projected from measured
  throughput.

### Failure interpretation

This is an engineering failure, not a model failure. Simplify operations,
reduce rollout or batch, and remeasure before changing the research design.

### Maximum wall-clock

One setup session. Each benchmark case should be minutes, not a training run.

## Stage 1: Learn Local Dynamics Without Audio

### Question

Can the architecture learn a bounded local rule that produces coherent motion
and remains stable beyond its training horizon?

### Why a teacher is appropriate here

The first dynamics test should have an unambiguous answer. Generate one simple
monochrome reaction-diffusion or Turing-pattern trajectory offline and train the
NCA to reproduce short transitions from that trajectory. This is imitation, not
novel synthesis. Its purpose is to verify recurrent learning, local propagation,
and long-rollout behavior.

The teacher should be computationally cheap and use no authored visualizer
primitives. Its exact equations and parameters become part of the test fixture.

### Minimal configuration

- one synthetic dynamical family;
- no audio;
- no genome;
- randomized starting times from the teacher trajectory;
- randomized NCA rollout length within a small range;
- trajectory reconstruction loss plus state overflow penalty;
- one central-seed and one field-noise initialization test.

### Evaluation

- short-horizon reconstruction error;
- free rollout much longer than training windows;
- motion energy over time;
- spatial spectrum and total variation relative to teacher range;
- per-channel state bounds;
- qualitative target/prediction/difference video.

### Gate

- The NCA produces recognizable coherent dynamics rather than only matching a
  static frame.
- Motion remains nonzero and bounded during a long free rollout.
- The state does not collapse immediately when rolled beyond the supervised
  horizon.

### Failure interpretation

- Good one-step error but bad free rollout indicates exposure bias or an
  insufficient training horizon.
- Static reconstruction indicates the objective underweights dynamics.
- Explosion indicates update scale or state regularization failure.
- Blurred dynamics may indicate insufficient hidden width, poor boundary
  conditions, or an overly pixelwise loss.

### Maximum wall-clock

- Smoke: 3 minutes.
- Rapid architecture/loss comparisons: 10 minutes each.
- One selected full run: 1 hour initially, extend to 2 only if learning is
  continuing and stability is improving.

## Stage 2: Hand-Controlled Audio Steering

### Question

Can aligned audio features visibly and promptly perturb a stable material
without destroying it?

### Method

Freeze the Stage 1 rule. Introduce a deterministic, non-learned control that
modulates one known-safe quantity, such as update step size or selected hidden
features. Use absolute RMS first, then a synthetic onset pulse.

This is a plumbing control similar in spirit to Artificial Dancing
Intelligence. It is explicitly not evidence that the NCA learned audio control.

### Probe audio

- constant nonzero energy;
- full silence;
- one isolated impulse;
- alternating high/low energy blocks;
- a stepped low-band/high-band synthetic sequence.

### Evaluation

Run every probe from the same starting state and stochastic mask sequence.
Measure motion-energy response, lag, state bounds, and recovery after the control
returns to baseline.

### Gate

- Audio events are aligned correctly.
- Silence and impulses have visibly distinct effects.
- The material recovers rather than dying or exploding.
- The response can affect motion or organization, not only final-video contrast.

### Failure interpretation

Failure here means the feature pipeline or control surface is inadequate. It is
premature to train learned conditioning.

### Maximum wall-clock

No training. A complete probe should render in less than 10 minutes and ideally
near real time.

## Stage 3: Learn Audio-Conditioned Dynamics With a Forced Teacher

### Question

Can the NCA learn to use time-aligned audio features when the correct visual
effect is known and audibly falsifiable?

### Method

Create a forced synthetic dynamical system in which audio features control
documented physical parameters. Examples:

- RMS changes reaction rate or damping;
- onset pulses inject localized activation;
- broad bands alter diffusion anisotropy or competing reaction terms.

Generate paired audio and teacher trajectories. Train the audio-conditioned NCA
to imitate them from matched starting states.

The primary objective can include trajectory reconstruction, but it must also
include correct-versus-shuffled pressure. A candidate form is:

```text
L_good = trajectory_loss(model(state, correct_audio), target)
L_bad  = trajectory_loss(model(state, shuffled_audio), target)

L_contrast = relu(L_good + margin - L_bad)
```

This repeats the proven lesson from the original project at the trajectory
level, where audio affects recurrence rather than one denoised frame.

### Gate

- Correct audio predicts the teacher trajectory better than silent,
  time-shuffled, and batch-shuffled audio.
- The difference is visible from an identical state.
- At least two distinct audio features produce distinguishable structural or
  motion responses.
- The model does not reduce the task to mean brightness.
- Long rollout remains stable.

### Failure interpretation

- If ordinary trajectory loss works but counterfactuals do not, the model again
  ignores audio and the incentive is insufficient.
- If RMS works but bands do not, feature injection or teacher identifiability is
  too weak.
- If correct and shuffled losses separate while videos do not, the metric is
  detecting an imperceptible shortcut.

### Maximum wall-clock

- Rapid loss comparison: 10 minutes.
- One full learned-conditioning run: 1-2 hours.

### Meaning of success

Passing this stage proves learned causal control in a constrained system. It
does not yet prove novel visual synthesis because the visual dynamics have a
teacher.

## Stage 4: Autonomous Material Under Constraints

### Question

Can a learned rule produce its own coherent material behavior while retaining
the audio causality established in Stage 3?

### Method

Reduce reliance on pixel-level teacher trajectories. Optimize a set of
constraints instead of one target video. Candidate terms include:

- state boundedness;
- nonzero but bounded motion;
- long-term survival;
- multiscale spatial structure;
- temporal persistence;
- silence settling;
- onset-triggered change;
- correct-versus-shuffled trajectory separation;
- penalties for response explained only by global luminance.

No single weighted sum should be called an aesthetic score. Constraint metrics
define a viable region. Human review chooses promising behavior within that
region.

Start by fine-tuning or adapting a stable Stage 3 rule rather than searching an
unbounded random rule space. Compare against training from scratch only after a
valid result exists.

### Gate

- The system passes the complete counterfactual matrix below.
- Motion is coherent and remains bounded well beyond the training horizon.
- Audio affects spatial organization or dynamical regime after mean luminance
  is removed from analysis.
- At least several seeds produce viable related trajectories.
- A human reviewer prefers at least one behavior for reasons beyond matching a
  teacher or known visualizer.

### Failure interpretation

- Noise indicates that complexity proxies are being gamed.
- Flicker indicates that correlation is being gamed through luminance.
- One attractor across all seeds indicates insufficient generative diversity.
- Attractive autonomous behavior with failed counterfactuals is the original
  audio-ignoring failure and must not advance.

### Maximum wall-clock

Individual candidates remain within 1-2 hours locally. Search breadth is kept
small and manually justified.

## Stage 5: Genome and Diversity

### Question

Can one update rule support multiple coherent behavioral regimes without
turning the genome into a set of discrete presets or weakening audio control?

### Method

Add a 4-8-dimensional persistent genome. Train or search over it only after one
working autonomous material exists.

Evaluate:

- endpoint and interpolation behavior;
- diversity across fixed genome samples;
- diversity across seeds within one genome;
- preservation of correct-versus-shuffled audio separation;
- whether genome effects dominate audio effects;
- whether regimes are continuous or collapse to a few categories.

### Gate

- Multiple genomes are viable and distinct.
- Interpolation mostly traverses viable behavior.
- Audio causality remains measurable in every accepted regime.
- Diversity is not explained solely by stochastic noise.

### Quality-diversity boundary

MAP-Elites or another archive search may enter after this gate. It should search
a low-dimensional genome or compact rule family using validated descriptors.
It should not be the method used to discover whether the basic NCA can work.

## Stage 6: Real Music and Long Duration

### Question

Does the learned relationship remain useful outside synthetic probes?

### Method

Use a small curated set of real tracks spanning differences in onset density,
spectral balance, dynamics, and silence. Do not add semantic labels or a large
audio encoder yet.

Render longer clips and inspect:

- delayed or accumulated response;
- attractor collapse;
- long-run numerical drift;
- sensitivity to mastering loudness;
- whether different tracks are visually distinguishable from identical state;
- whether the system follows local audio structure rather than only average
  track energy.

### Gate

At least two real tracks produce recognizably different but coherent
trajectories, and time-shuffling each track worsens alignment while preserving
the same global feature distribution.

## Counterfactual Evaluation Matrix

Every learned audio-conditioned checkpoint is evaluated from the same state and
random mask sequence under:

| Condition | Purpose |
|---|---|
| Correct audio | Reference trajectory |
| Exact repeat | Reproducibility |
| Full silence | Absolute-energy dependence and settling |
| Time-shuffled audio | Temporal alignment while retaining feature values |
| Batch-shuffled audio | Clip identity/conditioning dependence |
| Reversed audio features | Directional temporal dependence |
| Gain-scaled audio | Response monotonicity and normalization behavior |
| RMS-only | Test whether energy alone explains behavior |
| Bands-only | Test spectral control without energy |
| Feature-by-feature ablation | Attribute behavior to controls |
| Frozen mean feature vector | Detect whether temporal variation matters |

The exact audio waveform may remain attached for human viewing, but model input
and displayed audio must be labeled so a shuffled-condition video is not
mistaken for correct synchronization.

## Metrics

No metric below is sufficient alone.

### Motion energy

```text
M_t = mean(|Y_t - Y_(t-1)|)
```

Useful for detecting activity and silence response. Easily gamed by flicker and
global luminance, so also compute it after subtracting frame means and at
multiple spatial scales.

### Lagged audio-motion correlation

Compute cross-correlation between audio features and motion descriptors over a
reasonable lag window. Report the peak correlation and lag, not only zero-lag
correlation.

Caveat: a high value can come from trivial brightness modulation. Repeat with
mean-normalized frames and structural descriptors.

### Onset-triggered response

Average motion and structural change in windows around known synthetic onsets.
Measure baseline, peak magnitude, time-to-peak, and recovery.

### Counterfactual trajectory divergence

From identical state:

```text
D(t) = distance(Y_correct(t), Y_counterfactual(t))
```

Use raw pixel distance and multiscale descriptors. Chaotic systems can amplify
tiny differences, so divergence alone does not prove semantically appropriate
control. It must be paired with alignment and stability.

### Stability and boundedness

Track per-channel min, max, mean, variance, NaN/Inf count, visible motion, and
gradient norm. Report the first time a threshold is violated.

### Structure proxies

Candidate measurements:

- total variation;
- multiscale spatial power spectrum and spectral slope;
- spatial autocorrelation length;
- connected-component size distribution after fixed thresholds;
- frame compressibility;
- temporal autocorrelation;
- entropy at several quantizations.

These distinguish some static/noisy/saturated regimes, but none measures beauty
or novelty. Random noise scores highly on several complexity measures.

### Diversity

Measure pairwise trajectory-descriptor distances across seeds and genomes while
holding audio fixed, then across audio while holding seed/genome fixed. This
separates generative diversity from audio sensitivity.

### Regime occupancy

Cluster low-dimensional trajectory descriptors only after enough viable runs
exist. Use occupancy and transition statistics to test whether the system has
multiple regimes. Do not interpret every cluster as a meaningful behavior
without reviewing its videos.

## Required Artifacts Per Full Run

Each full run produces:

- immutable configuration;
- source revision;
- checkpoint and optimizer state;
- feature schema and normalization statistics;
- seed and stochastic-mask seed;
- scalar training log;
- per-channel stability trace;
- native-resolution frame data before compression;
- correctly labeled videos for the full counterfactual matrix;
- a synchronized comparison grid or contact sheet;
- metric plots including lag curves;
- a short written result: hypothesis, outcome, failure interpretation, next
  justified change.

## Run Ledger

Maintain a simple append-only ledger once implementation begins. Each entry
records:

```text
run identifier
hypothesis
single intentional change
wall-clock and optimizer steps
checkpoint
probe results
human observation
decision: reject / retain / repeat
```

The ledger prevents a visually memorable sample from replacing the history of
what configuration actually produced it.

## H100 Gate

An H100 run is justified only when:

- a local checkpoint passes causality and stability gates;
- local throughput and memory have been measured;
- the cloud experiment changes one capacity constraint that is plausibly
  limiting the result;
- evaluation is ready before the run starts;
- a smaller local ablation supports the scaling hypothesis.

Examples of valid reasons include longer differentiable rollouts, a larger
batch for diversity objectives, higher grid resolution after multiscale
behavior is established, or a validated quality-diversity archive that is too
slow locally.

"The output is not good enough" is not a scaling hypothesis.
