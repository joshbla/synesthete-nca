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

### Result

The runtime and reproducibility gate passed on the target MPS machine. The exact
`96 x 96 x 16`, 7,792-parameter graph completed finite forward/backward steps at
rollout lengths 1, 4, 8, 16, and 32 with MPS fallback disabled. A 10,000-update
no-gradient rollout of the zero-initialized output projection stayed finite and
kept live MPS allocation constant. A separate nonzero 64-step diagnostic
trajectory replayed bit-for-bit from the same state and mask seeds, including
after a strict checkpoint round trip.

State-trajectory hashes replace the originally proposed repeat-render hashes in
this stage because the current milestone explicitly excludes a rendering stack.
This is stronger evidence for recurrent numerical replay but does not prove the
future render mapping or video assembly. Batch and grid scaling also remain
separate follow-up measurements rather than being mixed into the rollout-length
baseline.

These results establish engineering feasibility only. The identity-initialized
long rollout cannot establish stability after learning, persistent motion,
visual quality, or audio causality. Detailed measurements are recorded in
`local-compute-envelope.md`.

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

### Result

Stage 1 passes for the intended asynchronous baseline when teacher and NCA local
clocks are aligned. A bounded oscillatory reaction-diffusion teacher replaced an
initial Gray-Scott teacher after repeated short-horizon fits amplified its
autocatalytic growth into long-run explosion. This was a teacher-design failure,
not evidence against local recurrence.

With deterministic full-cell updates, the clean retained 2,000-step run trained
in 49 seconds and remained finite over 512 evaluation updates. Central and distributed
initializations reached trajectory MSE 0.00879 and 0.01032, while predicted
motion energy and total variation remained close to teacher values. Maximum
state magnitude was 0.299 and 1.185 respectively. Their 16-step MSE was
0.00000038 and 0.00000371. The fixed distributed-field comparison remains
spatially coherent, although trajectory difference grows over the evaluation
horizon.

The initial stochastic-update comparison was confounded because a 0.5 fire rate
halved each cell's effective clock relative to the teacher. A corrected run used
a 0.05 teacher time step against the deterministic run's 0.1. It stayed bounded
below 0.77 and retained low 16-step error, but 512-step MSE reached 0.165 and
0.174 while motion decayed to roughly one-sixth of the teacher. An 8,000-step
deterministic run also failed, reaching state magnitudes above 57 and 81. More
optimization was not a stability solution.

The clean `c31be68` mask-aligned run applied the original teacher residual
through recorded 0.5-rate per-cell masks and supplied each same mask to the NCA.
It trained for 2,000 steps in 65 seconds and passed all 512-step checks. Recorded
mask central/distributed MSE was 0.00212/0.00711; unseen-mask MSE was
0.00211/0.00705. All 16-step MSE values were below 0.000005, maximum state
magnitude remained below 0.65, and predicted motion, total variation, and
spectral centroid remained close to teacher values. Exact state-trajectory
replay passed for both initializations. Human review found coherent behavior and
no recurrence of the prior grid-scale phase noise. The retained artifacts are
under `outputs/stage1-mask-aligned-rapid-c31be68`.

This result attributes the prior asynchronous failure to supervision against a
teacher with a different local update schedule. Similar performance under an
unseen mask seed argues against memorization of one stochastic sequence. It
establishes asynchronous teacher imitation only; autonomous material behavior
and audio control remain unproven.

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

- No NaN or Inf.
- Maximum state magnitude remains below 2.0 over the complete evaluation.
- The 16-step MSE remains below `1e-4`.
- The 512-step trajectory MSE remains below `0.02`.
- Predicted motion remains within a factor of four of teacher motion.
- Total variation and spatial spectral centroid remain within a factor of two
  of the teacher.
- The complete state trajectory replays exactly from the same initial state and
  recorded mask seed.
- A different recorded mask seed remains stable, accurate, and coherent.
- Human review confirms spatially coherent dynamics rather than static output,
  grid-scale phase noise, or structureless flicker.

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

Frequency-band separation is an exploratory measurement, not a requirement for
this plumbing gate. Stage 2 establishes that hand control works before learned
conditioning; it does not require every extracted feature to produce a distinct
visual response.

### Failure interpretation

Failure here means the feature pipeline or control surface is inadequate. It is
premature to train learned conditioning.

### Maximum wall-clock

No training. A complete probe should render in less than 10 minutes and ideally
near real time.

### Result

Stage 2 passes. The hand-controlled audio steering layer
freezes the retained Stage 1 `c31be68` rule and strictly loads its checkpoint
with fp32, MPS, and no fallback. It uses the recorded 0.5 fire masks, the
distributed initialization with state seed 4107 and mask seed 4206, fixes the Stage 1 render
mapping, and renders 24 fps synchronized H.264/AAC MP4 with the original WAV
attached. Probes and counterfactuals are implemented exactly: silence, constant,
isolated impulse, alternating energy, stepped bands, and a combined review
under correct, exact replay, silent, shuffled, and frozen mean conditions,
from the same state and masks with blind files carrying identical attached
audio.

Controls are deterministic and non-learned. Audio features are absolute RMS,
onset, spectral flux, low (80-300 Hz), and high (2500-6000 Hz) bands. RMS
scales update speed over 0.35-1.5 against fixed step recurrence; onset injects
a bounded channel-1 perturbation; band energy uses a broad low-band perturbation
and compact signed high-band wavelets; the current sustained variant keeps
spectral pressure at amplitude 0.008 per frame. There is no training, no
architecture change, and no pixel or audio postprocessing.

The safe candidate `outputs/stage2-smoke-candidate` passes all automated checks
except spectral spatial after metric correction. Measured: high/low energy
motion ratio 2.4009; impulse structural response ratio 2.2501; visible lag 4
updates / 1 frame / 41.67 ms; correct max state 0.7440; shuffled max 1.1318;
exact replay true; mean luminance contribution about 7%; correct-vs-silent
structural divergence about 0.389 and correct-vs-shuffled about 0.110. Human
blind review chose B, with the concealed mapping B=correct, and described the
response as mostly volume driven, movement accelerating with volume,
irrespective of high or low tones.

A stronger-amplitude variant `outputs/stage2-smoke-candidate-2` raised
amplitude to 0.18, pushing shuffled max to 2.0424 without improving spectral
separation; it is rejected and retained as a decision-relevant failure. A
transition-wavelet variant `outputs/stage2-smoke-spectral-wavelets` was bounded
and exact with B=correct again, but human review found everything seemed about
the same and could not tell what changed; its spectral spatial average failed
at roughly 0.77% centroid separation, so it is rejected as a spectral solution.
The current sustained spectral variant
`outputs/stage2-smoke-spectral-sustain` is bounded (correct max 0.7414,
shuffled 0.8509), exact replay passes, and all non-spectral checks pass, but
spectral spatial separation still fails at roughly 0.88% centroid separation.
Human review said it seemed roughly the same, would require close attention,
and could not reliably determine the distinction.

The deployment checkpoint artifact `outputs/stage2-partial-deploy` reruns the
current sustained control after evidence-integrity fixes. Exact replay, bounds,
blind audio identity, onset, energy, lag, recovery, divergence, and
non-brightness checks pass. High/low energy motion ratio is 2.4059, impulse
response ratio is 2.2501, visible lag is 4 updates / 1 frame / 41.67 ms,
correct/shuffled maximum state magnitude is 0.7414/0.8510, and mean-luminance
contribution is 6.5%. Spectral spatial separation remains small at 0.95%
centroid difference. No additional human review was requested for this
packaging-only rerun because spectral control is not required by this stage's
gate.

The final artifact `outputs/stage2-loudness-gate` reruns the same deterministic
control after reconciling the automated checks with the original gate. All 12
checks pass. It reproduces the prior initial-state, fire-mask, and state-trajectory
hashes exactly, preserves the same measured response values, and packages valid
H.264/AAC media with byte-identical blind audio streams.

The human timing gate identified the correct blind condition as B twice, which
is better than chance for these reviews. Two same-order trials are not strong
statistical evidence: the blind permutation happened to be the same because the
evaluation seed is deterministic.

Interpretation: audio extraction, synchronization, playback, rendering,
counterfactual plumbing, RMS motion control, the silence distinction, prompt
onset response, safety, exact replay, and a non-brightness response are
established. These results satisfy all four Stage 2 gate criteria. Spectral
spatial control is not perceptually established and is deferred; it is not a
requirement for this plumbing stage. Stage 2 does not establish learned audio
conditioning, autonomous material behavior, or semantic audiovisual
interpretation.

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
