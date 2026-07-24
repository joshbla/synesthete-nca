# Technical Direction

## Architectural Decision

The first implementation will be an audio-conditioned Neural Cellular
Automaton operating on a persistent 2D state. It is intended to be the smallest
system capable of answering the research question, not the smallest final
product architecture.

The baseline decisions are:

```text
grid                         96 x 96
state channels               16
visible channels             1 luminance channel
hidden state channels        15
perception                    identity, Sobel X, Sobel Y, Laplacian
update hidden width          96
update type                  stochastic asynchronous residual update
update step size             0.1
NCA updates per video frame  4-8
output frame rate            15 initially; compare 24 later
audio features               no more than 10 per aligned time window
genome                       4-8 persistent values, deferred until one rule works
boundary                     periodic/circular baseline
training precision           fp32 baseline
```

## State and Recurrence

Let the persistent state be:

```text
S_t in R^(B x 16 x 96 x 96)
```

The first channel is rendered as luminance. The other 15 channels are latent
cell state used for communication, memory, phase, and local organization. They
are not decoded by another network.

The same update rule is applied repeatedly:

```text
S_(t+1) = F_theta(S_t, a_t, g, m_t)
```

where:

- `a_t` is the audio feature vector aligned to update time;
- `g` is a persistent genome/style vector;
- `m_t` is a stochastic cell-update mask;
- `theta` is shared across every cell and every time step.

This parameter sharing is the primary inductive bias. The model cannot store a
separate renderer for each location or frame. Global behavior must arise from
local state propagation.

## Perception

Each cell observes four fixed transforms of every state channel:

```text
P_t = concat(
    S_t,
    SobelX(S_t),
    SobelY(S_t),
    Laplacian(S_t)
)
```

For 16 state channels, perception has 64 channels. Fixed filters keep the
learned network small and expose useful local quantities:

- identity: local state;
- Sobel X/Y: oriented gradients and local motion fronts;
- Laplacian: local curvature, diffusion, and reaction-diffusion-like behavior.

The baseline uses circular padding so boundaries do not become a privileged
visual frame. Reflective and fixed boundaries can be tested later because they
create materially different dynamics.

Multi-scale perception is deferred. DyNCA found it useful at larger grids, but
adding dilated or pooled perception before the local baseline is measured would
blur whether long-range communication is actually necessary at `96 x 96`.

## Update Network

The local update is a two-layer per-cell network implemented as `1 x 1`
convolutions:

```text
h_t     = activation(W_1 * P_t + b_1)
h'_t    = condition(h_t, a_t, g)
delta_t = W_2 * h'_t + b_2

S_(t+1) = S_t + step_size * fire_mask_t * bounded(delta_t)
```

`W_1` maps 64 perception values to 96 hidden values. `W_2` maps 96 hidden
values back to 16 state deltas.

The 96-unit hidden width is intentionally larger than required by parameter
count. Research on emergent NCA dynamics found that models tended to lose
spontaneous motion when update hidden width was less than roughly twice the
state-channel count. Here the ratio is `96 / 16 = 6`. That empirical finding is
a design clue from texture experiments, not a theorem that guarantees useful
dynamics.

Approximate parameters before conditioning:

```text
64 -> 96 layer:  64 * 96 + 96  = 6,240
96 -> 16 layer:  96 * 16 + 16  = 1,552
----------------------------------------
base update network                  7,792
```

An audio/genome modulation network should bring the complete baseline to only
about 20,000-30,000 trainable parameters, depending on final feature and genome
dimensions. Exact count must be recorded by the implementation.

The Stage 0 implementation initializes the final update projection and bias to
exactly zero. This makes the initial automaton an identity recurrence instead of
an immediately explosive random dynamical system. Training can then grow useful
updates gradually.

## Audio Representation

The first system should not learn an audio foundation model. It should use a
small time-aligned vector whose meaning can be inspected.

A likely eight-to-ten-dimensional schema is:

- absolute RMS or smoothed energy;
- log RMS;
- onset strength;
- spectral flux;
- spectral centroid;
- four or five broad spectral-band energies.

The exact schema will be fixed before training and saved with every checkpoint.

Important preprocessing rules:

- Preserve absolute RMS and silence information.
- Use dataset-level or explicitly fitted normalization for features that need
  scaling; do not independently normalize every clip.
- Compute features on a timeline aligned to rendered frames.
- Interpolate features over the NCA updates between frames so control does not
  jump only at frame boundaries.
- Include synthetic probes with exact known changes before using real music.

Raw waveform conditioning is deferred because it increases temporal context and
encoder complexity before the behavior of low-level controls is understood.
Learned embeddings can be introduced if the handcrafted schema demonstrably
cannot express a desired audio distinction.

## Recommended Audio Injection: Feature-Wise Modulation

The baseline should use a small conditioning MLP to produce a scale and shift
for the update network's hidden channels:

```text
c_t = concat(a_t, g)
gamma_t, beta_t = MLP(c_t)

h'_t = (1 + gamma_t) * h_t + beta_t
```

`gamma` and `beta` are broadcast across spatial positions. The cell's local
perception still determines its update, while audio changes how the shared rule
interprets that perception.

This is preferable to the simplest published music-NCA technique of multiplying
selected state channels directly by RMS. Direct multiplication proves that an
NCA can react in real time, but it does not learn the relationship and can
destabilize the state. It also encourages a shallow response in which audio
changes magnitude without changing organization.

Feature-wise modulation has several advantages:

- audio affects the update rule rather than postprocessing the image;
- different audio features can control different hidden computations;
- modulation can begin near identity;
- ablations can set `gamma` and `beta` to zero cleanly;
- the same state can be rolled under different audio counterfactuals.

It does not guarantee that the model will use audio. The loss and evaluation
must still penalize audio invariance.

### Alternatives to compare only if needed

- **Concatenate audio to perception at every cell.** Simpler, but gives the
  update network a direct global bias that may collapse into luminance control.
- **Generate update weights with a hypernetwork.** More expressive, but can
  make the system unstable because the dynamical rule itself changes at every
  audio step.
- **Cross-attention to audio tokens.** Appropriate for richer temporal context,
  but unnecessary for fewer than ten aligned features and much more expensive.
- **Direct state-channel multiplication.** Useful as a hand-controlled plumbing
  baseline, not as the learned final mechanism.

## Genome or Style Vector

A small vector `g` can describe a persistent behavioral tendency for an entire
run. Unlike frame-level randomness, it should remain constant through the clip.
Its purpose is to separate "what kind of material is this?" from "what is the
audio doing now?"

The genome is not required for the first stable NCA. It should be introduced
only after one rule passes audio counterfactuals, because otherwise it provides
another condition the model can use instead of audio.

When introduced, the genome should:

- use 4-8 dimensions initially;
- be recorded with the run;
- share the same feature-wise modulation path as audio or use a clearly
  separated branch;
- be tested by interpolation, not just random sampling;
- be evaluated for both diversity and preservation of audio causality.

## Asynchronous Updates

At each NCA step, each cell updates with a probability such as 0.5. The binary
fire mask breaks perfect grid synchronization and is common in NCA work.

```text
m_t[x, y] ~ Bernoulli(fire_rate)
```

For reproducibility, the random generator state is part of the run seed. Exact
replays must use the same masks. Evaluation should also compare deterministic
full updates against stochastic updates to determine whether observed novelty
comes from the learned rule or merely from injected noise.

Stage 1 found that this comparison is material rather than cosmetic. The
oscillatory teacher can be imitated coherently with deterministic full-cell
updates, while the first 0.5-rate asynchronous run introduces phase noise and
long-run drift. That comparison also changes effective local time: without
explicit compensation, each cell advances half as often as the synchronous
teacher. A clock-compensated comparison is therefore required before attributing
the failure to asynchronous noise alone. That comparison used half the teacher
time step and remained bounded, but motion decayed to roughly one-sixth of the
teacher over 512 updates and trajectory MSE remained high. The asynchronous
baseline remains the intended architecture and must pass before audio
conditioning begins.

## Initialization

At least two initialization families should be supported from the beginning:

1. **Central seed:** hidden channels activated near the center, suitable for
   growth and morphogenesis tests.
2. **Low-amplitude field noise:** state distributed across the grid, suitable
   for material and texture formation.

The visible channel should begin neutral. Seed definitions must be versioned
and saved. Comparing different rules on different unrecorded initial states
would make trajectory differences uninterpretable.

## Stability Controls

An NCA is an iterated nonlinear system. A numerically valid one-step update can
still diverge after thousands of steps.

The baseline should include:

- bounded or scaled state deltas;
- a learned or configured global step size with a safe initial value;
- overflow loss for state values outside a chosen range;
- gradient clipping;
- per-channel statistics during training and long inference;
- NaN/Inf hard failure rather than silent replacement;
- randomized rollout lengths so the rule does not optimize one fixed horizon;
- long-rollout evaluation substantially beyond the training horizon;
- optional damage tests after basic stability is established.

Hard-clamping the entire hidden state on every step may hide unstable dynamics
and create saturated attractors. Prefer penalties and bounded deltas initially;
use clipping only as an explicit experimental condition.

## Rendering

The visible channel should be mapped to luminance with a fixed documented
function. The renderer must not contain audio-reactive postprocessing.

A simple contract is:

```text
luminance_t = sigmoid(S_t[:, visible_channel])
```

or a fixed centered mapping if the state is explicitly bounded. The choice must
remain constant across experiments so apparent improvement is not a contrast
change.

Stage 1 uses the fixed centered mapping
`luminance_t = clamp(S_t[:, visible_channel] + 0.5, 0, 1)`. Native target,
prediction, and raw absolute-difference arrays are saved before GIF encoding;
only the human-facing difference image is amplified and clipped.

Video assembly, resizing, and audio muxing occur after the NCA rollout. Upscale
filters must be fixed and identified in artifacts. Evaluation metrics should be
computed on the native grid before video compression.

## Training Recurrence Versus Inference Recurrence

These must remain conceptually separate:

- **Inference recurrence:** the state can be advanced indefinitely while only
  the latest state is retained. Memory is nearly constant with duration.
- **Training recurrence:** gradients through an unrolled window require saved
  intermediate activations. Memory grows with the credit-assignment horizon.

A model trained on 16-step windows can still be rendered for 10,000 steps, but
there is no guarantee it remains stable or meaningful. Long inference tests are
therefore mandatory for every checkpoint.

If persistent states are carried across optimizer steps with `.detach()`, the
system experiences long histories but gradients do not cross the detach point.
This is truncated BPTT and must be documented as such.

## Checkpoint Contract

Every checkpoint must include or point immutably to:

- all model tensor shapes and parameter count;
- state-channel meanings and visible-channel index;
- perception kernels and boundary mode;
- audio feature schema and normalization statistics;
- genome schema;
- update rate, step size, and updates per rendered frame;
- training rollout distribution;
- loss names and weights;
- optimizer state when resumable;
- random seeds;
- source revision and complete configuration;
- evaluation results for the required fixed probes.

Loading incompatible metadata must fail. There will be no silent random-weight
fallback.

## Training Objective Is an Open Design Question

The architecture specifies how a material can evolve. It does not by itself say
what the material should learn.

Relevant objective families include:

- matching one or more static texture statistics while encouraging dynamics;
- imitating a simple forced reaction-diffusion or vector-field system as a
  bootstrap control;
- boundedness and persistence constraints;
- target ranges for motion energy and spatial frequency structure;
- correct-versus-shuffled trajectory contrast;
- onset-response and silence-settling objectives;
- seed/genome diversity without loss of stability;
- human selection among candidates after objective constraints are met.

Each has failure modes. A target exemplar can reduce novelty to imitation.
Handcrafted complexity metrics can reward noise. Audio-motion correlation can
reward global flicker. Human selection is expressive but expensive and hard to
reproduce.

The experiment plan treats objective design as a sequence of falsifiable steps
rather than pretending one weighted sum is already known to be correct.

## Deferred Architectures

### Latent neural ODE or SDE

A compact continuous latent trajectory with a decoder could provide smooth
audio-controlled motion. It is a reasonable fallback if NCA local interactions
cannot maintain global organization. It is deferred because the decoder would
again own much of the visual prior, while the NCA directly represents the
material.

### Coordinate neural field

A field `f(x, y, t, audio, z)` could render at arbitrary resolution. It has no
intrinsic persistent spatial state unless recurrence is added, and per-pixel MLP
throughput on MPS is uncertain. It may later serve as a renderer for NCA state.

### Hypernetwork

Generating neural-field or NCA weights from audio could produce powerful regime
changes, but rapidly changing dynamical weights pose a severe stability problem
and require a learned manifold of valid rules.

### Diffusion or autoregressive video

These model broad distributions well but return to the expensive
frame/token-generation formulation and the audio-ignoring prior. They are not
appropriate until the project needs a much broader visual distribution and has
the data and compute to support it.

### Quality-diversity search

An archive of varied NCA rules is attractive because novelty can become an
explicit search objective. It is deferred until one compact rule and its metrics
are valid. Searching thousands of invalid or easily gamed candidates would
scale confusion rather than discovery.
