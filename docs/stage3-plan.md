# Stage 3 Plan: Minimum Viable Learned Audio Conditioning

## Purpose

This document defines the smallest coherent experiment that can answer the
Stage 3 question: does the NCA learn to use audio, rather than ignore it?

It intentionally uses one audio signal (RMS/loudness) because Stage 2 already
proved that RMS produces a visible, bounded, non-brightness hand-controlled
response on the retained Stage 1 rule. Onset, spectral flux, and band energy
are optional follow-ups and are not worked on until a single learned
RMS-conditioned result passes the gate below.

The goal is a quick honest test of the NCA approach, not an exhaustive
demonstration of multiple steering abilities.

## Spectral Steering Is Not a Requirement

Stage 2 tried three spectral-control variants:

1. a stronger-amplitude sustained variant (amplitude 0.18);
2. a transition-wavelet variant;
3. a sustained spectral variant after metric fixes.

Stronger perturbation pushed the shuffled maximum state magnitude to 2.0424
without improving perceptible separation. The wavelet variant was bounded and
exact but human review could not tell what changed. The sustained variant
remained at roughly 0.88-0.95% centroid separation, below perceptibility.

Conclusion: spectral steering is not a blocker for Stage 3 and receives no
further work before a learned RMS-conditioned result exists. It may be
revisited later based on actual model output.

RMS/loudness is not a problem to work around. It is the proven path that
Stage 3 is intentionally built on.

## Inputs

- One scalar audio feature per NCA update: absolute RMS from the existing
  `synesthete.audio` extraction. Fit any required scaling across the training
  set, preserve silence and absolute level, and save the scaling parameters.
- Reuse the existing per-substep interpolation
  (`interpolate_features`, column 0 = `rms`) so audio is sampled at the NCA
  update rate, not only at the video frame rate.
- No genome. No onset, spectral flux, centroid, or band features as model
  input for this slice. They remain available for later ablations.

## Forced Teacher

Build a deterministic forced teacher by extending the retained Stage 1
oscillatory reaction-diffusion teacher so RMS scales its time step over a safe
range. This directly matches the direction Stage 2 already validated, where
RMS scales update speed from 0.35 to 1.5.

The teacher must make the RMS signal alter trajectory or motion in an
explicit, visually legible way: silent segments settle or slow, loud segments
accelerate or excite. Generate paired `(audio, teacher trajectory)` clips
from the same initial states and recorded fire masks used in Stage 1.

The teacher equations, parameter range, audio clips, and seeds become part of
the test fixture and are saved with the checkpoint, exactly as in Stage 1.

## Architecture Delta

Stay consistent with `docs/technical-direction.md`: feature-wise modulation
of the update network's hidden layer, not direct state-channel multiplication
and not a hypernetwork.

```text
c_t     = a_t                          # scalar RMS, broadcast
gamma_t = MLP_gamma(c_t)               # shape [B, 96] or [96]
beta_t  = MLP_beta(c_t)                # shape [B, 96] or [96]

h_t     = activation(W_1 * P_t + b_1)  # existing Stage 0/1 path
h'_t    = (1 + gamma_t) * h_t + beta_t # broadcast across spatial positions
delta_t = W_2 * h'_t + b_2             # existing projection

S_(t+1) = S_t + step_size * fire_mask_t * bounded(delta_t)
```

The delta is minimal:

- keep the existing perception kernels, `1 x 1` `W_1` (64 -> 96) and `W_2`
  (96 -> 16), circular boundary, fire rate 0.5, step size 0.1;
- add a small conditioning MLP that maps one scalar to `(gamma, beta)` of
  width `hidden_channels = 96`;
- initialize the conditioning MLP so `gamma = beta = 0` at start, load the
  retained Stage 1 weights, and verify that zero audio initially reproduces
  the retained unconditioned rule;
- freeze nothing. Fine-tune the whole conditioned rule, including `W_1` and
  `W_2`, against the forced teacher.

Record the exact parameter count in the checkpoint metadata. The full
conditioned network should remain in the low tens of thousands of trainable
parameters.

## Losses

```text
L_traj    = trajectory_loss(model(state, correct_audio), target)
L_silent  = trajectory_loss(model(state, silence),       target)
L_shuffle = trajectory_loss(model(state, shuffled_audio), target)

L_audio   = relu(L_traj + margin - min(L_silent, L_shuffle))

L = L_traj + lambda_audio * L_audio + lambda_overflow * overflow_loss
```

- `trajectory_loss`: short-horizon state MSE against the forced teacher, with
  randomized rollout lengths as in Stage 1.
- `overflow_loss`: penalty for state values outside the documented safe
  range, as in Stage 1; not a hard clamp of the full state.
- `margin`: a small positive constant; tune only if the contrast collapses.
- `lambda_audio` is the key anti-audio-ignoring pressure. It is the
  trajectory-level analogue of the Stage 2 counterfactual and is the primary
  mechanism that prevents the model from reducing the task to mean
  brightness or to an audio-invariant rule.

Do not add spectral, onset, or diversity losses in this slice.

## Run Tiers

- **Smoke (1-3 min):** tiny step counts and short rollouts. Verify shapes,
  device placement, finite gradients, checkpointing, rendering, and exact
  replay. Do not judge aesthetics.
- **Rapid (<=10 min):** compare one variable against a baseline. Render fixed
  probes. Reject instability or an objective that rewards the wrong behavior.
  No hyperparameter sweeps.
- **Full local (1-2 hr):** train one justified candidate. Run the required
  correct, exact-repeat, silence, and time-shuffled conditions. Produce
  artifacts for a human go/no-go.

## Pass/Fail Checks

All evaluated from the same initial state and recorded fire-mask sequence.

- **Finite and bounded:** no NaN/Inf; maximum state magnitude remains below
  2.0 over an evaluation rollout well beyond the training horizon.
- **Audio matters:** correct audio predicts the teacher trajectory better
  than both full silence and time-shuffled audio, by a clear margin.
- **Visible difference:** correct audio versus silence and correct versus
  time-shuffled produce visibly different trajectories from the identical
  state and masks, not just numerically different losses.
- **Not merely brightness:** the difference survives mean-luminance removal;
  audio changes motion, organization, or regime rather than only global
  brightness.
- **Exact replay:** the complete state trajectory replays bit-for-bit from
  the same initial state, fire-mask seed, and audio.

One reliable audio steering relationship (RMS) is sufficient to pass Stage 3.

## Immediate Implementation Slice

The smallest first slice, in order:

1. Add a scalar-conditioned variant of the NCA: the FiLM `(gamma, beta)` path
   above, zero-initialized, into the existing `1 x 1` update network. Load the
   retained Stage 1 weights and verify that with `a_t = 0` the rollout
   reproduces that rule exactly (regression guard).
2. Implement the RMS-scaled-time-step teacher and an `(audio, trajectory)`
   pair generator reusing Stage 1 seeds, fire masks, and initialization.
3. Implement `L_traj`, `L_audio`, and `overflow_loss` plus the silence and
   time-shuffled counterfactual rollouts.
4. Run the smoke tier, then rapid, then one full run. Do not sweep.

## What Stage 3 Success Proves and Does Not Prove

Passing proves that a compact locally-trained NCA can learn to use a single
audio signal to materially alter a visual dynamical trajectory that a teacher
specifies. That is the core thing the prior two repositories failed to
establish.

It does not prove:

- autonomous material behavior (no teacher is the goal of Stage 4);
- novel visual synthesis;
- semantic or spectral audio interpretation;
- behavior across multiple audio features or real music;
- that the same approach will scale.

Onset, spectral flux, and band conditioning are explicit optional follow-ups
that only become relevant after this RMS-conditioned slice passes.
