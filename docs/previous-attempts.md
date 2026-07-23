# Previous Attempts

## Why This Postmortem Exists

This repository follows two earlier projects in the same workspace:

- `../synesthete`: a full audio-to-video system based on a spatial VAE
  and a latent diffusion transformer;
- `../baby-synesthete`: a reduced audio-to-image conditional diffusion
  experiment.

Both projects produced useful evidence, but neither achieved the intended
experience. Their most important shared failure was not an inability to produce
visual output. Both eventually produced plausible visual structure. They failed
to make the supplied audio determine that structure strongly enough.

This distinction matters. Improving image quality does not necessarily improve
audio grounding. The new project should not rediscover that after another long
training run.

## Original Intent

The initial vision was broader than a conventional spectrum visualizer. Given
audio, the model would create visual motion that was aesthetically and perhaps
semantically representative of the sound. It should make a creative choice
rather than merely calculate bars, waveforms, or particles from frequencies.

The desired output implicitly combined several research problems:

1. Extract useful temporal structure from audio.
2. Generate visually coherent images.
3. Maintain identity and motion across video frames.
4. Make the image and motion causally depend on the audio.
5. Allow multiple valid visual interpretations of one sound.
6. Produce results beyond the authored training visualizers.
7. Train and render within a local Apple Silicon budget.

Each problem is tractable in isolation at a small scale. Combining them in one
from-scratch model made failures difficult to diagnose and allowed progress on
one dimension to conceal failure on another.

## Attempt One: Synesthete

### Initial direct-regression model

The earliest architecture directly regressed from audio to RGB video frames.
It used an audio encoder, temporal transformer, and convolutional frame decoder
with pixel MSE.

That formulation encountered the expected conditional-mean problem. The
training data deliberately allowed multiple visualizers and styles for similar
audio. For a squared-error regressor, the optimal prediction is the conditional
mean:

```text
f*(a) = E[x | a]
```

If the same audio can correspond to a red circle, blue waveform, or green set
of bars, their pixel average is not another valid style. It is low-contrast,
blurry output. Project notes called this result "gray sludge."

This diagnosis was conceptually correct. A one-to-many conditional distribution
cannot generally be represented by a deterministic pixel regressor without
averaging its modes.

### Latent-diffusion redesign

The project moved to two learned stages.

#### Spatial VAE

The VAE compressed each `128 x 128 x 3` frame to a `256 x 8 x 8` spatial
latent. Its encoder used four stride-two reductions and its decoder reversed the
process. The training objective combined reconstruction MSE and KL regularity.

The surviving VAE checkpoint contains exactly 1,514,442 tensor values and is
about 5.8 MB. Diagnostic reconstructions showed that it weakened some
audio-linked visual differences but retained them. It was not the principal
reason conditioning failed.

#### Diffusion transformer

The diffusion model flattened the `8 x 8` latent to 64 tokens. A transformer
decoder denoised those tokens while cross-attending to audio features and a
style token. It also accepted diffusion time, frame index, and the previous
frame latent.

The surviving checkpoint uses:

- model width `d_model = 256`;
- positional embedding shape `(1, 64, 256)`;
- six transformer decoder layers;
- 10,232,990 stored tensor values;
- approximately 39 MB on disk.

This is smaller than the later default configuration, which specifies
`d_model = 512`. The checkpoint and default path therefore do not describe the
same system.

#### Audio features

The later system replaced poorly aligned whole-waveform conditioning with a
frame-aligned feature timeline. Per frame, it computed measurements such as:

- RMS and optional absolute RMS;
- zero-crossing rate;
- spectral centroid and rolloff;
- flatness;
- broad spectral-band energies;
- optional onset strength and spectral flux.

This repaired an identifiability problem. The model could now know which audio
window corresponded to each visual frame. It did not, by itself, make using
that window necessary.

#### Synthetic data engine

Training pairs were generated in memory:

```text
procedural audio
    -> frame-aligned features
    -> randomly selected procedural visualizer
    -> RGB frames
    -> frozen VAE latents
    -> diffusion training examples
```

The visualizer library grew to include pulses, bars, waveforms, particles,
contours, trails, kaleidoscopic transformations, augmentations, and composites.
This provided variety but also changed the statistical problem. Visual style
and visualizer identity explained a large amount of output variance that the
audio did not explain. The unconditional image prior became increasingly useful.

### Why ordinary conditional diffusion ignored audio

The primary diffusion objective was noise-prediction MSE:

```text
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

L_main = ||epsilon_theta(x_t, t, audio, other_conditions) - epsilon||^2
```

The audio was available to the model, but the objective did not directly
require it. A denoiser could reduce loss substantially by learning the latent
distribution of visualizer-looking frames and using noisy image content,
diffusion time, style, frame position, and previous state. If audio contributed
only a small additional reduction, optimization could effectively disregard it.

This is the core conditional-generation shortcut:

> A condition is used reliably only when it is identifiable and when ignoring
> it is expensive under the training objective.

The project initially addressed identifiability but not necessity.

### Features added before the core proof

Several mechanisms were added in succession:

- frame-aligned features;
- a clip-level style latent and style dropout;
- more diverse visualizer programs;
- compositing and augmentation;
- contiguous training snippets;
- frame-index embeddings;
- previous-latent conditioning;
- robustness noise and dropout for the previous latent;
- an audio-latent matcher;
- classifier-free audio guidance;
- correct-versus-shuffled audio pressure;
- silence and track-dropout probes.

Many are defensible independently. Together, they made diagnosis difficult.
Most were introduced before the simplest question had been answered: can the
model visibly respond to audio at all?

The known-good bootstrap ultimately disabled much of the generality machinery.
It used only two deterministic debug visualizers, explicit silence/dropout
audio, no style variation, a smaller transformer, and strong audio guidance.

### The change that finally mattered

The decisive objective compared correct and shuffled audio:

```text
L_good = MSE(epsilon_theta(x_t, t, audio), epsilon)
L_bad  = MSE(epsilon_theta(x_t, t, shuffled_audio), epsilon)

L_shuffle = relu(L_good + margin - L_bad)
L_total   = L_good + weight * L_shuffle
```

If the model ignores audio, `L_good` and `L_bad` are nearly equal, so the hinge
penalty remains active. The model must make correct audio more useful than an
incorrect condition to reduce the total loss.

Silence and explicit track dropouts provided a human-auditable training and
evaluation signal. Disabling per-clip feature normalization preserved absolute
energy so silence could not be normalized into an ordinary-looking segment.

### What was actually achieved

After an initial short run, the correct-versus-shuffled output differed by only
1.5% in pixel space and was visually indistinguishable. After resuming to 33
total epochs, with training loss falling from roughly 0.65 to 0.20, the measured
difference reached 6.9%.

The human-visible behavior was:

```text
silence -> dark
sound   -> not dark
```

This was the first defensible evidence that the diffusion model used audio. It
did not demonstrate rhythmic synchronization, spectral-band differentiation,
strong geometric control, semantic correspondence, or general visual synthesis.
The output remained fuzzy and visually much less structured than the authored
debug targets.

The result should be described as a narrow proof of coarse audio causality, not
as a completed audio-to-video generator.

### Temporal behavior

Previous-latent conditioning improved continuity but introduced visible lag.
The system was trained using a previous ground-truth latent and generated using
its own previous estimate, creating a train/inference mismatch. Removing the
previous-latent contribution at inference reduced the delay.

This exposed another tradeoff: temporal smoothing is effectively a low-pass
filter. It can suppress flicker while also suppressing the prompt response that
an audio-reactive system needs.

### Compute and pipeline inefficiency

The successful configuration used 90-frame clips at 128 pixels and 15 fps. For
each generated training clip, the data pipeline rendered and VAE-encoded all 90
frames, then yielded a short contiguous sequence for optimization. Data loading
used no worker processes in the surviving path, so procedural audio, feature
extraction, visual rendering, and VAE encoding could serialize around training.

Inference was inherently sequential. A six-second clip could require:

```text
90 frames x 15 diffusion steps x 2 classifier-free-guidance passes
= 2,700 transformer forward passes
```

The latent representation made memory manageable, but it did not make the total
process fast. This distinction is important for the new project: a model fitting
in memory says little about whether it supports the desired iteration loop.

### Engineering failures

The conceptual difficulty was compounded by implementation drift:

- The default configuration specifies 256-pixel data and a 512-wide model,
  while the surviving VAE/diffusion checkpoint path is the 128-pixel,
  256-wide bootstrap system.
- Checkpoint loading catches broad exceptions and may continue with random
  weights, turning an architecture mismatch into misleading output rather than
  a hard error.
- The default configuration disables shuffled-audio pressure, disables the
  matcher, and does not use silence/dropout audio. The only mechanisms known to
  produce visible conditioning were not the normal path.
- One evaluation suite refers to configuration outside its valid scope and
  cannot run as shipped.
- The README describes broad visual-generation success more strongly than the
  surviving evidence supports.

These problems did not cause the central conditioning shortcut, but they made
the system harder to reproduce and easier to misinterpret.

## Attempt Two: Baby Synesthete

### Goal

The baby project removed the VAE, temporal model, cross-attention, and video. It
trained a small conditional U-Net to produce one `64 x 64` image from one second
of synthetic audio.

The intended mapping was explicit:

```text
frequency    -> hue
waveform     -> shape
amplitude    -> size and brightness
harmonics    -> concentric rings
```

Audio was converted to a `64 x 64` mel spectrogram and concatenated as a fourth
spatial channel beside the three noisy image channels. A DDPM noise objective
trained the U-Net, and DDIM used 50-100 forward passes to sample an image.

### What ran

The surviving checkpoints contain exactly 2,156,099 parameters, corresponding
to a U-Net base width of 64. Training continued to 15,000 steps in approximately
66 minutes. Preview artifacts exist for every 1,000 steps.

Those previews are informative. Early generations are colored noise. Later
generations develop coherent rings, enclosed forms, gradients, and colorful
spatial structure. The model did learn how images in its target distribution
looked.

It did not reliably make those images match their supplied targets. Across
different conditioned examples, generated hue and geometry remained much more
similar than the deterministic target images. The correct diagnosis is weak or
ignored conditioning, not complete image collapse.

### Why diffusion was unnecessary

Unlike the first project's randomized visualizer data, the baby mapping was
almost deterministic. Given the synthesis parameters, there was one correct
target image. Its conditional distribution was approximately unimodal:

```text
p(image | audio) approximately one target
```

Diffusion is valuable when many distinct outputs are valid for the same
condition and a point estimate would average them. Here, a direct conditional
regressor could have learned the mapping in one forward pass. Diffusion added a
noisy training objective and a long sampling loop without solving a real
multimodality problem.

This does not mean diffusion always ignores conditions. It means the project
paid diffusion's complexity cost where its principal benefit was not needed.

### Why spectrogram concatenation was poorly matched

The mel spectrogram was resized to match image dimensions and concatenated with
the image. This made the tensors compatible but not semantically aligned:

- spectrogram rows represented frequency;
- spectrogram columns represented audio time;
- image rows and columns represented visual space.

A `3 x 3` image convolution therefore treated neighboring time-frequency bins
as if they occupied neighboring image pixels. The U-Net had to discover a
global cross-domain relationship through operations biased toward local spatial
correspondence.

A compact audio vector, feature-wise modulation, or cross-attention would have
respected the fact that audio and image coordinates have different meanings.
The new NCA direction uses time-aligned features to modulate the shared update
rule rather than pretending the audio is another image.

### Amplitude information was weakened

Each spectrogram was independently min-max normalized. This discards absolute
scale, even though amplitude controlled target size and brightness. Some
amplitude cues may remain through waveform composition and numerical details,
but the preprocessing removed the most direct measurement of the target factor.

This repeated a lesson from the full project: normalization decisions can erase
the exact absolute signal needed for silence or energy response.

### Code and checkpoint mismatch

The final committed model defaults to base width 32, approximately 677,000
parameters. Every existing checkpoint uses base width 64 and 2,156,099
parameters. The generation script constructs the smaller model and therefore
cannot load the saved checkpoints.

The committed README describes the smaller, 5,000-step version, but the actual
artifacts came from the larger, 15,000-step run. The optimization changes were
made while the previous training process was already running and were not
subsequently validated end to end.

## Shared Root Cause

The two projects looked different but reached the same statistical shortcut:

1. The model learned a visual prior.
2. The condition was available.
3. The main loss could improve substantially without using the condition.
4. Attractive or coherent output was mistaken for progress on conditioning.

The full project eventually demonstrated that explicit counterfactual pressure
could change this. The baby project never included that pressure or a rigorous
same-noise, different-audio evaluation.

## Lessons Retained

### 1. Separate visual quality from audio causality

Evaluate them independently from the beginning. A compelling autonomous NCA
can still ignore audio. Its visual quality is not evidence of control.

### 2. Use identical-state counterfactuals

Start from the same exact state and random seed, then change only the audio.
Compare correct audio with silence, time-shuffled audio, batch-shuffled audio,
and feature ablations. This is more diagnostic than comparing unrelated runs.

### 3. Make conditioning necessary

If ordinary task loss does not distinguish correct from incorrect audio, add an
objective that does. The prior shuffle hinge is one candidate. Trajectory-level
contrastive losses may be more meaningful for the recurrent NCA.

### 4. Preserve absolute energy

RMS, silence, and gain are meaningful controls. Do not erase them through
per-clip normalization and later try to reconstruct them indirectly.

### 5. Bootstrap before generalizing

First prove life. Then prove hand-controlled steering. Then learn steering.
Only after these pass should the system add genome diversity, real music, large
grids, learned audio encoders, or cloud-scale search.

### 6. Match representation to the problem

Use persistent recurrence for persistent material. Use audio features as audio
features, not as fake image coordinates. Use diffusion only for a truly
multimodal distribution that needs sampling.

### 7. Hard-fail invalid experiments

Checkpoints must record and validate architecture, feature schema, and training
configuration. Missing or incompatible state must stop evaluation. Defaults
must represent the known-good path.

### 8. Optimize the iteration loop, not only peak quality

The target is evidence in minutes and meaningful local output within two hours.
A model that barely fits but needs thousands of serial sampling passes is not a
good local research instrument.

## Evidence Carried Forward

The previous work did not establish general audio-visual synthesis. It did
establish four useful facts:

1. The target machine can train small spatial generative models.
2. A small VAE can preserve coarse audio-linked visual differences.
3. Frame-aligned low-level audio features are sufficient for at least a
   silence-versus-sound distinction.
4. Explicit correct-versus-shuffled pressure can turn an ignored condition into
   a measurable one.

The new project starts from those facts while changing the visual
representation from generated frames to a persistent learned dynamical system.
