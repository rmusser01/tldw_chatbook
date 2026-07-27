# STT INT8 Artifact Qualification Design

**Status:** Approved for implementation planning

**Backlog task:** TASK-593 — Qualify Parakeet v2 and v3 INT8 artifacts

**Governing decision:** [ADR-025](../../../backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md)

**Reference design:** [Cross-Platform STT Runtimes and Shared Model Artifacts](2026-07-23-stt-parakeet-onnx-transcribe-cpp-design.md)

## Purpose

Produce reproducible evidence about whether the proposed stock Parakeet v2 and
v3 INT8 ONNX artifacts are acceptable artifact candidates relative to their
F32 references and the faster-whisper base/int8 baseline.

The current macOS machine is the initial reference machine for throughput and
memory evidence. This work does not claim Windows or Linux qualification and
does not open the semantic-default, legacy-removal, or release gates.

## Scope

This task adds an isolated evaluation harness, immutable corpus and model
manifests, deterministic metrics, fail-closed gate evaluation, and versioned
macOS evidence.

The evaluated populations cover:

- clean and deterministically noisy English;
- silence and non-speech controls;
- short, beyond-direct-input-limit, ten-minute, and long-form stress media;
- Spanish, French, German, Polish, Greek, Russian, and Ukrainian Parakeet v3
  slices;
- stock Parakeet v2/v3 INT8 and F32 artifacts; and
- the local faster-whisper base/int8 comparison baseline.

## Non-goals

- Changing production routing, provider registration, or defaults.
- Adding a production model downloader, artifact store, or browser.
- Removing NeMo or MLX providers.
- Claiming cross-platform qualification from one macOS run.
- Treating a task-specific artifact result as a complete release-promotion
  decision.
- Mirroring large public speech datasets in this repository.

## Approaches considered

### Hybrid immutable corpus — selected

Use a pinned FLEURS snapshot for the shared multilingual core, a pinned
Common Voice archive obtained by the evaluator through Mozilla's distribution
boundary for English accent coverage, and deterministic generated
silence/noise/long-form fixtures.

This provides one comparable multilingual core while retaining the English
accent and robustness coverage required by the governing design.

### FLEURS only

This is operationally simpler and covers all target languages, but its English
slice is not sufficient evidence for the required accent and real-speaker
variation.

### Multiple independent benchmark suites

This broadens external validity but multiplies acquisition, licensing,
normalization, and pairing complexity. It is not justified for the first
artifact-qualification task.

## Corpus and acquisition

### Immutable sources

The corpus manifest records, for every external source:

- stable source identifier and immutable revision;
- source URL and license;
- source archive or shard size and SHA-256;
- split, locale, and stable upstream sample identifier;
- expected transcript and its source field;
- prepared audio size and SHA-256;
- sample rate, channels, duration, and encoding; and
- evaluation tags such as clean, accent, noise, silence, long-form, or stress.

FLEURS is pinned to an immutable repository commit rather than a moving branch.
Only the required language shards and fixed sample IDs are prepared. The
manifest uses the FLEURS test split and never changes its population after
model output has been observed.

Common Voice is not mirrored or fetched implicitly. The evaluator explicitly
obtains the declared Mozilla Data Collective snapshot, supplies its local
archive to the preparation command, and the harness verifies the expected
archive digest before reading it.

Only tiny synthetic fixtures used by automated tests live in the repository.
Prepared benchmark audio remains outside Git and is reconstructable from the
manifest.

### Deterministic derived fixtures

Noise, silence, beyond-limit, and long-form media are derived by a versioned
recipe. A recipe records its source sample IDs, ordered concatenation, silence
durations, gain, deterministic random seed, noise source and license, sample
rate, channel conversion, output size, and digest.

The preparation phase computes the output and refuses to continue when it does
not match the manifest. Generated perturbations supplement rather than replace
real-speaker accent evidence.

### Statistical sufficiency

Population membership and minimum sample/reference-unit counts are declared
before inference. A run is incomplete when any declared minimum is unmet.
Samples cannot be dropped after observing a hypothesis, error, or runtime
failure.

## Model and runtime manifest

Each model variant records:

- provider, model, precision, and local-only execution identity;
- immutable upstream repository and revision;
- license;
- every required local file name, byte size, and SHA-256;
- Python, `onnx-asr`, ONNX Runtime, faster-whisper, and execution-provider
  versions;
- CPU thread settings;
- expected capabilities, including timestamp support; and
- the exact VAD artifact, revision, files, sizes, hashes, and settings required
  for long-form evaluation.

The harness accepts only verified local files. It does not call a model hub or
allow an inference runtime to download a missing artifact.

The `onnx-asr==0.12.0` long-form path is invoked with VAD ASR
`batch_size=1`. Missing VAD provenance, a VAD hash mismatch, or an inability to
enforce the declared batch size makes long-form qualification incomplete.

## Run identity

A run fingerprint is the SHA-256 of canonical JSON containing:

- corpus manifest and derived-fixture recipe revisions;
- normalizer and metric revisions;
- model and VAD artifact identities;
- runtime and operating-system metadata;
- CPU, memory, execution provider, and thread settings;
- gate configuration;
- bootstrap seed and iteration count; and
- harness source revision.

Raw results from different fingerprints cannot be combined. A report embeds
its fingerprint and refuses mismatched inputs.

## Execution flow

### Prepare

1. Validate corpus and model manifests before filesystem or network work.
2. Display a dry-run preflight containing required sources, licenses, bytes,
   local inputs, destination, and available space.
3. Require an explicit preparation command before any transfer or extraction.
4. Verify source archive/shard size and digest.
5. Extract only declared contained regular files without following symlinks.
6. Normalize audio and build deterministic derived fixtures in staging.
7. Verify every prepared output before atomically publishing the prepared
   corpus.

Preparation failure leaves no usable partial corpus.

### Run

1. Validate the prepared corpus, selected model files, and complete run
   fingerprint.
2. Spawn one isolated child process for one model variant.
3. Load that variant once and process the fixed ordered sample population.
4. Emit one result record per sample with the exact reference, hypothesis,
   timings, timestamps, warnings, and model/runtime identity.
5. Record child OS high-water RSS and parent-sampled process-tree RSS.
6. Finish with one terminal run record containing counts and completeness.
7. Publish raw JSONL atomically; a partial temporary file is not reportable.

A child crash, timeout, malformed record, or missing sample makes the complete
run ineligible for comparison. The harness does not silently retry with another
engine or precision.

### Report

1. Verify every input fingerprint and terminal run record.
2. Apply the predeclared versioned normalizer for each language.
3. Compute per-sample edit/reference counts.
4. Aggregate WER and CER populations separately.
5. Compute fixed-seed paired-bootstrap confidence intervals.
6. Validate timestamp, silence, throughput, memory, and reuse gates.
7. Write deterministic machine-readable and human-readable reports atomically.

## Metrics and confidence intervals

For a population, corpus WER or CER is:

`sum(edit_count) / sum(reference_unit_count)`

It is not the arithmetic mean of per-sample percentages. WER and CER are
reported separately and never averaged together. Each language declares its
primary gate metric before a run; the other metric remains diagnostic.

Paired comparisons resample sample identities with replacement and recompute
both corpus rates and their delta for each bootstrap replicate. The first
version uses 10,000 replicates, a fixed manifest seed, and the percentile 95%
confidence interval. Both the point estimate and the adverse confidence bound
must meet the applicable ADR-025 threshold.

The report applies the existing thresholds without reinterpretation:

- v2 INT8 English aggregate WER is at most 1.0 absolute point worse than
  faster-whisper base/int8 and no English slice is more than 3 points worse;
- every v3 primary-metric population is at most 4 points worse than its
  faster-whisper baseline;
- the v3 macro average is at most 1.5 points worse, calculated only within a
  single primary-metric family;
- INT8 is within 0.5 aggregate points of F32 within each primary-metric
  population;
- silence produces no non-empty transcript;
- declared timestamps are monotonic, nonnegative, and within audio duration;
- warm CPU throughput is faster than real time on the macOS reference machine;
- peak INT8 heavy-process RSS is at most 3 GiB; and
- a 100-file same-model batch does not reload the model and post-warm-up RSS
  remains within 15%.

An unsupported timestamp capability is recorded explicitly. It is not
fabricated or treated as observed timestamp evidence.

## Memory measurement

The child records the operating system's process high-water RSS using an
OS-specific normalized probe. The parent samples RSS for the child process tree
through `psutil` during model load and inference.

Both values are preserved. Polling alone is not accepted as the peak, and
divergent values are reported rather than silently reconciled. On the initial
macOS reference run, Darwin units and conversion are captured in the report.
Future Windows/Linux probes require native validation before their results can
be compared or used as platform evidence.

## Decision model

Every artifact and language receives one of:

- `pass`: all task-specific evidence and thresholds passed;
- `fail`: complete valid evidence exceeded at least one threshold; or
- `incomplete`: required evidence, provenance, pairing, or capability was
  absent or invalid.

The top-level report uses `artifact_candidate`, never `default_promoted`.
It also records `semantic_default_eligible=false` and lists the outstanding
artifact-core, executor, provenance, platform, dictation, migration, and release
gates. A task-specific pass cannot be misread as approval to change production
routing.

## Error handling and safety

- Manifest parsing is strict and rejects unknown schema versions and fields.
- User-selected paths use Chatbook's path-validation boundary.
- Archive members and manifest filenames are untrusted data and cannot escape
  staging.
- Symlinks and irregular files are rejected.
- Digests are verified before prepared data or model files become runnable.
- External credentials are never stored in manifests, results, or logs.
- Logs prefer sample/model IDs and do not expose unnecessary local paths.
- A failure never converts into a different provider, artifact, precision, or
  sample population.

## Test strategy

Automated tests use tiny generated WAV fixtures and fake model executables; they
do not download datasets or load production models.

Tests cover:

- strict manifest validation and canonical fingerprints;
- size, digest, path-containment, symlink, staging, and atomic-publish failures;
- deterministic fixture generation;
- language normalization and edit counts;
- corpus WER/CER aggregation;
- paired-bootstrap determinism, pairing, and adverse-bound gates;
- insufficient populations and mismatched run fingerprints;
- silence and timestamp validation;
- one model load across a batch;
- child crash, timeout, malformed output, and incomplete runs;
- high-water and process-tree RSS normalization;
- throughput and 100-file reuse gates;
- deterministic JSONL and summary output; and
- explicit macOS-only evidence labeling.

The real qualification command is separate and explicit. TASK-593 is complete
only when the pinned macOS run produces full valid evidence for every declared
population and the versioned report is reviewed. Any incomplete population
keeps the task open and all promotion gates closed.

## ADR check

**ADR required:** yes

**ADR path:** `backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md`

**Reason:** ADR-025 already defines the artifacts, routing policy, qualification
thresholds, process boundary, VAD behavior, and promotion gates. This design
implements its evaluation slice without changing those decisions, so no new ADR
is needed.

## Sources

- [FLEURS dataset](https://huggingface.co/datasets/google/fleurs)
- [FLEURS paper](https://arxiv.org/abs/2205.12446)
- [Mozilla Common Voice terms](https://commonvoice.mozilla.org/en/terms)
- [`onnx-asr` v0.12.0 VAD implementation](https://github.com/istupakov/onnx-asr/blob/v0.12.0/src/onnx_asr/vad.py)
- [ADR-025](../../../backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md)
