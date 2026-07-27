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

The complete proposed v3 routing-candidate set is:

`bg`, `hr`, `cs`, `da`, `nl`, `et`, `fi`, `fr`, `de`, `el`, `hu`, `it`,
`lv`, `lt`, `mt`, `pl`, `pt`, `ro`, `sk`, `sl`, `es`, `sv`, `ru`, and `uk`.

The set is closed and versioned in the experiment manifest. Adding or removing
a language creates a new experiment identity and requires a new complete
comparison. The evaluated populations cover:

- clean and deterministically noisy English;
- silence and non-speech controls;
- short, beyond-direct-input-limit, ten-minute, and long-form stress media;
- every language in the complete v3 routing-candidate set, with deep slices for
  Spanish, French, German, Polish, Greek, Russian, and Ukrainian;
- stock Parakeet v2/v3 INT8 and F32 artifacts; and
- the local faster-whisper base/int8 comparison baseline.

The experiment manifest contains a closed model × population × metric/gate
matrix:

- Parakeet v2 INT8 and F32 plus faster-whisper base/int8 cover every required
  English population;
- Parakeet v3 INT8 and F32 plus faster-whisper base/int8 cover every required
  population for all 24 proposed non-English routing candidates; and
- required silence, timestamp, long-form, throughput, memory, and batch-reuse
  cells are enumerated for every applicable variant using separate quality,
  throughput, and memory/reuse measurement profiles.

Every matrix cell has predeclared minimum sample, reference-unit, and audio
duration requirements. The task cannot complete with a missing or incomplete
cell.

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
- evaluation tags such as clean, accent, noise, silence, long-form, or stress;
  and
- a resampling cluster ID that groups statistically dependent observations.

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

Every derivative of one source utterance shares a resampling cluster. Repeated
utterances from one known speaker use the speaker as the higher-level cluster.
When speaker identity is unavailable, the immutable source utterance is the
cluster. Cluster assignments are fixed in the corpus manifest before inference.

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
`batch_size=1`. The raw result records both requested and effective VAD batch
size. Missing VAD provenance, a VAD hash mismatch, an omitted effective value,
or a value other than one makes long-form qualification incomplete.

## Run identity

An experiment fingerprint is the SHA-256 of canonical JSON containing:

- corpus manifest and derived-fixture recipe revisions;
- normalizer and metric revisions;
- the closed comparison matrix and complete set of model and VAD artifact
  identities;
- runtime and operating-system metadata;
- CPU, memory, execution provider, and thread settings;
- gate configuration;
- bootstrap seed and iteration count; and
- harness source revision.

Every variant run also has a run fingerprint computed from the experiment
fingerprint plus the selected model variant, measurement profile, and its
effective execution settings. INT8, F32, and faster-whisper runs therefore
share one experiment fingerprint but retain different run fingerprints.

A report compares runs only when their experiment fingerprints match, their
variant identities fill the closed comparison matrix, and every non-variant
dimension is identical. Runs with different experiment fingerprints cannot be
combined.

## Execution flow

### Prepare

1. Validate corpus and model manifests before filesystem or network work.
2. Display a dry-run preflight containing required sources, licenses, bytes,
   local inputs, destination, and available space.
3. Require an explicit preparation command before any transfer or extraction.
4. Verify source archive/shard size and digest.
5. Reject unknown or duplicate archive members, symlinks, and irregular files,
   and enforce declared member-count, per-file-size, cumulative uncompressed
   byte, and staging-space limits before and during extraction.
6. Extract only declared contained regular files without following symlinks.
7. Normalize audio and build deterministic derived fixtures in staging.
8. Verify every prepared output before atomically publishing the prepared
   corpus.

Preparation failure leaves no usable partial corpus.

### Run

1. Validate the prepared corpus, selected model files, experiment fingerprint,
   and variant run fingerprint.
2. Spawn one isolated child process for one model variant and declared
   measurement profile.
3. Load that variant once and process the fixed ordered sample population.
4. Emit one raw result record per sample with the exact reference, hypothesis,
   timings, timestamps, warnings, model/runtime identity, resampling cluster,
   and requested/effective VAD settings.
5. Record the evidence required by the declared measurement profile. Only the
   `memory_reuse` profile enables child high-water and parent process-tree RSS
   sampling.
6. Finish with one terminal run record containing counts and completeness.
7. Publish raw JSONL atomically; a partial temporary file is not reportable.

A child crash, timeout, malformed record, or missing sample makes the complete
run ineligible for comparison. The harness does not silently retry with another
engine or precision.

### Report

1. Verify every experiment/run fingerprint and terminal run record.
2. Apply the predeclared versioned normalizer for each language.
3. Compute per-sample edit/reference counts.
4. Aggregate WER and CER populations separately.
5. Compute fixed-seed paired-bootstrap confidence intervals.
6. Validate timestamp, silence, throughput, memory, and reuse gates.
7. Emit scored per-sample records preserving raw and normalized references and
   hypotheses, normalizer identity and hash, reference-unit and edit counts,
   and gate-population membership.
8. Write deterministic machine-readable and human-readable reports atomically.

## Metrics and confidence intervals

For a population, corpus WER or CER is:

`sum(edit_count) / sum(reference_unit_count)`

It is not the arithmetic mean of per-sample percentages. WER and CER are
reported separately and never averaged together. Each language declares its
primary gate metric before a run; the other metric remains diagnostic.

Paired comparisons resample cluster identities with replacement and include all
member samples for each selected cluster. The same resampled clusters and
member samples are used for both model arms. A missing member in either arm
makes the comparison incomplete.

The reported delta is always `candidate error rate - baseline error rate`, so a
positive value means the candidate is worse. INT8 is the candidate against F32;
Parakeet INT8 is the candidate against faster-whisper.

The first version uses 10,000 replicates, a fixed manifest seed, and a two-sided
percentile 95% confidence interval. The adverse bound is the upper 97.5th
percentile of candidate-minus-baseline deltas. Both the point estimate and that
upper bound must be less than or equal to the applicable ADR-025 threshold.

For a v3 macro-family replicate, the harness cluster-resamples within each
language, computes each language's corpus-rate delta, then takes the unweighted
mean across languages sharing the same primary metric. The observed point
estimate and upper 97.5th percentile of those replicate macro deltas are gated.
WER-primary and CER-primary languages never enter the same macro family.

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

### Performance protocol

Every INT8, F32, and faster-whisper variant uses three declared measurement
profiles under the same experiment fingerprint:

- `quality` processes the complete corpus without a high-frequency resource
  sampler;
- `throughput` performs one untimed warm-up followed by three timed passes over
  the fixed performance population without process-tree RSS polling; and
- `memory_reuse` records high-water and process-tree RSS while exercising the
  fixed memory population and 100-file reuse sequence. Timing from this profile
  is diagnostic only and never feeds the throughput gate.

The task-specific throughput and 3 GiB gates apply to Parakeet INT8; F32 and
faster-whisper measurements remain required comparison evidence.

For each throughput pass, timed recognition begins when the already-decoded,
normalized 16 kHz mono waveform is submitted and ends when the complete
recognition result has been materialized. Model load, corpus preparation,
download, audio decoding, and resource-sampler overhead are excluded. Runtime
preprocessing, VAD segmentation, and ASR inference are included. Long-form
timing therefore includes the pinned VAD path.

Warm inverse real-time factor for each pass is:

`sum(audio_duration_seconds) / sum(recognition_wall_seconds)`

over the fixed performance population. All three pass values are preserved.
The predeclared aggregate is their median, and the Parakeet INT8 gate requires
that median to be greater than one.

## Memory measurement

In the `memory_reuse` profile, the child records the operating system's process
high-water RSS using an OS-specific normalized probe. The parent samples RSS
for the child process tree through `psutil` during model load and inference.

The parent samples the complete process tree at a fixed 10 ms monotonic
interval, rediscovering descendants at every sample. Both peak values are
preserved. The gate uses the larger of child high-water RSS and maximum sampled
process-tree RSS. If either required measurement is unavailable, the memory
gate is incomplete rather than passing on the remaining value.

For the 100-file reuse gate, one warm-up file is followed by a one-second idle
window sampled at the same interval. The median total process-tree RSS in that
window is the baseline. After the fixed 100-file run, another one-second idle
window supplies the post-run median. The same worker PID and a model-load count
of exactly one are required, and:

`(post_run_median - baseline_median) / baseline_median <= 0.15`

The peak gate still uses the maximum observed value across load, warm-up, the
100 files, and both idle windows.

On the initial macOS reference run, Darwin high-water units and conversion are
captured in the report. Future Windows/Linux probes require native validation
before their results can be compared or used as platform evidence.

## Decision model

Every artifact, matrix cell, and language receives one of:

- `pass`: all task-specific evidence and thresholds passed;
- `fail`: complete valid evidence exceeded at least one threshold; or
- `incomplete`: required evidence, provenance, pairing, or capability was
  absent or invalid.

Each non-English language first receives
`routing_candidate: pass|fail|incomplete` from its own complete v3 INT8/F32 and
faster-whisper comparisons. A language-level INT8 failure excludes only that
language and never selects F32 as a substitute. Incomplete language evidence
also excludes that language, but because every proposed language is a required
matrix cell, it keeps TASK-593 incomplete rather than producing a final
qualification result.

The provisional candidate-language set contains only languages whose individual
quality gates pass. The v3 macro-family point estimate and confidence interval
are then calculated over that explicitly recorded provisional set. A failed
macro-family gate is an artifact-global failure; it does not trigger repeated
language removal until the macro happens to pass.

The v3 artifact decision is evaluated in this order:

1. Any incomplete required cell produces `artifact_candidate=incomplete`.
2. If all languages fail conclusively, the empty provisional set produces
   `artifact_candidate=fail`; macro-family gates are `not_applicable`, not
   incomplete.
3. Complete per-language failures are recorded as excluded languages without
   failing the artifact for the remaining languages.
4. Global model/runtime, silence, long-form, memory, throughput, batch-reuse,
   or surviving-set macro failure produces `artifact_candidate=fail` and blocks
   every v3 language.
5. When all global gates pass and at least one language survives,
   `artifact_candidate=pass` contains the final candidate-language set plus the
   explicitly excluded failed languages.

Only non-empty primary-metric families among the surviving languages are
bootstrapped and gated. If one family is empty while another has survivors, the
empty family is recorded as `not_applicable`; it is neither bootstrapped nor
treated as incomplete.

The v2 artifact has no per-language subset: incomplete English/global evidence
is incomplete, and any complete English or global failure produces
`artifact_candidate=fail`.

An artifact-global INT8 failure blocks all promotion for that artifact and
never selects F32 as a substitute. F32 remains comparison evidence only.

The top-level report never uses `default_promoted`. It records
`semantic_default_eligible=false` and lists the outstanding artifact-core,
executor, provenance, platform, dictation, migration, and release gates. A
task-specific pass cannot be misread as approval to change production routing.
This task may complete with fully valid `fail` evidence because qualification
has then produced a conclusive result; any `incomplete` required cell keeps the
task open.

## Error handling and safety

- Manifest parsing is strict and rejects unknown schema versions and fields.
- User-selected paths use Chatbook's path-validation boundary.
- Archive members and manifest filenames are untrusted data and cannot escape
  staging.
- Symlinks and irregular files are rejected.
- The declared archive-member set, member count, per-member size, cumulative
  uncompressed bytes, and available staging space are bounded and enforced
  while streaming extraction, not only during preflight.
- Digests are verified before prepared data or model files become runnable.
- External credentials are never stored in manifests, results, or logs.
- Logs prefer sample/model IDs and do not expose unnecessary local paths.
- A failure never converts into a different provider, artifact, precision, or
  sample population.

## Test strategy

Automated tests use tiny generated WAV fixtures and fake model executables; they
do not download datasets or load production models.

Tests cover:

- strict manifest validation, experiment fingerprints, and variant run
  fingerprints;
- size, digest, path-containment, unknown/duplicate member, symlink, irregular
  file, member-count, per-file, cumulative-uncompressed-byte, staging-space,
  and atomic-publish failures;
- deterministic fixture generation;
- golden normalization fixtures for every evaluated language covering case,
  punctuation, whitespace, Unicode normalization, unitization, and edit counts;
- corpus WER/CER aggregation;
- paired clustered-bootstrap determinism, derivative/speaker clustering, model
  pairing, delta direction, macro-family intervals, and adverse-bound gates;
- the complete 24-language comparison matrix, insufficient populations,
  missing cells, and mismatched experiment/run fingerprints;
- a conclusive zero-survivor v3 result and an empty primary-metric family when
  another family still has survivors;
- silence and timestamp validation;
- requested/effective VAD `batch_size=1`, including missing, ignored, and
  mismatched effective settings;
- one model load across a batch;
- child crash, timeout, malformed output, and incomplete runs;
- high-water and process-tree RSS normalization;
- isolated quality, three-pass throughput, and memory/reuse profiles, including
  the predeclared throughput median and 100-file reuse gates;
- scored-sample audit fields for raw and normalized text, normalizer identity
  and hash, units, edit counts, and gate-population membership;
- deterministic JSONL and summary output; and
- explicit macOS-only evidence labeling.

The real qualification command is separate and explicit. TASK-593 is complete
only when the pinned macOS run produces full valid pass-or-fail evidence for
every cell in the closed matrix and the versioned report is reviewed. Any
incomplete cell keeps the task open and all promotion gates closed.

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
