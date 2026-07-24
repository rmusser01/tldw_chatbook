# Cross-Platform STT Runtimes and Shared Model Artifacts — Design

**Status:** approved by the user on 2026-07-23; review amendments approved on 2026-07-23

**Date:** 2026-07-23

**Related tasks:** TASK-404 through TASK-417; see the
[delivery map](../plans/2026-07-23-stt-artifact-runtime-delivery-map.md)

**Canonical ADR:** [ADR-025](../../../backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md)

**Primary upstreams reviewed:** `onnx-asr` 0.12.0 and `transcribe.cpp` 0.1.3

## Executive decision

Chatbook will make Parakeet TDT ONNX, through `onnx-asr`, the stable default
speech-to-text runtime for explicitly selected supported languages:

- New and unset language selections default to `en`.
- `provider=default` plus explicit `en` selects Parakeet TDT 0.6B v2 ONNX.
- `provider=default` plus an explicitly selected, Chatbook-validated
  non-English Parakeet v3 language selects Parakeet TDT 0.6B v3 ONNX.
- `provider=default` plus `auto`, an unsupported language, or a translation
  request selects faster-whisper.
- INT8 is the default Parakeet variant. F32 is an explicit, separately
  downloaded option.

`transcribe.cpp` is adopted as an optional, curated GGUF breadth engine. It is
not the default, the universal provider abstraction, or an automatic fallback.
The initial catalog contains one representative model from four families:
Whisper small, Canary 180M Flash, Moonshine tiny, and Qwen3-ASR 0.6B.

Failures never silently cross engine boundaries. A failed eligible request
offers an explicit **Retry with faster-whisper** action. A single
accelerator-to-CPU retry is allowed within the same provider and model during
runtime initialization.

The existing GGUF browser/downloader becomes a frontend over a new
format-neutral model artifact service. The service supports verified,
revision-pinned managed downloads and managed local import for both single-file
GGUF artifacts and multi-file ONNX bundles, including explicitly declared
dependent artifacts such as the VAD model required for long-form Parakeet
recognition. STT is the first consumer; moving LLM model management onto the
service is a separate future task.

Batch media ingestion is the first delivery surface. Audio/video work moves
from the shared parse pool to a physically separate one-process heavy-media
lane with one resident STT model. Legacy `parakeet` and `parakeet-mlx`
implementations are removed in the landing release only after the specified
batch, buffer, migration, platform, quality, performance, and recovery gates
pass.

## Why this design

Chatbook already has broad STT functionality, but the current implementation
combines provider discovery, model loading, routing, inference, result shaping,
and caches in one large `TranscriptionService`. Provider identifiers and
privacy assumptions are also repeated in dictation and UI code.

The current Library ingestion dispatcher limits audio/video concurrency to one,
but those jobs still run in a shared multi-process parse pool. Heavy tasks can
land in different worker processes over time, allowing the same multi-gigabyte
model to be loaded into several processes. A native crash can also invalidate
unrelated parse work in that pool generation.

The existing Hugging Face GGUF downloader resolves mutable repository state and
writes model files directly without the full revision pinning, expected-digest
verification, resumable staging, atomic activation, bundle handling, and
interprocess leases required for native model artifacts.

The selected architecture addresses these issues without making a young
pre-1.0 native runtime responsible for Chatbook's stable default path:

- `onnx-asr` supplies the cross-platform Parakeet v2/v3 path.
- faster-whisper retains broad language detection and translation.
- `transcribe.cpp` adds carefully bounded model-family breadth.
- A coordinator and provider contract stop the current service monolith from
  accumulating another set of branches.
- A shared artifact service gives STT trustworthy lifecycle semantics and
  creates a reusable foundation for later local-model consumers.

## Goals

- Make English Parakeet v2 ONNX INT8 the normal batch-ingestion path.
- Support Parakeet v3 ONNX for explicit, validated non-English languages.
- Keep `auto`, unsupported languages, and translation dependable through
  faster-whisper.
- Support every platform for which the selected Python runtimes publish the
  required wheels.
- Add a small, representative transcribe.cpp model catalog without mirroring
  the whole upstream catalog.
- Make model acquisition explicit, revision-pinned, integrity-checked,
  resumable, atomic, and safe across processes.
- Preserve current file and buffer transcription entry points while delivering
  batch media ingestion first.
- Bound native model residency and native crash impact.
- Remove the overlapping MLX and NeMo Parakeet implementations once the ONNX
  replacement has passed release gates.
- Establish reproducible quality, performance, memory, migration, and recovery
  evidence before changing the default.
- Persist enough versioned transcription provenance to explain the provider,
  artifact, language routing, device resolution, and explicit retry lineage of
  every newly stored transcript.

## Non-goals

- Making transcribe.cpp the universal STT runtime.
- Mirroring every model or quantization offered upstream.
- Automatically downloading a default model without user consent.
- Silently rerunning a failed request through a different engine.
- Adding a separate language-identification prepass.
- Treating Parakeet v3's internal language selection as proof that the input
  was one of its supported languages.
- Shipping true live-streaming semantics in this batch-first milestone.
- Rewriting unrelated legacy STT providers in the same change.
- Migrating LLM artifacts to the new store in the STT milestone.
- Content-addressed deduplication between artifacts in the first version.
- Building, compiling, or converting transcribe.cpp models in Chatbook.
- Automatically deleting external Hugging Face, NeMo, or MLX caches.

## Current-state anchors

The implementation plan should verify line numbers again, but these are the
current architectural anchors:

- `tldw_chatbook/Local_Ingestion/transcription_service.py` owns provider
  imports, routing, caches, file/buffer APIs, and result normalization.
- `tldw_chatbook/Audio/dictation_service.py` calls `transcribe_buffer()` on
  repeated short microphone buffers.
- `tldw_chatbook/Audio/dictation_service_lazy.py` contains local-provider and
  privacy lists that have drifted from the service.
- `tldw_chatbook/app.py` owns a spawn-context shared parse pool plus a
  dispatch-only audio/video concurrency cap.
- `tldw_chatbook/Local_Ingestion/audio_processing.py` creates transcription
  service instances during media parsing.
- `tldw_chatbook/Widgets/HuggingFace/download_manager.py` and
  `tldw_chatbook/LLM_Calls/huggingface_api.py` implement the existing browser
  and direct download behavior.
- `pyproject.toml` keeps STT dependencies in optional extras; the audio, video,
  and media-processing extras currently use faster-whisper as their default
  transcription dependency.
- `tldw_chatbook/Utils/optional_deps.py` and
  `tldw_chatbook/Library/ingest_capabilities.py` expose optional-dependency
  availability and contain legacy provider identities.

## Architecture

### Compatibility facade

`TranscriptionService` remains the public compatibility facade during the
migration. Existing file and buffer callers keep their entry points, but the
facade contains no provider-specific inference branches. It validates the
public call shape, constructs a canonical request, and delegates to a
`TranscriptionCoordinator`.

Deprecated provider spellings are handled only by the versioned configuration
migration and a small explicit compatibility map where required. The provider
registry never uses prefix or wildcard matching.

### Transcription coordinator

`TranscriptionCoordinator` owns:

- Default resolution and provider/model routing.
- Request validation and capability composition.
- Artifact preflight and operation leases.
- Audio preparation.
- Heavy-lane submission.
- Progress, cancellation, and force-stop coordination.
- Provider result normalization.
- Structured failure construction.
- Explicit retry action eligibility.

The coordinator does not download models implicitly, persist media records, or
contain inference implementation details.

### Provider contract

Native providers implement a narrow contract equivalent to:

- `describe()` returns declared metadata without importing or loading the
  native runtime.
- `probe()` lazily checks package and runtime availability.
- `load(artifact, options)` initializes one selected model.
- `transcribe(request, prepared_audio, progress, cancellation)` performs
  inference.
- `close()` releases provider-owned resources best-effort.

The exact Python names may change during planning, but the separation does not.

Provider metadata has two levels:

1. **Declared catalog capabilities** are cheap, deterministic metadata used by
   routing and UI before a model is loaded.
2. **Runtime-observed capabilities** are reported after loading and must be
   compatible with the declaration. A mismatch fails closed and marks the
   artifact/runtime combination incompatible.

Capability metadata distinguishes at least:

- Explicit language support.
- Automatic language detection.
- Transcription versus translation.
- File and buffer input.
- Batch and true-streaming behavior.
- Transcript-only, segment, and word/token timestamps.
- VAD/long-form behavior.
- Cancellation granularity.
- Punctuation and capitalization.
- Execution providers and precision variants.
- Language-input mode: enforced explicit hint, routing-only caller assertion,
  automatic detection, or automatic only.
- Whether a requested language is enforced by the decoder or used only to
  select a compatible model.

### Initial providers

#### `parakeet-onnx`

The stable default provider uses `onnx-asr` with managed Parakeet TDT 0.6B v2
and v3 ONNX bundles.

- v2 is the English default.
- v3 is used for explicit validated non-English languages.
- The pinned v3 TDT adapter does not accept or enforce a language hint. Its
  catalog entry therefore declares `language_routing=asserted_by_caller` and
  `language_constraint=false`: the requested language selects v3, after which
  v3 performs its own internal language selection.
- INT8 is the default artifact.
- F32 is optional and never downloaded as a side effect of selecting INT8.
- Long input uses the runtime's VAD/long-form path.
- The Parakeet artifact requirement includes a separately pinned managed VAD
  dependency. Provider initialization uses managed local directories and
  remains offline; it never lets `onnx-asr` acquire a missing model implicitly.
- Buffer input remains available for dictation compatibility.
- Long-form cancellation is checked before every VAD segment batch. The
  initial adapter sets the upstream VAD ASR batch size to one so cancellation
  is never delayed by a queued multi-segment inference batch; later batching
  changes require new cancellation-latency and throughput evidence.
- Translation is unsupported.
- The provider does not fabricate a detected language field.

#### `faster-whisper`

faster-whisper remains the broad-language and explicit-retry engine:

- `auto` routing.
- Languages outside Chatbook's validated Parakeet v3 set.
- Translation.
- Explicit retry after another engine fails.

Existing saved model choice continues to apply when it is compatible. A
missing faster-whisper model or package is surfaced during preflight; selecting
the retry action does not silently install it.

#### `transcribe-cpp`

transcribe.cpp is an optional provider using the official Python binding and
managed GGUF artifacts. The binding is imported lazily and pinned to an exact
pre-1.0 release. The provider:

- Accepts only cataloged or locally imported compatible GGUF artifacts.
- Converts the request-scoped audio representation to 16 kHz mono input when
  required.
- Reports per-model capabilities instead of claiming that every upstream
  family has the same feature set.
- Uses active cancellation when exposed by the binding.
- Allows at most one active inference in its worker.

The curated catalog is described below. Arbitrary compatible GGUF imports are
advanced, uncurated entries and never become automatic routing targets.

#### Temporary legacy bridge

Unrelated retained providers may remain behind a temporary bridge so the
coordinator can land without rewriting every provider. The bridge does not
include NeMo `parakeet` or `parakeet-mlx` after the ONNX removal gates pass.

## Routing and language semantics

### Default routing

The following table applies when the caller selects semantic
`provider=default`:

| Request | Selected path |
| --- | --- |
| Language omitted or unset | Resolve to `en`, then Parakeet v2 |
| Explicit `en` | Parakeet v2 |
| Explicit validated Parakeet v3 non-English language | Select Parakeet v3; the decoder does not force the requested language |
| `auto` | faster-whisper |
| Explicit language outside the validated v3 set | faster-whisper |
| Translation request | faster-whisper |

The upstream Parakeet v3 language list currently contains Bulgarian, Croatian,
Czech, Danish, Dutch, English, Estonian, Finnish, French, German, Greek,
Hungarian, Italian, Latvian, Lithuanian, Maltese, Polish, Portuguese, Romanian,
Slovak, Slovenian, Spanish, Swedish, Russian, and Ukrainian. Chatbook's routed
set is the intersection of that upstream list and languages passing Chatbook's
release evaluation. English still routes to v2.

An exact user-selected provider/model overrides semantic default routing but
must pass capability checks:

- Explicit Parakeet plus an unsupported language fails with
  `UnsupportedLanguage`.
- Explicit Parakeet plus translation fails with `UnsupportedCapability`.
- An explicitly selected compatible transcribe.cpp Whisper or Canary model may
  handle translation if its catalog entry declares that feature.
- The curated transcribe.cpp Qwen3-ASR port accepts automatic language
  selection only. Under an explicit language such as the default `en`, the
  model is unavailable until the user explicitly changes the request to
  `auto`; Chatbook never silently drops the language constraint.

### Requested, effective, and detected language

Normalized results keep separate fields:

- `requested_language`: the value the caller supplied, including `auto`.
- `effective_language`: the language constraint actually enforced by the
  provider, or the provider's resolved value when trustworthy.
- `detected_language`: nullable provider-reported detection.

For v2, an explicit or defaulted English request records
`requested_language=en` and `effective_language=en`. For v3, an explicit
request such as Spanish records `requested_language=es`,
`effective_language=auto`, and `detected_language=null`, plus the stable
`requested_language_not_enforced` warning. The request selected a model
validated for Spanish, but the pinned decoder did not receive or enforce a
Spanish constraint. UI copy says **Use Parakeet v3 for Spanish**, never
**Force Spanish**.

`auto` requests may populate detected language from faster-whisper. If a future
`onnx-asr` release exposes a reliable detected-language identity or accepts an
enforced hint, adopting it requires a reviewed catalog and contract update; it
does not silently change routing or provenance semantics.

### Visible resolution

Routing is resolved before work is enqueued and is visible in preflight and job
details. Therefore selecting faster-whisper for `auto`, unsupported languages,
or translation is policy resolution, not a hidden fallback.

## Request and result contracts

A canonical transcription request carries:

- Source reference: file or bounded in-memory buffer.
- Requested provider and opaque model/artifact ID.
- Requested language and task.
- Precision and execution-provider preference.
- Timestamp and diarization requirements.
- VAD/long-form options.
- Cancellation token and progress sink.
- Batch/job identity and privacy flags.

A normalized result carries:

- Text and normalized segments.
- Requested, effective, and detected language fields.
- Provider, model, root artifact revision, exact loaded dependency revisions,
  precision, and effective device.
- Timestamp granularity actually produced.
- Duration and provider-neutral timing metrics.
- Warnings that do not invalidate the transcript.
- Provenance required for persisted transcript metadata.

Diarization, VAD, and timestamp presentation are pipeline capabilities. The
coordinator evaluates the composed pipeline rather than requiring the inference
adapter itself to implement every feature.

### Persisted transcription provenance

The normalized provenance is part of the media persistence contract, not only
an in-memory result. The media database receives a versioned
`transcription_provenance` document containing at least:

- Schema version and transcription-attempt identity.
- Provider, model, immutable root artifact revision, exact loaded artifact
  dependencies, precision, and effective execution provider.
- Requested, effective, and detected language.
- Requested task and capabilities actually produced.
- Retry relationship and a bounded failed-attempt snapshot when the transcript
  came from an explicit retry.

The database migration adds nullable
`Media.transcription_provenance_json TEXT` while continuing to populate
`Media.transcription_model` as a compatibility summary. The JSON document is
validated at the persistence boundary and written in the same transaction as
the transcript content. Chatbook export/import, sync, search projections, and
API schemas preserve the versioned document without requiring old records to
synthesize missing provenance. The legacy `Transcripts.whisper_model` field is
not repurposed to hold structured provenance.

Failed attempts remain job history rather than transcript records. A successful
retry stores a generic `retry_of_attempt_id`, its effective faster-whisper
provenance, and `retry_of_job_id` when the caller has a durable Library job. It
never overwrites the failed attempt's job history.

Because bounded job-history retention may later prune the failed job,
`retry_of_job_id` is a best-effort navigation link rather than the canonical
provenance record. The successful transcript embeds a versioned, bounded
`failed_attempt` snapshot containing nullable attempt/job identity, provider,
model, root and dependency artifact revisions, precision,
requested/effective device, requested/effective language, task, and stable
error code. It excludes raw exception text, local paths, audio, and unbounded
logs. The Library ingest-job contract still stores structured failure
provenance, but transcript lineage remains interpretable after job pruning and
for non-Library callers.

## Audio preparation

Batch requests use a request-scoped `PreparedAudio` abstraction rather than
unconditionally writing a normalized WAV.

`PreparedAudio` retains:

- Source or already-extracted audio path.
- Sample rate, channels, duration, and encoding metadata when known.
- A job-owned temporary directory.
- Lazily derived representations requested by providers.

The first request for a representation may decode or resample it; subsequent
consumers in the same job reuse it. This avoids redundant decoding while still
meeting transcribe.cpp's 16 kHz mono input requirement.

Short dictation buffers remain in memory and are passed through bounded IPC.
They are not written to disk merely to share the heavy lane. Buffer sizes are
validated before copying across the process boundary.

All job-owned temporary assets are cleaned after success, failure,
cancellation, or worker recovery. An app-start cleanup removes only abandoned
temporary directories carrying Chatbook's own marker and never traverses an
unresolved broad path.

## Heavy-media execution lane

### Physical isolation

The existing audio/video dispatch cap is not sufficient because heavy jobs
still share the general parse pool. The new design creates a physically
separate spawn-context pool with exactly one heavy worker.

- Documents and other light media remain in the current general parse pool.
- Audio/video ingestion runs in the heavy pool.
- The heavy worker never writes to the media database; completed payloads
  return to the existing single-writer stage.
- A heavy-worker crash cannot invalidate light parse workers.

### One-model residency

The heavy worker retains at most one STT model identity:

`(provider, model, root artifact revision, dependency-closure fingerprint, precision, execution provider)`

Adjacent requests with the same identity reuse the loaded model. Before a
different identity is dispatched, the controller terminates and recreates the
heavy worker rather than trusting native allocators to release all memory
in-process. The worker is also recycled after a native crash, force stop, and a
configurable bounded number of completed jobs.

The heavy worker, not the parent coordinator, owns the operation leases for the
resident artifact and every loaded artifact in its dependency closure, including
the VAD model. It acquires the lease set in stable artifact-ID order before
loading the model and holds it across all same-identity requests for the full
resident-model lifetime. The leases are released only when `close()` completes
or the worker exits. Therefore an idle but resident model still blocks deletion
of the root or a loaded dependency. Process exit releases the OS-backed leases
after a crash or forced recycle; the parent never retains duplicate leases that
could leak after worker death.

This policy avoids per-file model loads and prevents a shared N-worker pool from
eventually holding N copies of a multi-gigabyte model.

### Dictation compatibility and priority

An app-owned `LocalSTTExecutor` controller owns the heavy process and is shared
by Library ingestion and dictation. Neither parse workers nor individual
`TranscriptionService` instances create their own heavy process.

The same executor accepts bounded buffer transcription requests so removing the
legacy Parakeet providers does not break the public buffer path. This milestone
does not claim true streaming. The compatibility
`create_streaming_transcriber()` entry point reports streaming unsupported for
Parakeet ONNX by returning `None`, allowing existing callers to use the
retained buffer fallback.

- At most one dictation inference is pending.
- New microphone audio is coalesced into the pending request rather than queued
  without bound.
- Coalesced audio has a configured maximum duration and byte size. When the
  bound would be exceeded, capture pauses and the UI reports a dictation
  overrun; audio is never silently dropped.
- Dictation receives priority when selecting the next job.
- Active batch inference is not preempted.
- The UI reports **Local transcription busy** and allows the user to pause
  dispatch of the next batch item.

The live-dictation experience remains an explicit parity gate because
batch-shaped ONNX buffer inference may not meet the latency of the removed MLX
implementation on every machine.

### Worker health

The parent tracks the last phase reported by the heavy worker. A worker exit
during model load or inference becomes `EngineCrashed` for the active request.
A worker exit during audio preparation is not blamed on the model.

The unhealthy circuit key is:

`(provider, model, root artifact revision, dependency-closure fingerprint, precision, device)`

One relevant native crash pauses matching queued work for the session to avoid
a crash loop. Explicit retry clears the circuit once. A second crash pauses it
again. Unrelated jobs and provider/model combinations continue.

## Progress, cancellation, and force stop

Stable progress phases are:

`queued → preparing → loading → transcribing → post-processing → saving → complete`

Providers may add real percentage or segment progress. Where they cannot, the
phase stays indeterminate; Chatbook never fabricates a percentage.

Cancellation behavior is phase-specific:

- Queued work is removed immediately.
- A download stops safely; incomplete data stays only in staging and may be
  resumed, but is not called verified or installed.
- Audio preparation terminates its decoder and cleans derived files.
- Cancellation requested during an uninterruptible native model load is
  recorded and honored when the load returns.
- Parakeet ONNX checks between VAD/chunk boundaries.
- transcribe.cpp uses active cancellation when available.
- A short atomic media write completes once begun, so cancellation never leaves
  a partial transcript record.

No fixed inference timeout is imposed based only on media duration. If
cooperative cancellation does not return after a grace period, the UI offers
**Force stop**. Force stop terminates only the heavy pool, marks the active
request cancelled, discards the resident model, and recreates the worker before
the next job.

Every heavy request, progress event, result, and error envelope carries the
attempt/job identity and executor generation. Force stop first detaches the
current generation from the writer path, then terminates and asynchronously
joins that generation before creating the replacement. Callbacks from a
detached generation are discarded, so an old success cannot reach the
single-writer stage after cancellation. Each active attempt reaches exactly one
terminal state.

Audio preparation subprocesses such as FFmpeg are started under a
platform-appropriate child-process group/tree owned by the heavy worker.
Cooperative cancellation terminates that tree before cleaning its temporary
directory. Forced worker teardown also terminates descendants—using a
process-group/session boundary on POSIX and the equivalent job/process-group
ownership on Windows—so killing the Python worker cannot leave a decoder
running against deleted temporary files.

## Shared model artifact core

### Ownership

`ModelArtifactService` is the sole writer for Chatbook-managed model artifacts.
The existing Hugging Face/GGUF browser becomes a frontend over this service.
Providers receive installed artifact handles; they do not download or delete
model files.

The core is format-neutral. An STT catalog layer adds speech-specific
capability and routing metadata without teaching the storage service about
languages or transcription tasks.

ADR-020's cloud model-ID cache remains separate. Cloud discovery tracks remote
identifiers and timestamps; it does not own local binary artifacts, hashes,
leases, or installation state.

### Artifact descriptor

Each catalog descriptor includes:

- Stable artifact, model, and variant IDs.
- Consumer and format: for example `stt`, `onnx_bundle`, or `gguf`.
- Model family and upstream source.
- Immutable repository revision.
- Required file list, byte size, and SHA-256 for every file.
- Precision and expected installed size.
- Compatible runtime name and version range.
- License, source URL, and usage notices.
- Declared platform constraints, if any.
- Provenance class.
- Required artifact dependencies, including their exact revision and variant.

STT catalog entries add:

- Languages and task support.
- Timestamp granularity.
- VAD, batch, streaming, and cancellation characteristics.
- Default/request-routing eligibility.
- Chatbook validation status and benchmark revision.

### Provenance labels

The UI uses precise labels:

- **Chatbook curated:** a reviewed descriptor, immutable revision, and expected
  digests ship in Chatbook's catalog.
- **Integrity verified:** an on-demand remote entry is pinned to an immutable
  revision and matches a digest independently supplied by the repository, but
  Chatbook has not curated its behavior or trustworthiness.
- **Local integrity recorded:** a validated local import is copied into the
  managed store and hashed, without an upstream provenance claim. A remote file
  for which Chatbook can only calculate the digest after downloading also uses
  this label rather than implying independent source verification.

Digest verification is not presented as malware safety. Model files are
untrusted native-runtime input. The curated catalog is the only source of
automatic routing candidates.

### Versioned installed layout

Installed artifact versions are immutable directories beneath the configured
model root. The logical layout is equivalent to:

`artifacts/<artifact-id>/<revision>/<variant>/`

An atomic, small active-version record selects the installed revision. An
update writes a new version directory and then atomically replaces the active
record; it never replaces a populated directory in place. The previous version
remains available until no operation lease references it and the user or
retention policy removes it.

For an artifact with dependencies, the root active/readiness record names the
exact dependency revisions and is written last, only after every immutable
dependency version is installed and verified. This record is the atomic
loadability boundary; the design does not claim a multi-directory filesystem
transaction. Crash recovery recomputes root readiness from manifests and never
exposes a partially activated dependency closure.

The canonical ordered set of root/dependency artifact IDs, revisions, and
variants produces a dependency-closure fingerprint. That fingerprint is part
of resident-model identity; changing a VAD or other loaded dependency therefore
recycles the worker. Result provenance records the individual dependency
identities rather than only the fingerprint.

The first version does not deduplicate identical file content across artifacts.

### Managed download

Download flow:

1. Resolve the catalog descriptor and immutable source revision.
2. Resolve the complete dependency closure, including a pinned VAD artifact
   where required.
3. Present model, precision, source, license, total size, destination, and
   required free space.
4. Obtain explicit user confirmation for the complete dependency set.
5. Acquire each artifact's interprocess installation lock in stable artifact-ID
   order.
6. Download into per-artifact staging directories with resume support.
7. Verify every file's final byte size and SHA-256.
8. Write and fsync each installed manifest.
9. Promote each verified immutable version and its own active-version record.
10. Atomically write the requested root artifact's readiness record last, only
    after every exact dependency revision is installed and verified.
    Dependencies shared by other installed artifacts remain independently
    visible and usable.

Failure or cancellation leaves the currently active artifact untouched.
Incomplete staging is never loadable. A later cleanup removes only staging
owned by the artifact service.

Gated or authenticated repositories fail with instructions to complete
upstream terms or configure credentials. Tokens come from supported
environment/config/keyring paths and are never written into manifests or logs.

### Local import

Local import copies content into staging; managed artifacts never depend on an
external path remaining present.

- GGUF import validates regular-file status, safe resolution, magic/version,
  bounded readable metadata, and declared model compatibility before copying.
  Validation does not load the model through the inference runtime in the UI
  process.
- The first release accepts local ONNX only as an offline import of an existing
  Chatbook catalog descriptor. The selected bundle manifest identifies that
  descriptor and enumerates every model and external-data file, but the catalog
  remains authoritative for the required file set, byte sizes, and digests.
  Every imported file must match it.
- Unknown or modified ONNX graphs fail with `UnsupportedArtifact`; arbitrary
  graph discovery is out of scope for this milestone.
- Symlinks and irregular files are rejected.
- ONNX external-data references must be relative, contained within the bundle,
  declared, and present. Absolute paths and `..` traversal are rejected.
- Input metadata is checked again after copying to reduce time-of-check versus
  time-of-use risk.
- Every copied file is hashed and the resulting provenance is local.

Future arbitrary ONNX graph import requires a separately pinned optional
`onnx` parser and a short-lived validation process that compares actual
external-data references with the proposed manifest under file-count, byte,
recursion, memory, and time limits. It must not parse an untrusted graph in the
UI process or resident inference worker.

The import screen shows the extra disk space required for the managed copy.

### Cross-process locks and operation leases

Installation, activation, loading, deletion, and worker recovery cross process
boundaries. Locks and leases therefore use a cross-platform interprocess
primitive whose ownership is released by process exit.

- Installation and activation take the artifact's mutation lock.
- The heavy worker acquires operation leases for the exact root artifact and
  every loaded dependency before model load, in stable artifact-ID order, and
  holds them for the full resident-model lifetime, including idle
  same-identity reuse.
- Deletion requires exclusive mutation ownership and no active operation
  leases.
- A worker crash releases its OS-backed leases.
- The service never relies solely on an in-memory counter or a PID file.

The implementation plan must select and test the concrete primitive on every
supported OS; the architectural requirement is automatic crash release and
correct shared/exclusive behavior.

This selection is a prerequisite technical spike, not an implementation detail
to discover after the artifact service is built. On Windows, macOS, and Linux,
the spike must prove that process A can hold shared load leases for a root and
dependency across multiple same-model requests and an idle-residency interval,
process B cannot delete either version, killing process A releases both leases
without a parent-held leak, and process B can then obtain exclusive deletion
ownership. The artifact-service task does not start until one primitive passes
this proof.

### Deletion and disk pressure

Deletion is refused while an artifact version is leased. The UI reports whether
the owner is an idle resident STT model or an active job without exposing
unnecessary local paths. An idle owner can be unloaded by recycling the heavy
worker; an active owner follows the normal cancellation/force-stop flow.
Installed and staging space are reported separately.

Preflight accounts for:

- Download staging.
- Existing version retained during upgrade.
- Managed local-copy space.
- Derived audio temporary space.
- A safety margin for filesystem metadata and runtime output.

## Model acquisition UX

The renovated model browser has three explicit views:

- **Curated:** reviewed Chatbook catalog entries with recommended variants.
- **Remote:** on-demand Hugging Face browsing with integrity and trust labels.
- **Installed / Import local:** installed inventory, local import, version,
  disk usage, and safe deletion.

Selecting a transcription request never starts a download. If preflight returns
`ModelNotInstalled`, interactive callers may offer:

- **Download and transcribe**.
- Choose an installed model.
- Choose faster-whisper where the routing/fallback policy allows it.

The download decision happens before a batch is enqueued, so worker processes
never prompt the user. Multiple files sharing one missing artifact produce one
preflight and one installation.

Headless callers receive a structured error containing the required artifact
ID and no interactive side effect.

## Error model and fallback policy

Every user-visible failure contains:

- Stable error code.
- Phase.
- Provider, model, artifact revision, and precision.
- Requested and effective device when relevant.
- A concise, sanitized explanation.
- Same-configuration retryability.
- Explicit eligible actions.

Initial codes include:

- `ModelNotInstalled`
- `ArtifactCorrupt`
- `ArtifactIncompatible`
- `ProviderUnavailable`
- `ProviderRemoved`
- `UnsupportedLanguage`
- `UnsupportedCapability`
- `InsufficientDiskSpace`
- `InsufficientMemory`
- `InferenceFailed`
- `EngineCrashed`
- `Cancelled`

Cross-engine fallback is never automatic. **Retry with faster-whisper** creates
a new visible request with its own routing and artifact preflight. It neither
mutates the failed request nor claims that the original engine succeeded.

A single automatic accelerator-to-CPU retry is permitted only when the same
provider/model fails during execution-provider initialization. Progress
reports the retry, and the final result records requested and effective
devices. The controller terminates and recreates the heavy worker before the
CPU attempt so a partially initialized accelerator runtime cannot contaminate
the retry. General inference failures and native crashes are not automatically
rerun on CPU.

Ordinary per-file failures do not stop unrelated batch items. A systemic
artifact/provider failure pauses only matching work and offers:

- Retry the failed item.
- Retry failed and remaining matching items with faster-whisper.
- Skip affected items.
- Cancel the batch.

## Dependency and platform policy

Chatbook's minimal base installation remains free of heavy STT runtimes.

- Repurpose the existing `transcription_parakeet` extra from `parakeet-mlx` to
  `onnx-asr[cpu]==0.12.0`. The `hub` extra is not required because
  `ModelArtifactService` performs acquisition and providers load only managed
  local paths.
- Include the same Parakeet ONNX dependencies in the audio, video, and
  media-processing extras alongside faster-whisper.
- The first release declares only the CPU ONNX Runtime profile in managed
  Chatbook extras. `all-tools` explicitly selects this CPU baseline.
- Do not layer accelerator-specific ONNX Runtime packages on top of a CPU
  profile: upstream `onnx-asr` treats its CPU and GPU extras as mutually
  exclusive. Compatible user-managed accelerator runtimes remain
  opportunistic. Any future Chatbook accelerator extra must be a complete
  alternative installation profile with its own lockfile, resolver CI, and
  explicit incompatibility with CPU/all-tools profiles.
- Add transcribe.cpp under its own optional extra.
- Remove `parakeet-mlx`.
- Retain NeMo dependencies only where still required by unrelated retained
  providers such as Canary; removing NeMo Parakeet does not imply deleting the
  entire NeMo feature set.
- Update `optional_deps.py` and ingest capability reporting to probe packages
  lazily without importing native libraries at app startup.

The required CPU smoke matrix covers every selected-runtime wheel target:

- Linux x86_64.
- Linux aarch64.
- Windows x86_64.
- macOS arm64.
- macOS x86_64.

CPU availability is the baseline. Metal, CoreML, CUDA, TensorRT, DirectML,
Vulkan, ROCm, or other accelerators are opportunistic and must not be required
for provider availability.

Release lockfiles and CI pin exact native package builds. Pre-1.0 dependency
upgrades require an explicit compatibility and benchmark review. Resolver CI
installs every supported CPU extra alone and in documented combinations,
including `all-tools`, and rejects documentation or profiles that combine
mutually exclusive ONNX Runtime distributions. Direct ONNX Runtime consumers
such as local TTS must share a compatible CPU pin in those combinations.

## Curated transcribe.cpp catalog

The initial catalog targets the released `transcribe.cpp==0.1.3` binding and
contains:

| Model | Default | Optional | Reason |
| --- | --- | --- | --- |
| Whisper small | Q8_0 | F32 | Broad multilingual, auto-detection, segment timestamps, and speech-to-English translation baseline. |
| Canary 180M Flash | Q8_0 | F32 | Compact English/German/Spanish/French ASR and bidirectional translation; capability metadata states that timestamps are unavailable in the v1 port. |
| Moonshine tiny | Q8_0 | F32 | Small English batch model for constrained CPU systems. |
| Qwen3-ASR 0.6B | Q8_0 | BF16 | Modern multilingual, automatic-language-only family representative without admitting a model above 1B. |

Other upstream quantizations may be imported or browsed as uncurated artifacts
but are not recommended defaults. No transcribe.cpp model participates in
semantic `provider=default` routing.

Qwen3-ASR's reference artifact is BF16 because the pinned upstream port does
not publish F32. Its catalog entry declares automatic language selection and
no explicit language hints. Whisper small remains the curated multilingual
choice when the user needs explicit language constraints or translation.

The catalog excludes:

- Parakeet GGUF, because ONNX is Chatbook's supported Parakeet path.
- Streaming-only variants in the batch-first milestone.
- Gated and domain-specific models.
- Models above 1B.
- Every upstream variant not independently exercised by Chatbook.

Each catalog addition or runtime upgrade requires its own digest/license review,
provider contract smoke, and benchmark evidence.

## Configuration migration and legacy removal

Migration is versioned, idempotent, and written atomically. It preserves unknown
configuration keys and creates the normal config backup used by Chatbook.

- Persisted `parakeet` and `parakeet-mlx` provider selections become semantic
  `default`.
- Existing explicit language selections are preserved.
- Existing `auto` remains `auto`.
- Missing or empty language becomes `en`.
- Provider-specific legacy values that have no ONNX equivalent are retained
  only in a migration backup/legacy namespace or ignored with a migration
  notice; they are not misapplied to ONNX.
- A one-time notice reports when the effective provider changes.

Historical transcript provenance is never rewritten. New API/CLI requests that
explicitly name a removed provider receive `ProviderRemoved` with replacement
guidance; only persisted configuration is migrated automatically.

At the release boundary, remove:

- NeMo `parakeet` and `parakeet-mlx` provider code.
- Their provider registrations and exact UI/privacy-list entries.
- MLX-specific Parakeet dependency declarations and settings.
- Provider-specific caches and model-loading branches.
- Stale tests except explicit migration fixtures.

Do not delete external Hugging Face, NeMo, or MLX caches automatically because
they may be shared with other applications. Offer an explicit cleanup notice
instead.

Provider metadata becomes centralized so dictation, Settings, ingest
capability reporting, and troubleshooting screens no longer carry independent
provider lists.

## Data flow

### Interactive batch submission

1. Build a canonical request for each selected media item.
2. Resolve missing language to `en`.
3. Resolve semantic default routing.
4. Validate composed capabilities.
5. Resolve runtime package and artifact requirements for the batch.
6. If required, obtain explicit download/import confirmation and install once.
7. Enqueue jobs with resolved provider/model/artifact identities.
8. Dispatch light work to the general pool and audio/video work to the
   one-process heavy pool.
9. Lazily prepare provider-required audio.
10. Attach the current executor generation and attempt/job identity.
11. In the heavy worker, acquire the exact root and loaded-dependency operation
    leases when the selected identity is not already resident.
12. Load or reuse the one resident model; keep its lease while resident.
13. Transcribe with progress and cancellation.
14. Normalize result and provenance.
15. Return a generation-fenced complete parsed payload to the existing
    single-writer stage.
16. Persist transcript content, compatibility model summary, and versioned
    provenance atomically, then mark the job complete.

Failure before step 15 produces no media transcript write. Failure during the
short writer transaction rolls back through the existing database transaction
boundary. The artifact lease is not request-scoped: it is released when the
resident model closes or its worker generation exits.

### Explicit retry

1. Preserve the original failure and its provider provenance.
2. User selects **Retry with faster-whisper**.
3. Create a new request linked to the failed job as a retry.
4. Run faster-whisper package/model preflight.
5. Enqueue only after requirements are satisfied.
6. Persist the successful retry with faster-whisper provenance plus the bounded
   failed-attempt snapshot, so lineage survives job-history pruning.

## Testing strategy

### Dependency-free unit tests

Normal unit tests use provider and artifact fakes. They cover:

- Every default-routing row.
- `en` as the new/unset default.
- Preservation of saved `auto`.
- Manual-provider override and capability failures.
- Requested/effective/detected language semantics.
- V3 routing-only language selection, `effective_language=auto`, and the
  `requested_language_not_enforced` warning.
- Translation routing.
- Declared versus runtime capability mismatch.
- Result normalization and provenance.
- Stable error codes and action eligibility.
- One permitted device fallback and no cross-engine fallback.
- Versioned, idempotent config migration.
- Provider-registry uniqueness and removal aliases.
- Progress phases and cancellation state transitions.
- Executor-generation fencing, exactly one terminal state, and stale callback
  rejection.
- Retry provenance remaining intelligible after the failed job is pruned.

### Artifact integration tests

A local HTTP fixture, not public network access, covers:

- Immutable-revision resolution.
- Interrupted and resumed downloads.
- Cancellation leaving only staging.
- Size and SHA mismatch.
- Corrupt active records.
- Concurrent installs from separate processes.
- Dependency-closure preflight and atomic visibility.
- Root readiness written last and crash recovery hiding partial dependency
  closures.
- A missing or corrupt VAD dependency preventing Parakeet activation.
- Shared operation leases and exclusive deletion.
- An idle resident model retaining root and dependency leases across same-model
  requests.
- Worker death releasing a lease.
- Atomic activation and rollback to the previous version.
- Insufficient disk space.
- GGUF validation and local copy.
- Descriptor-backed offline multi-file ONNX import.
- Rejection of unknown, modified, or manifestless ONNX graph import.
- Missing external data.
- Absolute and traversal external-data references.
- Symlink and irregular-file rejection.
- Staging cleanup that cannot escape the managed root.

### Provider contract tests

Gated real-model jobs exercise:

- Parakeet v2 INT8 and F32.
- Parakeet v3 INT8 and F32.
- faster-whisper's configured baseline.
- Every curated transcribe.cpp family and its default Q8_0 artifact.
- Qwen3-ASR BF16 reference loading and rejection of explicit language hints.
- File and bounded buffer input.
- Long-form VAD.
- Single-segment VAD batching and cancellation checks before every segment
  batch.
- Timestamp normalization where declared.
- Same-model reuse.
- Worker recycle on model identity change.
- Cooperative cancellation and force stop.
- Generation-fenced stale-result rejection and decoder subprocess-tree cleanup.
- Native crash simulation.

### Platform tests

Every wheel-supported target runs:

- Clean installation of the appropriate extras.
- Clean resolution of every documented CPU extra combination, including
  `all-tools`, without conflicting ONNX Runtime distributions.
- Import/probe without startup-time native imports.
- CPU inference smoke.
- INT8 Parakeet v2 and v3 smoke.
- One small transcribe.cpp curated-model smoke.
- faster-whisper retry smoke.
- Process creation, termination, and crash recovery.
- No surviving FFmpeg/decoder descendants after cancellation or force stop.
- Cross-process artifact locks and deletion leases.

Accelerator tests run separately and never substitute for CPU coverage.

### End-to-end coverage

End-to-end tests include:

- Audio and video ingestion.
- Several files sharing one preflight and one resident model.
- Light document ingestion continuing beside the isolated heavy pool.
- One heavy-worker crash not failing light parse jobs.
- Buffer transcription compatibility.
- Bounded/coalesced dictation requests.
- Pause-batch behavior for interactive dictation.
- Missing-model download consent.
- Explicit faster-whisper retry.
- Application shutdown during download, model load, and inference.
- Both legacy provider-ID migrations.
- Historical transcript provenance remaining unchanged.

## Evaluation corpus

A versioned corpus manifest covers:

- Clean and noisy English with multiple accents.
- Silence, music, non-speech noise, and long pauses.
- Short clips, clips beyond 30 seconds, ten-minute media, and a long-form stress
  sample.
- Deep Parakeet v3 slices for Spanish, French, German, Polish, Greek, Russian,
  and Ukrainian.
- Enough licensed samples to compute a per-language gate for every language
  enabled in the validated v3 routing set.
- Japanese, Mandarin, and Arabic samples verifying faster-whisper routing.
- Variable sample rates, channel layouts, encodings, malformed audio, and
  video-extracted audio.

Only small redistributable fixtures live in the repository. Larger public
benchmark assets are obtained through a reproducible script with source,
license, immutable revision, size, and digest metadata. Evaluation normalizers
are language-appropriate and versioned with the results.

### Early INT8 artifact qualification

Artifact qualification precedes adapter implementation and default-promotion
work. Using immutable revisions, the first model task compares the stock
Parakeet v2 and v3 INT8 exports with their F32 artifacts on:

- Short clean and noisy samples.
- Samples beyond the runtime's direct-input limit.
- Ten-minute VAD/long-form samples.
- Every deep-language slice used for v3.

The stock INT8 artifact cannot become Chatbook's default if it fails the
INT8-versus-F32 quality gate. An alternative quantized export is considered
only if its source, conversion and calibration procedure, license, immutable
files, and hashes are reproducible and reviewed. Until a qualifying INT8
artifact exists, semantic default promotion and legacy-provider removal remain
blocked; Chatbook does not silently substitute F32 for an INT8 selection.

## Default-promotion gates

Before changing the shipped semantic default:

- Parakeet v2 INT8 aggregate English WER is no more than 1.0 absolute
  percentage point worse than the current faster-whisper base/int8 baseline,
  and no English slice is more than 3 points worse.
- Every evaluated language declares one primary metric and normalizer before
  results are run. WER and CER populations are aggregated and gated separately;
  they are never averaged into one mixed value.
- For each v3 primary-metric population, the macro-average error rate is no more
  than 1.5 absolute points worse than the same-language faster-whisper
  baseline, and no routed language is more than 4 points worse.
- A v3 language that fails its gate is excluded from Chatbook's validated
  routing set and resolves to faster-whisper.
- Within each primary-metric population, INT8 is within 0.5 aggregate points of
  F32.
- Accuracy deltas use paired samples and report a paired-bootstrap 95 percent
  confidence interval. Promotion requires both the point estimate and the
  confidence interval's adverse bound to satisfy the applicable threshold.
- Silence fixtures produce no non-empty transcript.
- Timestamp sequences are monotonic, nonnegative, within audio duration, and
  correctly mapped into Chatbook segments.
- Warm CPU throughput is faster than real time on the designated reference
  machine. The benchmark manifest records exact CPU, memory, operating system,
  Python, ONNX Runtime, execution-provider, thread, VAD, and model-artifact
  revisions so the gate is reproducible.
- Peak Parakeet INT8 heavy-worker RSS is at most 3 GiB.
- A 100-file same-model batch does not reload the model per file, and
  post-warm-up RSS remains within 15 percent.
- Changing model identity replaces the heavy-worker PID and returns the prior
  process memory to the OS.
- Cancellation, force stop, and native-crash recovery leave no partial media
  record or artifact activation.
- Force-stop tests prove that detached-generation callbacks cannot persist a
  result and that decoder subprocess descendants are gone before temporary
  cleanup.
- Same-model idle residency continues to block deletion of the root artifact
  and loaded dependencies until model close or worker exit.

Initial benchmark reports are committed or attached as versioned release
artifacts. Later runtime, model, quantization, VAD, or decoder changes rerun the
same gates.

## Legacy-removal gates

The old Parakeet providers are removed in the landing release only after:

- Every required platform smoke passes.
- Batch file and buffer contracts pass.
- Dictation backpressure behavior is usable and bounded.
- Artifact download, import, resume, verification, lease, and deletion tests
  pass.
- Config migration is idempotent and preserves explicit languages.
- Default routing and explicit retry behavior pass.
- Quality, throughput, timestamp, and memory gates pass.
- Heavy-worker crash containment and force stop pass.
- No live UI/service code retains old provider IDs outside migration fixtures.
- Package extras and optional-dependency messages match the new runtime.
- License, source, hash, runtime compatibility, and local-data privacy
  documentation is complete.

If any gate fails, Parakeet ONNX remains non-default and the legacy providers
remain until the same release candidate satisfies the gate. Development may
stage adapter and migration work, but the final merge/release state does not
remove the old implementations prematurely.

## Security, privacy, and licensing

- All source, destination, and temporary paths use Chatbook's path-validation
  boundary.
- Remote metadata and filenames are data, never trusted path fragments.
- Artifact promotion never follows symlinks.
- Hashes are compared before an artifact becomes active.
- Credentials are never stored in artifact manifests or logged.
- Logs prefer job/artifact IDs and sanitize unnecessary local paths.
- Local microphone buffers remain memory-only for compatibility transcription.
- External server/provider privacy is not conflated with local providers.
- Every curated artifact records the upstream model license and runtime
  license.
- The transcribe.cpp binding/runtime is MIT; individual model licenses remain
  separately visible and enforceable.
- A future decision to redistribute model binaries inside Chatbook requires a
  separate packaging and license review.

## Observability

Structured events record:

- Routing decision and reason.
- Preflight outcome.
- Artifact installation phases and verification outcome.
- Queue wait, preparation, model load, inference, and write durations.
- Requested/effective execution provider.
- Model reuse versus worker recycle.
- Cancellation request, cooperative completion, or force stop.
- Heavy-worker exit phase and circuit state.
- Retry lineage and final provider.

Events never record transcript audio, credentials, or full local source paths
by default.

## Alternatives considered

### Make transcribe.cpp the universal STT runtime

Rejected. It maximizes model breadth but makes a young pre-1.0 native project
the sole reliability boundary, complicates compatibility with Chatbook's
current result contract, and duplicates the better-established Parakeet ONNX
path.

### Add all new engines as branches in `TranscriptionService`

Rejected. The service already mixes imports, routing, caches, inference, and
normalization. More branches would preserve duplicated capability and UI
metadata and make later provider removal harder.

### Keep provider-specific downloaders and caches

Rejected. GGUF and ONNX need the same revision, digest, staging, activation,
lease, deletion, disk, and provenance semantics. Separate implementations
would drift and would not create the requested reusable artifact foundation.

### Route `auto` to Parakeet v3

Rejected. Parakeet v3 internally selects among its supported languages, but the
reviewed `onnx-asr` result contract does not expose a trustworthy detected
language identity. Chatbook could not reliably detect unsupported input or
explain a recovery decision.

### Add a language-identification prepass

Rejected for the first milestone. It adds another runtime, model artifact,
download, latency, evaluation surface, and failure mode. faster-whisper already
provides the required automatic-language path.

### Silently fall back to faster-whisper

Rejected. It hides provider reliability problems, can cause unexpected model
downloads or compute, and corrupts user understanding of transcript
provenance.

### Keep the dispatch-only heavy-lane cap

Rejected for model caching. The shared pool can place sequential heavy jobs in
different processes, multiplying model residency, while one native worker
failure can affect unrelated in-flight parses.

### Load and unload the model for every file

Rejected as the default batch strategy. It bounds memory but makes model-load
latency dominate batches of short files and repeatedly exercises native
allocators.

### Cache a model in every general parse worker

Rejected. It improves per-worker speed but permits several copies of a
multi-gigabyte model and makes residency depend on incidental task placement.

### Replace installed artifact directories in place

Rejected. It weakens rollback and is not reliably atomic for non-empty
directories across platforms. Immutable versions plus an atomic small active
record are simpler and safer.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Pre-1.0 runtime/API churn | Exact or bounded release pins, manifest compatibility, gated upgrades, contract tests. |
| ONNX model or external-data corruption | Multi-file manifest, path containment, per-file hashes, atomic activation. |
| Native crash or illegal instruction | Lazy probe, separate spawned heavy worker, CPU baseline, crash circuit, explicit retry. |
| Native allocator retains memory | One-model worker and process recycle on identity change. |
| Dictation stalls behind long batch work | Bounded coalescing, next-job priority, visible busy state, pause batch dispatch. |
| V3 quality varies by language | Chatbook-validated routing subset and per-language gates. |
| First-run default model is absent | Pre-enqueue consent flow; no worker prompt or silent download. |
| Arbitrary remote GGUF appears trusted | Separate curated, integrity-verified, and local provenance labels. |
| Disk usage doubles during update/import | Up-front free-space calculation, immutable staging, visible installed/staging totals. |
| Legacy config breaks after provider removal | Versioned idempotent migration, preserved language, one-time notice, migration fixtures. |
| Artifact delete races inference | Interprocess operation lease and exclusive deletion lock. |
| Resident model outlives request lease | Worker owns the root/dependency lease set for the full resident-model lifetime. |
| Stale result arrives after force stop | Attempt and executor-generation fencing before the writer path. |
| Decoder survives worker cancellation | Platform-owned subprocess tree is terminated before temporary cleanup. |
| Retry ancestor is pruned | Successful provenance embeds a bounded failed-attempt snapshot. |
| CPU and accelerator runtimes conflict | CPU-only managed v1 extras; future accelerators are alternative profiles. |
| Untrusted ONNX graph exhausts the UI process | V1 imports only catalog-matching bundles; future arbitrary parsing is isolated and capped. |

## Implementation-planning boundary

Before implementation planning:

1. Create or select atomic Backlog tasks in dependency order.
2. Link [ADR-025](../../../backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md)
   from each affected task and plan.
3. Create one independently reviewable task/PR for each dependency-ordered
   delivery slice:
   1. Prove the cross-platform lease primitive.
   2. Qualify Parakeet v2/v3 INT8 artifacts against F32.
   3. Build shared artifact descriptors, activation/readiness, leases, and
      deletion.
   4. Add managed download, resume, verification, and recovery.
   5. Renovate the browser for curated/remote/installed inventory.
   6. Add bounded local GGUF import.
   7. Add descriptor-backed local ONNX-bundle import.
   8. Add coordinator and provider contracts.
   9. Add durable transcript provenance and retry-lineage migration.
   10. Add the generation-fenced heavy executor and subprocess-tree lifecycle.
   11. Integrate Parakeet ONNX routing and batch ingestion.
   12. Restore bounded dictation-buffer compatibility.
   13. Add the curated optional transcribe.cpp provider.
   14. Promote defaults, migrate configuration, and remove legacy Parakeet only
       after every release gate passes.
4. Record dependencies only on already-created lower-ID Backlog tasks; no task
   depends on a future task.
5. Select the concrete cross-platform interprocess locking primitive and record
   it in the implementation plan without weakening this design's crash-release
   requirement.
6. Pin exact artifact revisions, filenames, sizes, hashes, and licenses only
   after revalidating them against the upstream release at implementation time.

No production code is authorized by this design document alone.

## References

- [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp)
- [transcribe.cpp v0.1.3](https://github.com/handy-computer/transcribe.cpp/releases/tag/v0.1.3)
- [transcribe.cpp v0.1.3 model matrix](https://github.com/handy-computer/transcribe.cpp/blob/v0.1.3/README.md)
- [transcribe.cpp Whisper small](https://github.com/handy-computer/transcribe.cpp/blob/v0.1.3/docs/models/whisper-small.md)
- [transcribe.cpp Canary 180M Flash](https://github.com/handy-computer/transcribe.cpp/blob/v0.1.3/docs/models/canary-180m-flash.md)
- [onnx-asr](https://github.com/istupakov/onnx-asr)
- [onnx-asr v0.12.0 result and decoding contract](https://github.com/istupakov/onnx-asr/blob/v0.12.0/src/onnx_asr/asr.py)
- [onnx-asr v0.12.0 Parakeet TDT adapter](https://github.com/istupakov/onnx-asr/blob/v0.12.0/src/onnx_asr/models/nemo.py)
- [onnx-asr v0.12.0 VAD batching](https://github.com/istupakov/onnx-asr/blob/v0.12.0/src/onnx_asr/vad.py)
- [onnx-asr v0.12.0 dependency profiles](https://github.com/istupakov/onnx-asr/blob/v0.12.0/pyproject.toml)
- [ONNX external data](https://onnx.ai/onnx/repo-docs/ExternalData.html)
- [ONNX external-data security](https://onnx.ai/onnx/repo-docs/ExternalDataSecurity.html)
- [NVIDIA Parakeet TDT 0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
- [NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
