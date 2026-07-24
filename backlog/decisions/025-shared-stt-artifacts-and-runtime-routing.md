# ADR-025: Adopt Parakeet ONNX routing, optional transcribe.cpp, and shared model artifacts

Status: Accepted
Date: 2026-07-23
Related Tasks: TASK-505 through TASK-518
Supersedes: N/A

## Decision

Chatbook will introduce a provider-neutral STT coordinator behind the existing
`TranscriptionService` compatibility facade.

The semantic default routing policy is:

- New and unset language values resolve to `en`.
- Explicit `en` uses Parakeet TDT 0.6B v2 ONNX through `onnx-asr`.
- Explicit, Chatbook-validated non-English Parakeet v3 languages use Parakeet
  TDT 0.6B v3 ONNX.
- `auto`, unsupported languages, and translation use faster-whisper.
- Parakeet INT8 is the default artifact; F32 is an explicit option.

The policy applies to semantic `provider=default`. An exact user-selected
provider/model is honored only when its declared and runtime capabilities
satisfy the request.

The pinned `onnx-asr` v3 TDT adapter does not enforce a caller-supplied
language. An explicit supported language selects the validated v3 model but is
not passed as a decoder constraint. Such results record the requested language,
`effective_language=auto`, nullable detected language, and
`requested_language_not_enforced`; the UI does not claim that v3 is forced to
the requested language.

Chatbook will adopt `transcribe.cpp` as an optional, exactly versioned GGUF
provider with a curated initial catalog: Whisper small, Canary 180M Flash,
Moonshine tiny, and Qwen3-ASR 0.6B. transcribe.cpp is not an automatic routing
target or the sole STT runtime. Qwen3-ASR uses Q8_0 by default and BF16 as its
reference-precision option because the pinned port does not publish F32. Its
catalog capability is automatic-language-only; an explicit language constraint
must fail closed until the user changes the request to `auto`.

Cross-engine fallback will never be silent. Eligible failures offer an explicit
**Retry with faster-whisper** action that creates a new request and preserves
both attempts' provenance. One automatic accelerator-to-CPU initialization
retry is permitted within the same provider and model, but only after recycling
the heavy worker.

Chatbook will create a format-neutral `ModelArtifactService` as the sole writer
for managed model artifacts. It will provide immutable-revision descriptors,
per-file size and SHA-256 verification, resumable staging, atomic activation,
multi-file ONNX bundle support, GGUF support, managed local import,
interprocess operation leases, safe deletion, disk preflight, and precise
provenance labels. Descriptors can require other exact artifacts; Parakeet
long-form support depends on a separately pinned managed VAD artifact. Remote
content is called integrity verified only when the repository independently
supplies the expected digest. STT is the first consumer. LLM artifact migration
is outside this decision.

Dependency closures become loadable through a root readiness record written
last after every immutable dependency revision is verified; the design does not
claim a multi-directory atomic transaction. Initial local ONNX import is an
offline import path for bundles whose files, sizes, and digests match an
existing Chatbook catalog descriptor. Unknown or modified graphs are rejected.
Arbitrary graph inspection is deferred until it can run through a separately
pinned parser in a resource-capped validation process.

Newly persisted media transcripts will store a validated, versioned provenance
document in nullable `Media.transcription_provenance_json`, written atomically
with transcript content. `Media.transcription_model` remains a compatibility
summary. The document records provider, model, immutable root artifact revision,
exact loaded dependency revisions, precision, effective execution provider,
requested/effective/detected language, task, attempt identity, and explicit
retry lineage. Existing records remain valid without synthesized provenance,
and `Transcripts.whisper_model` is not repurposed. The Library ingest-job store
also gains `retry_of_job_id` and structured STT failure provenance; a retry
count alone is not sufficient to preserve attempt lineage. Because bounded job
history may prune the failed row, successful retry provenance also embeds a
bounded, sanitized failed-attempt snapshot. The job link is navigational; the
snapshot is the durable lineage record.

An app-owned `LocalSTTExecutor` will run audio/video ingestion in a separate
spawn-context heavy-media pool with exactly one worker and at most one resident
STT model. Same-identity jobs reuse the model. A provider, model, root artifact,
dependency-closure fingerprint, precision, or device change recycles the worker
rather than relying on native libraries to release memory in-process. Library
ingestion and dictation share the controller; neither parse workers nor facade
instances create private heavy processes. Buffer transcription remains
compatible without claiming true streaming. Dictation coalescing is capped by
duration and bytes, pauses capture visibly on overflow, and never silently
drops audio.

The heavy worker acquires leases for the root artifact and every loaded
dependency before model load and retains them for the entire resident-model
lifetime, including idle same-model reuse. Deletion of the root or VAD
dependency remains blocked until model close or worker exit. Every heavy
request and callback carries an executor generation and attempt identity; force
stop detaches the old generation before termination so stale results cannot
reach the writer. Decoder subprocess descendants are owned and terminated as a
platform process tree before temporary cleanup.

Cross-platform shared/exclusive artifact locking is gated by a prerequisite
technical spike that proves root/dependency load-lease deletion blocking and
automatic crash release across multiple requests and idle residency on Windows,
macOS, and Linux, without a parent-held lease leak. Artifact-service
implementation does not proceed until a concrete primitive passes that proof.

The first release manages only the CPU `onnx-asr` dependency profile.
`all-tools` also selects that CPU baseline. Accelerator ONNX Runtime packages
are not layered on top because upstream CPU and GPU profiles conflict; a future
managed accelerator option must be a complete alternative installation profile
with independent resolver and runtime qualification.

Stock Parakeet v2/v3 INT8 artifacts must pass an early short- and long-form
comparison against their F32 artifacts before being accepted as default
candidates. An alternative INT8 export requires reproducible conversion and
calibration inputs plus normal artifact, license, and review evidence. A failed
INT8 qualification blocks default promotion and legacy-provider removal.

The NeMo `parakeet` and `parakeet-mlx` providers, registrations, and
provider-specific settings will be removed in the landing release only after
the design's platform, batch, buffer, migration, artifact, accuracy,
performance, memory, cancellation, and crash-recovery gates pass. External
model caches will not be deleted automatically.

## Context

The current transcription service combines optional imports, provider routing,
model caches, inference, and result shaping in one large module. Dictation and
UI code repeat provider identities and privacy assumptions. Adding both
Parakeet ONNX and transcribe.cpp as more branches would deepen those
maintenance and capability-consistency problems.

The Library's current heavy-lane cap is a dispatcher constraint over a shared
parse pool, not a separate execution boundary. Heavy jobs can land in different
workers and leave multiple large native models resident. A native worker death
can also affect unrelated parse jobs in the same pool generation.

The existing GGUF browser/downloader does not provide the artifact guarantees
required for native inference: immutable source revision, expected per-file
digests, resumable isolated staging, atomic activation, multi-file bundles, or
cross-process load/delete leases.

Current audio and video persistence writes transcript text to `Media.content`
but does not pass even the existing `transcription_model` field. The existing
schema therefore cannot satisfy normalized provider/artifact/device/language
provenance or durable retry lineage without an explicit migration and
export/import contract.

Parakeet v2 is English-only. Parakeet v3 supports a defined group of European
languages and internally selects among them, but the reviewed `onnx-asr` result
contract neither accepts an enforced language hint nor exposes a reliable
detected-language identity. Routing `auto` to v3 would prevent Chatbook from
recognizing unsupported-language input or explaining a safe recovery.
faster-whisper therefore remains the automatic and broad-language path.

The pinned `onnx-asr` package defines mutually exclusive CPU and GPU dependency
profiles. Chatbook already has other optional ONNX Runtime consumers, so
additive accelerator extras would create an installation contract that cannot
be made reliable across all documented extra combinations.

This work requires a canonical ADR because it changes provider/runtime
boundaries, dependency policy, process ownership, model storage and trust,
configuration migration, and cross-module service contracts.

ADR-020 remains the authority for cloud-provider model-ID refresh. It does not
own local binary artifacts and is not superseded by this decision.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Make transcribe.cpp the universal STT runtime | Places every STT path behind a young pre-1.0 native runtime and gives up the more mature Parakeet ONNX integration path. |
| Add ONNX and transcribe.cpp branches to the current service | Preserves the routing/cache/import monolith and repeated UI capability lists. |
| Route `auto` to Parakeet v3 | The selected wrapper does not expose trustworthy language identity, so unsupported input cannot be detected or recovered transparently. |
| Add a language-identification model before routing | Adds a runtime, model download, latency, failure surface, and evaluation obligation that faster-whisper already avoids. |
| Silently retry failed work with faster-whisper | Hides reliability failures, compute changes, and transcript provenance from the user. |
| Keep provider-specific model downloaders | Duplicates security, integrity, staging, disk, lease, and deletion behavior across formats and consumers. |
| Keep only the shared parse pool's heavy dispatch cap | Does not guarantee worker affinity, bounded model residency, or crash isolation from light parse jobs. |
| Load and unload a model for every file | Bounds residency but makes model-load time dominate batches and repeatedly stresses native allocators. |
| Cache one model in each general parse worker | Can retain several copies of a multi-gigabyte model based on incidental scheduling. |
| Replace installed artifact directories in place | Weakens rollback and is not reliably atomic for populated directories across supported operating systems. |
| Delete legacy model caches during migration | Caches may be shared with other applications; automatic deletion is unsafe and outside provider migration. |

## Consequences

- `TranscriptionService` becomes a thin compatibility facade over a
  coordinator and explicit provider adapters.
- Provider discovery and capability metadata are centralized and available
  without importing native libraries at startup.
- `en` is the default language, making Parakeet v2 the normal installed-model
  path.
- `auto`, unsupported languages, and translation continue through
  faster-whisper.
- Parakeet v3 routing is limited to languages passing Chatbook's evaluation
  gates, but the requested language is transparently routing-only rather than
  an enforced decoder constraint.
- transcribe.cpp expands optional model-family breadth without determining
  default reliability.
- Managed downloads require explicit consent and never occur in ingestion
  workers.
- Curated, remote integrity-verified, and local artifacts have visibly
  different provenance.
- A remote artifact without a repository-supplied digest is locally recorded,
  not labeled independently integrity verified.
- Parakeet installation and preflight include its exact VAD dependency; a
  missing dependency cannot trigger an implicit provider download.
- Artifact installs are immutable and versioned; an atomic small record selects
  the active version.
- Dependency closures expose a root readiness record only after all exact
  revisions are present; their canonical fingerprint participates in resident
  model identity and their individual revisions are persisted as provenance.
- Downloads and local imports may require space for staging or a managed copy.
- Initial local ONNX import is descriptor-backed; arbitrary graph parsing is
  not performed in the UI or inference worker.
- Deletion of the root or a loaded dependency is blocked for the complete
  lifetime of a resident model while its worker holds operation leases.
- Audio/video parsing moves to a dedicated one-process pool; light document
  parsing keeps its existing parallel pool.
- Native model changes recycle the heavy process, trading process-start latency
  for predictable memory reclamation.
- Force stop generation-fences the writer and terminates decoder subprocess
  descendants before temporary cleanup.
- Dictation buffer compatibility is retained through the app-owned executor,
  with bounded visible backpressure; true streaming and preemption are
  deferred.
- Accelerator support is opportunistic; CPU inference is the release baseline.
- Managed v1 extras install only the CPU ONNX Runtime profile; future managed
  accelerator profiles must be mutually exclusive alternatives.
- Accelerator initialization fallback begins in a fresh worker process.
- The minimal Chatbook install remains free of STT runtimes, while audio,
  video, and media-processing extras include Parakeet ONNX and faster-whisper.
- Legacy Parakeet provider IDs receive a versioned, idempotent persisted-config
  migration. Historical transcript provenance is unchanged.
- New transcripts receive versioned provenance through a database migration;
  old records remain readable with null provenance.
- Successful retry provenance remains intelligible after failed-job pruning
  because it contains a bounded failed-attempt snapshot.
- Removal of old providers is blocked on reproducible quality, performance,
  memory, platform, migration, and recovery gates.
- A future LLM migration can reuse the artifact service but requires its own
  task and provider-specific design.

## Rollback plan

- Before the release gates pass, keep the legacy providers registered and leave
  semantic default selection on the current stable path.
- If Parakeet ONNX fails a release gate, keep it available as an explicit
  provider while retaining the prior default and legacy implementations.
- If no reviewed INT8 artifact passes early qualification, do not substitute
  F32 silently; retain the prior semantic default and legacy providers.
- If transcribe.cpp proves unstable, omit its optional extra and curated catalog
  without changing Parakeet ONNX or faster-whisper routing.
- If the artifact browser renovation must be rolled back, keep new installs
  disabled rather than reverting to unverified direct writes. Existing
  immutable installed artifacts remain readable through their manifests.
- Never silently fall back during rollback. The selected and effective provider
  remain visible.
- Config migration is applied only in the release that removes the old
  providers. Historical transcript records and external caches require no
  rollback.

## Links

- [Design spec](../../Docs/superpowers/specs/2026-07-23-stt-parakeet-onnx-transcribe-cpp-design.md)
- [ADR-020: Automatic cloud model catalog refresh](020-automatic-model-catalog-refresh.md)
- [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp)
- [transcribe.cpp v0.1.3](https://github.com/handy-computer/transcribe.cpp/releases/tag/v0.1.3)
- [transcribe.cpp Qwen3-ASR 0.6B model notes](https://github.com/handy-computer/transcribe.cpp/blob/v0.1.3/docs/models/qwen3-asr-0.6b.md)
- [onnx-asr](https://github.com/istupakov/onnx-asr)
- [onnx-asr v0.12.0 result and decoding contract](https://github.com/istupakov/onnx-asr/blob/v0.12.0/src/onnx_asr/asr.py)
- [onnx-asr v0.12.0 Parakeet TDT adapter](https://github.com/istupakov/onnx-asr/blob/v0.12.0/src/onnx_asr/models/nemo.py)
- [onnx-asr v0.12.0 VAD batching](https://github.com/istupakov/onnx-asr/blob/v0.12.0/src/onnx_asr/vad.py)
- [onnx-asr v0.12.0 dependency profiles](https://github.com/istupakov/onnx-asr/blob/v0.12.0/pyproject.toml)
- [onnx-asr v0.12.0 local resolver](https://github.com/istupakov/onnx-asr/blob/v0.12.0/src/onnx_asr/resolver.py)
- [ONNX external data](https://onnx.ai/onnx/repo-docs/ExternalData.html)
- [ONNX external-data security](https://onnx.ai/onnx/repo-docs/ExternalDataSecurity.html)
- [Community Parakeet v3 INT8 comparison requiring local verification](https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-smoothquant-onnx)
- [NVIDIA Parakeet TDT 0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
- [NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
