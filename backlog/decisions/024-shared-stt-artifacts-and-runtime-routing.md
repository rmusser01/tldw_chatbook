# ADR-024: Adopt Parakeet ONNX routing, optional transcribe.cpp, and shared model artifacts

Status: Accepted
Date: 2026-07-23
Related Task: N/A — create or select Backlog tasks before implementation planning
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

Chatbook will adopt `transcribe.cpp` as an optional, exactly versioned GGUF
provider with a curated initial catalog: Whisper small, Canary 180M Flash,
Moonshine tiny, and Qwen3-ASR 0.6B. transcribe.cpp is not an automatic routing
target or the sole STT runtime.

Cross-engine fallback will never be silent. Eligible failures offer an explicit
**Retry with faster-whisper** action that creates a new request and preserves
both attempts' provenance. One automatic accelerator-to-CPU initialization
retry is permitted within the same provider and model.

Chatbook will create a format-neutral `ModelArtifactService` as the sole writer
for managed model artifacts. It will provide immutable-revision descriptors,
per-file size and SHA-256 verification, resumable staging, atomic activation,
multi-file ONNX bundle support, GGUF support, managed local import,
interprocess operation leases, safe deletion, disk preflight, and precise
provenance labels. STT is the first consumer. LLM artifact migration is outside
this decision.

Audio/video ingestion will run in a separate spawn-context heavy-media pool
with exactly one worker and at most one resident STT model. Same-identity jobs
reuse the model. A provider/model/precision/device change recycles the worker
rather than relying on native libraries to release memory in-process. Bounded
buffer transcription may use the same lane for compatibility, without claiming
true streaming.

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

Parakeet v2 is English-only. Parakeet v3 supports a defined group of European
languages and internally selects among them, but the reviewed `onnx-asr` result
contract does not expose a reliable detected-language identity. Routing `auto`
to v3 would prevent Chatbook from recognizing unsupported-language input or
explaining a safe recovery. faster-whisper therefore remains the automatic and
broad-language path.

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
  gates.
- transcribe.cpp expands optional model-family breadth without determining
  default reliability.
- Managed downloads require explicit consent and never occur in ingestion
  workers.
- Curated, remote integrity-verified, and local artifacts have visibly
  different provenance.
- Artifact installs are immutable and versioned; an atomic small record selects
  the active version.
- Downloads and local imports may require space for staging or a managed copy.
- Deletion is blocked while a worker holds an operation lease.
- Audio/video parsing moves to a dedicated one-process pool; light document
  parsing keeps its existing parallel pool.
- Native model changes recycle the heavy process, trading process-start latency
  for predictable memory reclamation.
- Dictation buffer compatibility is retained, but true streaming and
  preemption are deferred.
- Accelerator support is opportunistic; CPU inference is the release baseline.
- The minimal Chatbook install remains free of STT runtimes, while audio,
  video, and media-processing extras include Parakeet ONNX and faster-whisper.
- Legacy Parakeet provider IDs receive a versioned, idempotent persisted-config
  migration. Historical transcript provenance is unchanged.
- Removal of old providers is blocked on reproducible quality, performance,
  memory, platform, migration, and recovery gates.
- A future LLM migration can reuse the artifact service but requires its own
  task and provider-specific design.

## Rollback plan

- Before the release gates pass, keep the legacy providers registered and leave
  semantic default selection on the current stable path.
- If Parakeet ONNX fails a release gate, keep it available as an explicit
  provider while retaining the prior default and legacy implementations.
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
- [onnx-asr](https://github.com/istupakov/onnx-asr)
- [NVIDIA Parakeet TDT 0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
- [NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
