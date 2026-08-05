# ADR-041: Use explicit local GGUF paths before managed acquisition

Status: Accepted
Date: 2026-08-02
Related Tasks: TASK-597, TASK-601, TASK-604, TASK-1915
Amends: ADR-025

## Decision

The first usable transcribe.cpp release accepts one explicitly configured local
GGUF path. Chatbook validates that path with a bounded, native-runtime-free
admission boundary when selected and inside the existing spawn-isolated
ingestion worker immediately before native model load.

The first release does not require the file to be copied into the managed model
artifact store. Curated GGUF catalogs, verified downloads, managed local import,
activation, and artifact promotion are deferred until after direct-path
transcription works end to end.

The local path is provider configuration, not transcript provenance. Only the
path persists; source identity is recomputed per admission and is not retained
as an expected identity. A compatible replacement at the same path is accepted
as the new current file rather than treated as a persistent-identity violation.
TASK-601 may later use the same snapshot for resident-worker reuse/recycle.

Direct-local transcript provenance uses `artifact_root = null` and an empty
artifact dependency set. It records provider/model/precision/device/language/
attempt fields but neither the path nor a fabricated artifact revision. The
provider package is pinned, but the first release does not add a runtime-version
field to the persisted provenance schema. This explicitly amends ADR-025's
immutable-root expectation for the manual direct-local provider. Exact-byte
reproducibility is unavailable until managed acquisition lands.

The file is never an automatic-routing candidate and is not labeled installed,
curated, or integrity verified. transcribe.cpp remains an exact manual provider
and never a silent fallback.

The first usable provider reuses the existing Library parse pool and its
one-heavy-job gate. It does not wait for TASK-601's dedicated resident executor,
fine-grained cross-process cancellation, artifact leases, or heavy/light pool
separation. TASK-601 remains later hardening; managed GGUF acquisition remains
TASK-1915.

## Context

ADR-025 chose a shared managed artifact core and required transcribe.cpp to
consume managed handles. That boundary provides strong immutability, digest,
recovery, and deletion guarantees, but implementing it before the provider
delays the first user-visible transcription outcome.

Users who already possess a compatible GGUF need only select it and run the
pinned provider. The standard filesystem is sufficient for that first path.
Managed acquisition can later supply a managed path to the same provider
without changing inference behavior.

## Consequences

- TASK-597 contains parser, compatibility, and direct-file admission only.
- Existing managed-import prototype code may remain in a private module marked
  for TASK-1915, but it has no public export, registration, production import,
  call site, or execution path until that later task reviews and activates it.
- TASK-604 owns file selection/configuration, the real Library batch selector
  and production wiring, and actual transcribe.cpp use.
- The ingestion worker revalidates immediately before native model load but
  cannot eliminate a race between validation and the native runtime reopening
  the path.
- After the single per-job native load, the worker reads authoritative
  capabilities, builds the exact per-job declaration and sealed registry, and
  lets coordinator preflight equality-check that already-loaded observation
  before inference. No coordinator or registry contract is weakened.
- The first release loads once per ingest job. A native worker crash is
  contained by the existing pool monitor but can make other in-flight parse
  jobs retryable; TASK-601 later adds dedicated heavy-process isolation and
  model residency.
- Missing, symlinked, unreadable, or newly incompatible current files fail
  clearly and require re-selection.
- A different compatible file at the same path is admitted as the current
  model on the next job; no cross-restart expected snapshot is persisted.
- Direct-local transcripts have no immutable artifact revision or exact-byte
  reproducibility claim.
- Local paths may be stored in provider configuration but never in transcript
  provenance or generic logs.
- Worker failures extend the existing bounded `error_detail` payload with a
  path-safe STT failure code and allowlisted recovery actions for the parent job
  record and Library failure UI; raw native exceptions do not cross that
  boundary.
- Managed acquisition remains the stronger optional path and lands later.
- Existing Parakeet/faster-whisper routing and explicit retry policy are
  unchanged.

## Alternatives considered

| Option | Why not first |
|---|---|
| Managed GGUF store before provider | Delays usable transcription behind copying, descriptors, catalog, and lifecycle UI. |
| Register external paths as artifacts | Adds state without making the bytes immutable or independently verified. |
| Dedicated resident executor before provider | Adds useful reuse and cancellation but delays the first batch transcription even though the existing spawn pool already contains native faults. |
| Native validation in the UI process | Unnecessarily exposes UI stability to untrusted/native parsing. |

## Rollback

Disable direct-path configuration and keep transcribe.cpp unavailable. No
managed artifacts or transcript records require migration because the path is
provider configuration only and provenance excludes it.

## Links

- [Revised TASK-597 design](../../Docs/superpowers/specs/2026-08-01-task-597-local-gguf-import-design.md)
- [ADR-025](025-shared-stt-artifacts-and-runtime-routing.md)
- [transcribe.cpp v0.1.3](https://github.com/handy-computer/transcribe.cpp/releases/tag/v0.1.3)
