# ADR-040: Use explicit local GGUF paths before managed acquisition

Status: Proposed
Date: 2026-08-02
Related Tasks: TASK-597, TASK-604, TASK-1861
Amends: ADR-025

## Decision

The first usable transcribe.cpp release accepts one explicitly configured local
GGUF path. Chatbook validates that path with a bounded, native-runtime-free
admission boundary when selected and before request dispatch.

The first release does not require the file to be copied into the managed model
artifact store. Curated GGUF catalogs, verified downloads, managed local import,
activation, and artifact promotion are deferred until after direct-path
transcription works end to end.

The local path is provider configuration, not transcript provenance. Only the
path persists; source identity is recomputed per admission and participates in
resident-worker reuse/recycle for that run. A compatible replacement at the
same path is accepted as the new current file rather than treated as a
persistent-identity violation.

Direct-local transcript provenance uses `artifact_root = null` and an empty
artifact dependency set. It records provider/model/runtime/precision/device/
language/attempt fields but neither the path nor a fabricated artifact
revision. This explicitly amends ADR-025's immutable-root expectation for the
manual direct-local provider. Exact-byte reproducibility is unavailable until
managed acquisition lands.

The file is never an automatic-routing candidate and is not labeled installed,
curated, or integrity verified. transcribe.cpp remains an exact manual provider
and never a silent fallback.

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
- TASK-604 owns file selection/configuration and actual transcribe.cpp use.
- The provider revalidates before request dispatch but cannot eliminate a race
  between validation and the native runtime reopening the path.
- Missing, symlinked, unreadable, or newly incompatible current files fail
  clearly and require re-selection.
- A different compatible file at the same path is admitted as the current
  model and forces resident-model recycle; no cross-restart expected snapshot
  is persisted.
- Direct-local transcripts have no immutable artifact revision or exact-byte
  reproducibility claim.
- Local paths may be stored in provider configuration but never in transcript
  provenance or generic logs.
- Managed acquisition remains the stronger optional path and lands later.
- Existing Parakeet/faster-whisper routing and explicit retry policy are
  unchanged.

## Alternatives considered

| Option | Why not first |
|---|---|
| Managed GGUF store before provider | Delays usable transcription behind copying, descriptors, catalog, and lifecycle UI. |
| Register external paths as artifacts | Adds state without making the bytes immutable or independently verified. |
| Native validation in the UI process | Unnecessarily exposes UI stability to untrusted/native parsing. |

## Rollback

Disable direct-path configuration and keep transcribe.cpp unavailable. No
managed artifacts or transcript records require migration because the path is
provider configuration only and provenance excludes it.

## Links

- [Revised TASK-597 design](../../Docs/superpowers/specs/2026-08-01-task-597-local-gguf-import-design.md)
- [ADR-025](025-shared-stt-artifacts-and-runtime-routing.md)
- [transcribe.cpp v0.1.3](https://github.com/handy-computer/transcribe.cpp/releases/tag/v0.1.3)
