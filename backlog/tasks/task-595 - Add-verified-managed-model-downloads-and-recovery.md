---
id: TASK-595
title: Add verified managed model downloads and recovery
status: Done
assignee: []
created_date: '2026-07-24 01:02'
updated_date: '2026-08-01 00:59'
labels:
  - stt
  - artifacts
  - downloads
dependencies:
  - TASK-594
references:
  - backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
documentation:
  - Docs/superpowers/specs/2026-07-23-stt-parakeet-onnx-transcribe-cpp-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add consent-driven managed acquisition over the shared artifact lifecycle so curated GGUF and ONNX bundles can be resumed, verified, activated, and recovered safely.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Preflight resolves the complete dependency closure and reports source, license, precision, total bytes, destination, staging overhead, retained versions, and required free space before any transfer.
- [x] #2 Downloads use per-artifact staging, bounded HTTP behavior, resume validation, stable-order installation locks, and final per-file size and SHA-256 verification.
- [x] #3 The root readiness record is written last; cancellation, hash failure, network failure, and process interruption leave the prior active version usable and incomplete staging non-loadable.
- [x] #4 Provider workers cannot initiate downloads, and every first-use acquisition requires explicit caller confirmation before enqueue.
- [x] #5 Authenticated repositories use supported credential boundaries without persisting or logging secrets.
- [x] #6 Local fixture integration tests cover resume, changed validators, corrupt payloads, concurrent installers, insufficient space, crash recovery, and staging cleanup containment.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
ArtifactAcquisitionService (async preflight/consent/provision) + stream_fetch over the sealed 594 core; consume_source install + orphans-only staging GC as the only core changes; session-lease serialization with busy semantics; resume/pre-verify/crash-recovery/credential fixtures per AC #6. Spec: Docs/superpowers/specs/2026-07-30-managed-model-acquisition-design.md.

Delivered across 10 sequential tasks (commits tagged `TASK-595` on
`feat/managed-model-acquisition`), composed entirely over the sealed 594
core (`ModelArtifactService`) via a new `ArtifactAcquisitionService`:

- **Core additions (exactly two, by design):** `install(..., consume_source=True)`
  (in-root move with EXDEV copy fallback) and `reconcile()`'s
  orphans-only managed-staging GC (a directory survives iff its
  `fetch-state.json` sidecar parses as JSON; anything else -- missing,
  corrupt, wrong depth -- is swept). Both are exercised directly by this
  task's real-subprocess crash tests.
- **`acquisition.py`:** async `preflight()` (closure walk + staged-byte
  credit + space math + gating probe) → `PreflightReport.grant()` →
  `provision(root, consent, catalog)` (note: explicit `root` param is a
  reviewed deviation from an earlier consent-only signature) → durable
  per-file fetch with Range/validator resume (`fetch.py`'s `stream_fetch`)
  → pre-verify with one automatic refetch on hash mismatch →
  `consume_source` install → activate-last. A typed, never-a-raw-trap
  error family (`AcquisitionError` and subclasses) covers every failure
  mode from catalog cycles to insufficient space to gated repositories.
- **Concurrency:** an exclusive, non-blocking `ACQUISITION_SESSION_LEASE_KEY`
  OS lease (busy → `AcquisitionBusyError`, not a hang) plus an in-process
  `asyncio.Lock` so same-process concurrent callers queue instead of
  racing that lease.
- **Credentials (Task 9):** `CredentialResolver` protocol +
  `EnvConfigCredentialResolver` (env, then `[API] huggingface_api_key`
  config; keyring deliberately deferred). A dedicated subprocess
  import-boundary test proves the synchronous STT/transcription worker
  surface never imports `.acquisition`/`.fetch` -- provider workers
  structurally cannot initiate a download.
- **Crash recovery (Task 10, this close-out):** three real-subprocess,
  SIGKILL-based tests in `Tests/Model_Artifacts/test_provision_crash_recovery.py`
  (harness: `Tests/Model_Artifacts/provision_processes.py`, mirroring
  `lease_processes.py`'s style) prove, under an actual OS-level kill
  rather than asyncio cancellation: (1) a genuinely partial-but-durable
  fetch checkpoint survives `reconcile()`, the session lease is released
  by the kernel, and a fresh `provision()` resumes via an observed
  `Range` request and completes; (2) killing after both artifacts in a
  closure are installed but before `activate()` lets a fresh `provision()`
  activate with zero fixture requests; (3) `reconcile()` after a real
  crash removes only the orphan it names, leaving a crash-surviving valid
  entry and unrelated sibling files completely untouched. Both guards
  were mutation-tested (a targeted one-line break in the resume path and
  in the orphan classifier each turned the corresponding assertion red).
- **Public exports:** `Model_Artifacts/__init__.py.__all__` now includes
  the full async acquisition surface. `.acquisition`/`.fetch` (both
  `import httpx`) are resolved lazily via a module-level `__getattr__`
  (PEP 562), not eagerly at package-import time -- Task 9's boundary
  invariant ("plain `import tldw_chatbook.Model_Artifacts` never loads
  httpx") stays intact; `from tldw_chatbook.Model_Artifacts import
  ArtifactAcquisitionService` (and every other new name) still resolves
  correctly on first access.

Full gate: `Tests/Model_Artifacts/ Tests/STT/test_boundaries.py` --
358 passed (355 pre-existing + 3 new), stable across repeated runs.
<!-- SECTION:NOTES:END -->
