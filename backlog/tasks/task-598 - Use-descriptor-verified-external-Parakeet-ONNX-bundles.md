---
id: TASK-598
title: Use descriptor-verified external Parakeet ONNX bundles
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-24 01:03'
updated_date: '2026-08-09 15:30'
labels:
  - stt
  - artifacts
  - import
  - onnx
dependencies:
  - TASK-594
  - TASK-596
references:
  - backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
  - backlog/decisions/041-direct-local-gguf-before-managed-acquisition.md
  - backlog/decisions/050-external-parakeet-roots-with-managed-vad.md
documentation:
  - Docs/superpowers/specs/2026-07-23-stt-parakeet-onnx-transcribe-cpp-design.md
  - >-
    Docs/superpowers/specs/2026-08-09-task-598-external-parakeet-bundles-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users select known Parakeet ONNX model directories and transcribe from those directories without copying the model into Chatbook's managed store. Chatbook remains responsible for the verified Silero VAD dependency, while an optional managed-copy action remains available for users who want immutable store ownership.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 First-run setup and Lab Models let a user select and persist an external Parakeet v2 or v3 directory for an exact INT8 or F32 catalog descriptor, while Library retains a job-scoped directory override that does not change global source configuration; using either path does not copy, modify, delete, or label the external directory as managed or installed.
- [ ] #2 Before use or persistent selection, every descriptor-required model and external-data file is contained within the selected directory, is a materialized regular non-symlink file, and matches the exact catalog byte size and SHA-256; unknown, missing, modified, irregular, or changed bundles fail with stable path-safe errors without parsing ONNX graphs in the UI or resident worker.
- [ ] #3 Chatbook reuses or explicitly offers to download only the pinned managed Silero VAD dependency; the user never supplies a VAD path, transcription never initiates a download, and the resident worker holds the exact VAD lease for its full residency interval. Persistent selection commits only after root verification and VAD readiness succeed, so offline operation, failure, or cancellation leaves the prior source unchanged.
- [ ] #4 Resolution is deterministic: an explicit per-job directory wins, then the configured preferred source for the exact descriptor (external or managed), then a matching active managed model only when no preference exists, then the existing verified legacy fallback only when neither exists. A remembered non-preferred external path is not a candidate; an invalid explicit or preferred source fails clearly rather than silently switching providers or model sources, and the legacy singular v2 INT8 path migrates safely.
- [ ] #5 External-source identity participates in resident reuse/recycle without entering generic logs or transcript provenance as a path or fabricated managed root. Results retain provider/model/precision/language/device fields, use a null artifact root, and record the exact managed VAD dependency identity; UI labels external sources as user-owned rather than installed or managed.
- [ ] #6 After validation, an optional managed-copy action copies only the Parakeet root through existing staging, revalidates it, and reuses the managed VAD without writing readiness, changing the active selector, or changing source preference. A valid installed root without readiness is labeled activation-required and can be explicitly activated; activation verifies the complete closure, writes readiness last, switches the selector, and only then prefers managed. Cancellation or failure leaves the external source active and unchanged, and dependency-only rows never offer Activate.
- [ ] #7 Focused tests cover v2/v3, INT8/F32, external data, corruption and containment, cancellable/coalesced verification, batch-lifetime reuse and cleanup for identical per-job overrides, source mutation, configuration precedence and migration, atomic managed-VAD consent/config commit, VAD lease/deletion behavior, mixed-source runtime provenance, first-run/Models/Library wiring, optional-copy rollback and later activation, and one real macOS external-mode smoke. The task remains In Progress until the parent design's Linux x86_64, Linux aarch64, Windows x86_64, macOS arm64, and macOS x86_64 wheel-supported evidence gates pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
<!-- Added after the approved design is committed and reviewed. -->
<!-- SECTION:PLAN:END -->
