---
id: TASK-598
title: Use descriptor-verified external Parakeet ONNX bundles
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-24 01:03'
updated_date: '2026-08-09 15:13'
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
  - Docs/superpowers/specs/2026-08-09-task-598-external-parakeet-bundles-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users select known Parakeet ONNX model directories and transcribe from those directories without copying the model into Chatbook's managed store. Chatbook remains responsible for the verified Silero VAD dependency, while an optional managed-copy action remains available for users who want immutable store ownership.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 First-run setup and Lab Models let a user select and persist an external Parakeet v2 or v3 directory for an exact INT8 or F32 catalog descriptor, while Library retains a job-scoped directory override that does not change global source configuration; using either path does not copy, modify, delete, or label the external directory as managed or installed.
- [ ] #2 Before activation, every descriptor-required model and external-data file is contained within the selected directory, is a materialized regular non-symlink file, and matches the exact catalog byte size and SHA-256; unknown, missing, modified, irregular, or changed bundles fail with stable path-safe errors without parsing ONNX graphs in the UI or resident worker.
- [ ] #3 Chatbook reuses or explicitly offers to download only the pinned managed Silero VAD dependency; the user never supplies a VAD path, transcription never initiates a download, and the resident worker holds the exact VAD lease for its full residency interval.
- [ ] #4 Resolution is deterministic: an explicit per-job directory wins, then a matching persistent external source, then a matching managed model, then the existing verified legacy fallback. An invalid explicit or persistent external source fails clearly rather than silently switching providers or model sources, and the legacy singular v2 INT8 path migrates safely.
- [ ] #5 External-source identity participates in resident reuse/recycle without entering generic logs or transcript provenance as a path or fabricated managed root. Results retain provider/model/precision/language/device fields, use a null artifact root, and record the exact managed VAD dependency identity; UI labels external sources as user-owned rather than installed or managed.
- [ ] #6 After validation, an optional managed-copy action copies only the Parakeet root through existing staging, revalidates it, reuses the managed VAD, and writes root readiness last. Cancellation or failure leaves the external source active and unchanged, and the managed copy does not become active without explicit activation.
- [ ] #7 Focused tests cover v2/v3, INT8/F32, external data, corruption and containment, cancellable/coalesced verification, source mutation, configuration precedence and migration, managed-VAD consent/lease/deletion behavior, mixed-source runtime provenance, first-run/Models/Library wiring, optional-copy rollback, and one real macOS external-mode smoke. The task remains In Progress until required Windows and Linux platform evidence is available.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
<!-- Added after the approved design is committed and reviewed. -->
<!-- SECTION:PLAN:END -->
