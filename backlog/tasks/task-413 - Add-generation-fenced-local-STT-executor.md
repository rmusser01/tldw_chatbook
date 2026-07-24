---
id: TASK-413
title: Add generation-fenced local STT executor
status: To Do
assignee: []
created_date: '2026-07-24 01:04'
labels:
  - stt
  - processes
  - ingestion
dependencies:
  - TASK-404
  - TASK-406
  - TASK-411
references:
  - backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
documentation:
  - Docs/superpowers/specs/2026-07-23-stt-parakeet-onnx-transcribe-cpp-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create one app-owned heavy-media process boundary that gives batch transcription predictable model residency, artifact lease lifetime, cancellation, crash isolation, and writer safety.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 LocalSTTExecutor owns one spawn-context heavy worker, and neither parse workers nor TranscriptionService instances create private heavy processes.
- [ ] #2 The worker holds at most one model identity including provider, model, root revision, dependency-closure fingerprint, precision, and device, reusing identical work and recycling on identity change or bounded lifetime.
- [ ] #3 The worker owns root and loaded-dependency leases for the full resident-model lifetime, including idle reuse, and releases them only on close or process exit.
- [ ] #4 Every request, progress event, result, and error carries attempt and executor-generation identity; detached-generation callbacks cannot reach the single-writer stage.
- [ ] #5 Cooperative cancellation and force stop produce exactly one terminal state, recycle only the heavy pool, and leave light parse workers unaffected.
- [ ] #6 FFmpeg and other preparation subprocesses are owned and terminated as a platform process tree before temporary cleanup on Windows, macOS, and Linux.
- [ ] #7 Process tests cover same-model reuse, identity recycle, idle leases, crash release, stale callbacks, child cleanup, CPU retry in a fresh worker, and shutdown.
<!-- AC:END -->
