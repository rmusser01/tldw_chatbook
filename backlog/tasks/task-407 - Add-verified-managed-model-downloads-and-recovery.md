---
id: TASK-407
title: Add verified managed model downloads and recovery
status: To Do
assignee: []
created_date: '2026-07-24 01:02'
labels:
  - stt
  - artifacts
  - downloads
dependencies:
  - TASK-406
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
- [ ] #1 Preflight resolves the complete dependency closure and reports source, license, precision, total bytes, destination, staging overhead, retained versions, and required free space before any transfer.
- [ ] #2 Downloads use per-artifact staging, bounded HTTP behavior, resume validation, stable-order installation locks, and final per-file size and SHA-256 verification.
- [ ] #3 The root readiness record is written last; cancellation, hash failure, network failure, and process interruption leave the prior active version usable and incomplete staging non-loadable.
- [ ] #4 Provider workers cannot initiate downloads, and every first-use acquisition requires explicit caller confirmation before enqueue.
- [ ] #5 Authenticated repositories use supported credential boundaries without persisting or logging secrets.
- [ ] #6 Local fixture integration tests cover resume, changed validators, corrupt payloads, concurrent installers, insufficient space, crash recovery, and staging cleanup containment.
<!-- AC:END -->
