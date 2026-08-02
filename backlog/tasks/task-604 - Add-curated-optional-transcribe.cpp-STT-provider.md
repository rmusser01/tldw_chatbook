---
id: TASK-604
title: Add optional transcribe.cpp provider for a selected local GGUF
status: To Do
assignee: []
created_date: '2026-07-24 01:04'
labels:
  - stt
  - gguf
  - native
dependencies:
  - TASK-597
  - TASK-599
  - TASK-600
  - TASK-601
references:
  - backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
  - backlog/decisions/040-direct-local-gguf-before-managed-acquisition.md
documentation:
  - Docs/superpowers/specs/2026-07-23-stt-parakeet-onnx-transcribe-cpp-design.md
  - Docs/superpowers/specs/2026-08-01-task-597-local-gguf-import-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let a user choose and configure an existing local GGUF, then transcribe with the pinned optional transcribe.cpp runtime while keeping Parakeet ONNX and faster-whisper authoritative for semantic default routing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The optional extra pins transcribe.cpp 0.1.3 exactly, imports it lazily, and reports unavailable or incompatible runtime states without affecting application startup.
- [ ] #2 Provider settings expose **Choose GGUF…**, persist only `[transcription.transcribe_cpp] model_path`, and rerun TASK-597 admission before request dispatch; a changed compatible per-admission source snapshot recycles the resident model.
- [ ] #3 Missing, unreadable, symlinked, malformed, or incompatible current files fail clearly with **Choose another GGUF…** recovery and no application-startup failure; a compatible replacement at the same path becomes the current model.
- [ ] #4 The adapter loads the current validated local path only in the app-owned heavy worker, normalizes required 16 kHz mono audio, allows one active inference, and records precise normalized results with null artifact identity and no local path in provenance.
- [ ] #5 transcribe.cpp remains an exact manual provider, never participates in semantic default routing or silent fallback, and unsupported family capabilities fail before enqueue; eligible failures may offer explicit faster-whisper retry.
- [ ] #6 File/configuration round trips, lazy import, supported-family smoke, cancellation, crash containment, path revalidation, provenance exclusion, and all five wheel-platform tests pass.
<!-- AC:END -->
