---
id: TASK-604
title: Add direct-local transcribe.cpp batch STT provider
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
references:
  - backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
  - backlog/decisions/041-direct-local-gguf-before-managed-acquisition.md
documentation:
  - Docs/superpowers/specs/2026-07-23-stt-parakeet-onnx-transcribe-cpp-design.md
  - Docs/superpowers/specs/2026-08-01-task-597-local-gguf-import-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let a user choose an existing local GGUF and complete real Library audio/video batch transcription with the pinned optional transcribe.cpp runtime, without waiting for a managed model store or a new resident executor.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The optional extra pins transcribe.cpp 0.1.3 exactly, imports it only inside the spawn-isolated ingestion worker, and reports final wheel/ABI or runtime unavailability without affecting application startup.
- [ ] #2 Provider settings expose **Choose GGUF…**, validate the selection off the Textual event loop, persist only `[transcription.transcribe_cpp] model_path` through key-only atomic config persistence, and never log or generically render the path value.
- [ ] #3 The real Library audio/video ingest form offers the exact manual `transcribe-cpp` provider, and a submitted job reaches the production spawn worker, parent-side writer, persisted transcript, and normalized provenance rather than stopping at adapter-only wiring.
- [ ] #4 Inside the worker, TASK-597 admission reruns immediately before a single native model load; the worker then reads authoritative capabilities, constructs the exact per-job declaration and sealed registry, and coordinator preflight equality-checks that already-loaded observation before inference. Missing, changed, symlinked, malformed, incompatible, or unsupported files fail clearly with **Choose another GGUF…** recovery.
- [ ] #5 The adapter normalizes required 16 kHz mono audio, obeys the existing one-heavy-job gate, returns precise normalized results with `model_id=local-gguf:<allowlisted-architecture>`, null artifact identity, and no local path, runtime-version schema addition, or raw native exception in provenance, logs, result representations, or generic errors.
- [ ] #6 transcribe.cpp never participates in semantic default routing or silent fallback. Worker failures extend the existing bounded `error_detail` payload with a path-safe STT failure code and only eligible `choose_another_gguf` and `retry_faster_whisper` actions; the parent job record and Library failure UI preserve those actions, including the explicit provenance-linked **Retry with faster-whisper** flow.
- [ ] #7 Focused tests cover picker/config restart, key-only persistence, the complete production Library ingestion path, worker-side revalidation, lazy import, final ABI probe, load-before-seal capability equality, typed failure/action propagation, native path redaction, crash containment, shutdown, and package-resolution/provider smoke for all five released OS/CPU wheel lanes including Linux ABI coverage.
<!-- AC:END -->
