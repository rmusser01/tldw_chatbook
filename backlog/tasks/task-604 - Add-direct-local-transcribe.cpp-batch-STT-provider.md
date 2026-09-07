---
id: TASK-604
title: Add direct-local transcribe.cpp batch STT provider
status: Done
assignee:
  - '@codex'
created_date: '2026-07-24 01:04'
updated_date: '2026-08-02 23:05'
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
- [x] #1 The optional extra pins transcribe.cpp 0.1.3 exactly, imports it only inside the spawn-isolated ingestion worker, and reports final wheel/ABI or runtime unavailability without affecting application startup.
- [x] #2 Provider settings expose **Choose GGUF…**, validate the selection off the Textual event loop, persist only `[transcription.transcribe_cpp] model_path` through key-only atomic config persistence, and never log or generically render the path value.
- [x] #3 The real Library audio/video ingest form offers the exact manual `transcribe-cpp` provider, and a submitted job reaches the production spawn worker, parent-side writer, persisted transcript, and normalized provenance rather than stopping at adapter-only wiring.
- [x] #4 Inside the worker, TASK-597 admission reruns immediately before a single native model load; the worker then reads authoritative capabilities, constructs the exact per-job declaration and sealed registry, and coordinator preflight equality-checks that already-loaded observation before inference. Missing, changed, symlinked, malformed, incompatible, or unsupported files fail clearly with **Choose another GGUF…** recovery.
- [x] #5 The adapter normalizes required 16 kHz mono audio, obeys the existing one-heavy-job gate, returns precise normalized results with `model_id=local-gguf:<allowlisted-architecture>`, null artifact identity, and no local path, runtime-version schema addition, or raw native exception in provenance, logs, result representations, or generic errors.
- [x] #6 transcribe.cpp never participates in semantic default routing or silent fallback. Worker failures extend the existing bounded `error_detail` payload with a path-safe STT failure code and only eligible `choose_another_gguf` and `retry_faster_whisper` actions; the parent job record and Library failure UI preserve those actions, including the explicit provenance-linked **Retry with faster-whisper** flow.
- [x] #7 Focused tests cover picker/config restart, key-only persistence, the complete production Library ingestion path, worker-side revalidation, lazy import, final ABI probe, load-before-seal capability equality, typed failure/action propagation, native path redaction, crash containment, shutdown, and package-resolution/provider smoke for all five released OS/CPU wheel lanes including Linux ABI coverage.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add the exact optional pin and a lazy single-job transcribe.cpp adapter; revalidate before one native load, derive authoritative capabilities, seal the per-job registry, and normalize output through the existing coordinator.
2. Carry the direct-local request, normalized provenance, and a bounded path-safe failure envelope through the existing spawn parser and parent writer for audio/video.
3. Add exact manual-only batch routing plus an explicit faster-whisper retry override that preserves existing job/attempt lineage.
4. Add the Library and first-run Choose GGUF picker, off-loop admission, dedicated key-only config persistence, and the two bounded recovery actions.
5. Run only focused TASK-604 tests/static checks, self-review the branch diff, and document completion.

ADR required: no
ADR path: backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md and backlog/decisions/041-direct-local-gguf-before-managed-acquisition.md
Reason: the accepted ADRs already govern the provider/runtime boundary, direct-local configuration, worker revalidation, provenance, and explicit no-managed-store/no-resident-executor scope.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the direct-local transcribe.cpp Library batch provider without adding managed acquisition or a resident executor. The optional extra pins transcribe-cpp 0.1.3; worker execution revalidates the selected GGUF, lazily imports and ABI-checks the runtime, loads once, derives exact capabilities, routes through the sealed STT coordinator, persists normalized transcript/provenance, and closes the model. Library and first-run surfaces now provide a path-private GGUF picker, and failures retain only allowlisted Choose another GGUF / Retry with faster-whisper actions with retry lineage.

Review hardening added Vulkan provenance for the default Linux/Windows backend, CPU-accelerator aliases, token-to-word timestamp capability normalization, admission-before-runtime failure ordering, missing-config recovery, and traceback-level native path redaction. Existing ADR-025 and ADR-041 remain authoritative; no new ADR was required.

Focused verification: 307 TASK-604/shared-contract cases passed. Ruff passed for modified scope (legacy FirstRun E402/F401 and two unrelated Library-runner F841/F401 findings were isolated and not changed); py_compile, TOML parsing, and git diff --check passed. Published 0.1.3 package metadata was checked for macOS arm64/x86_64, Linux aarch64/x86_64 manylinux, and Windows x86_64 wheels. Windows/Linux native execution was unavailable locally, so that platform gate remains preserved and is not claimed as executed.

Post-review remediation: rebased cleanly onto origin/dev at c53880843; aligned inference_seconds and total_seconds to the same wall-clock basis, completed Google-style public API docstrings, and documented why the worker-local native import intentionally bypasses optional_deps so ABI/backend failures remain sanitized. Re-ran the focused TASK-604 suite after the rebase: 307 passed; scoped Ruff, py_compile, TOML parsing, and diff checks passed.
<!-- SECTION:NOTES:END -->
