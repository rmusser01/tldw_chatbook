---
id: TASK-601
title: Add generation-fenced local STT executor
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-24 01:04'
updated_date: '2026-08-12 06:03'
labels:
  - stt
  - processes
  - ingestion
dependencies:
  - TASK-505
  - TASK-594
  - TASK-599
references:
  - backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
  - backlog/decisions/041-direct-local-gguf-before-managed-acquisition.md
documentation:
  - Docs/superpowers/specs/2026-07-23-stt-parakeet-onnx-transcribe-cpp-design.md
  - Docs/superpowers/specs/2026-08-02-task-601-local-stt-executor-design.md
  - >-
    Docs/superpowers/specs/2026-08-11-task-601-platform-process-tree-evidence-design.md
  - Docs/superpowers/plans/2026-08-02-task-601-local-stt-executor.md
  - Docs/superpowers/plans/2026-08-11-task-601-platform-process-tree-evidence.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create one app-owned heavy-media process boundary that gives batch transcription predictable model residency, artifact lease lifetime, cancellation, crash isolation, and writer safety.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 LocalSTTExecutor owns one spawn-context heavy worker, and neither parse workers nor TranscriptionService instances create private heavy processes.
- [x] #2 The worker holds at most one model identity including provider, model, root revision, dependency-closure fingerprint, precision, and device, reusing identical work and recycling on identity change or bounded lifetime.
- [x] #3 The worker owns root and loaded-dependency leases for the full resident-model lifetime, including idle reuse, and releases them only on close or process exit.
- [x] #4 Every request, progress event, result, and error carries attempt and executor-generation identity; detached-generation callbacks cannot reach the single-writer stage.
- [x] #5 Cooperative cancellation and force stop produce exactly one terminal state, recycle only the heavy pool, and leave light parse workers unaffected.
- [ ] #6 FFmpeg and other preparation subprocesses are owned and terminated as a platform process tree before temporary cleanup on Windows, macOS, and Linux.
- [x] #7 Process tests cover same-model reuse, identity recycle, idle leases, crash release, stale callbacks, child cleanup, CPU retry in a fresh worker, and shutdown.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Generalize the two native process-tree descendant contracts across Linux, Windows, and macOS with non-destructive platform liveness/finalization, and strengthen the production controller test to prove tree termination completes before scratch removal.
2. Add a bounded standard-library JUnit normalizer and strict same-commit/same-run aggregate whose failure documents remain red and path-private.
3. Add a label/manual three-runner GitHub Actions workflow for the exact TASK-601 nodes, with explicit Bash semantics and no model/runtime downloads or general-CI dependency.
4. Rebase before evidence, freeze the executable commit, collect all three native artifacts from one green workflow run, validate and document the aggregate, then close AC6 and TASK-601 through the Backlog CLI.
5. Run final correctness and Ponytail review; any executable correction invalidates prior evidence and requires a fresh three-platform run.

ADR required: no.
ADR paths: backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md and backlog/decisions/041-direct-local-gguf-before-managed-acquisition.md.
Reason: ADR-025 already fixes platform process-tree ownership and cleanup ordering; ADR-041 leaves that boundary unchanged. This remaining work supplies native release evidence only.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented one lazy app-owned spawn executor for Parakeet ONNX and transcribe.cpp Library batch work. The worker keeps one exact model identity resident, owns managed-artifact leases for that residency, validates direct-local snapshots, and returns generation- and attempt-fenced events to the existing parent writer. Cooperative cancellation, force stop, crash quarantine, bounded lifetime recycling, and POSIX/Windows process-tree containment are implemented without replacing the general parse pool.

The transcribe.cpp path now reuses one loaded GGUF runtime, recognizes only the pinned binding's typed and unambiguous accelerator-initialization failures for a single fresh-worker CPU retry, and persists a `device_fallback_to_cpu` warning with truthful requested/effective-device provenance. Terminal callbacks may adopt the initial positive generation before the asynchronous submitted callback, while established generation fencing remains exact.

Verification stayed scoped to TASK-601. The focused implementation gate recorded 325 passing tests before final review remediation; the final changed-path gate passed 11/11 tests plus Ruff and `git diff --check`. Native macOS containment/process evidence passed 10/10 checks and a process-table check found no surviving local-STT/decoder workers. Final code review found no Critical or Important issues. Windows and Linux hosts were unavailable, so acceptance criterion #6 remains open and this task intentionally remains In Progress.

After rebasing onto current `dev`, the TASK-601-focused STT, Library, ingestion, Parakeet, and UI gate passed 943 tests. An isolated current-`dev` control run reproduced one transcription-facade dependency failure and three under-initialized Library test-helper failures exactly, so those upstream baseline failures were excluded rather than changing unrelated production code. The rebase also replaced an in-process `importlib.reload()` import-boundary test with a subprocess check after the reload split IPC dataclass identity and caused spawned worker bootstrap rejection.

PR review remediation centralized explicit Parakeet directory validation, hardened managed GGUF paths against Windows/UNC and symlink escapes, completed the public snapshot docstrings and import grouping, added callback context to marshal failures, and coordinated parse-pool plus STT-executor shutdown through one background thread. The final focused gate passed 951 tests. A recycle test that intermittently failed during that gate exposed concurrent attempts by the reader and controller to reap one spawned process; generation ownership now decides the sole reaper before `join()`. The deterministic ownership regression and crash paths passed together, followed by 20 consecutive bounded-lifetime recycle passes.

ADR required: no. ADR path: `backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md` and `backlog/decisions/041-direct-local-gguf-before-managed-acquisition.md`. Those accepted decisions already govern the runtime boundary, lease ownership, retry policy, and direct-local GGUF behavior.
<!-- SECTION:NOTES:END -->
