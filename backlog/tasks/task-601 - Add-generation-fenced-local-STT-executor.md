---
id: TASK-601
title: Add generation-fenced local STT executor
status: Done
assignee:
  - '@codex'
created_date: '2026-07-24 01:04'
updated_date: '2026-08-12 08:52'
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
  - Docs/STT_Evaluation/task-601/README.md
  - Docs/STT_Evaluation/task-601/platform-evidence.json
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
- [x] #6 FFmpeg and other preparation subprocesses are owned and terminated as a platform process tree before temporary cleanup on Windows, macOS, and Linux.
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

Native platform evidence attempt 1 remained red and did not close acceptance criterion 6. Workflow run https://github.com/rmusser01/tldw_chatbook/actions/runs/31575959082 tested executable commit 48c4ccadb9ef93f64ddfffd9954aede9526ea48e: Linux x86_64 and macOS x86_64 passed; Windows x86_64 produced bounded test_execution failure evidence. All three required evidence nodes passed on Windows, but selected unit Tests/STT/test_executor_process_tree.py::test_unproven_tree_death_quarantines_containment failed before its assertion because the test monkeypatch required os.killpg to pre-exist. Windows does not expose os.killpg. This is a native test-portability defect, not a containment production failure. No aggregate was created. Remediation is limited to a RED-backed portable monkeypatch and requires a fresh full three-platform run on a new executable commit.

Native platform evidence attempt 2 also remained red and did not close acceptance criterion 6. New workflow_dispatch run https://github.com/rmusser01/tldw_chatbook/actions/runs/31576646463 tested reviewed executable commit 83c68c30d82fe04b53db02612ff358fb2fb6a0ec: Linux x86_64 and macOS x86_64 passed; Windows x86_64 produced bounded test_execution failure evidence. All three required nodes again passed. The prior os.killpg monkeypatch repair allowed the same selected POSIX-emulation unit to reach its next Windows-unavailable POSIX symbol: signal.SIGKILL. No aggregate was created. Remediation remains test-only: install the simulated SIGKILL constant without skipping coverage, review it, and run a new full three-platform workflow on a new executable commit.

Native platform evidence attempt 3 passed and closes acceptance criterion 6. Workflow run https://github.com/rmusser01/tldw_chatbook/actions/runs/31577352552 tested executable commit 5c6a446c8d050587f141561319e58e1ce72c528d: Linux x86_64 on ubuntu-24.04, Windows x86_64 on windows-2022, and macOS x86_64 on macos-15-intel all passed the exact three required process-tree and scratch-ordering nodes. All three downloaded JSON records passed individual validation, matched the tested commit and workflow run, and produced the checked-in aggregate through the repository script; aggregate validation passed. The fresh local gate passed 98 tests with one intentional local Windows-only skip in 1.37 seconds; Ruff check, Ruff format check for all five scoped Python files, py_compile for the evidence script, and git diff --check origin/dev...HEAD all passed. Attempts 1 and 2 exposed only Windows portability gaps in a selected POSIX-emulation unit test; both test-only remediations were reviewed, no production fix was needed, and every native lane was rerun after each executable change. Evidence and complete rerun history are documented in Docs/STT_Evaluation/task-601/README.md and Docs/STT_Evaluation/task-601/platform-evidence.json. ADR required: no; ADR-025 and ADR-041 continue to govern the unchanged boundary.

Native three-platform evidence and its evidence-related documentation and validation requirements are complete, and acceptance criterion 6 remains checked. TASK-601 stays In Progress pending Task 5 final correctness and Ponytail review; final task completion and the full Definition of Done are not yet claimed.

Task 5 final review completed. Independent correctness review approved the complete origin/dev...HEAD range with no Critical, Important, or Minor findings and independently confirmed the checked-in aggregate reproduces the successful three-platform artifacts for frozen executable commit 5c6a446c8d050587f141561319e58e1ce72c528d. Final Ponytail review found no blocking complexity; its optional deletion of redundant CLI test coverage is intentionally deferred because changing executable/test content after native evidence would violate the frozen-evidence boundary. A fresh final local gate passed 98 tests with one intentional host-only skip; aggregate validation, Ruff check and format check, py_compile, and origin/dev...HEAD diff check passed. All acceptance criteria and the full applicable Definition of Done are complete.

PR #1561 review after the initial evidence closeout identified a real Windows failure-finalizer hazard: the test reopened captured raw PIDs for cleanup, permitting PID-reuse termination of an unrelated process, and its helper conflated WAIT_TIMEOUT with Win32 failure. TASK-601 and AC6 are reopened before remediation. The fix is test-only and will use only the owned multiprocessing.Process API; because selected native-test content changes, a brand-new three-platform run and replacement aggregate are required before completion.

PR review remediation is complete. Commit 0e34f7462e4dbd5a724fbe9a6f93ded959623d3e removed raw-PID Windows cleanup, uses only the owned multiprocessing.Process and kill-on-close Job Object handles, added a RED/GREEN regression, and completed Google-style documentation for public evidence-script functions. Review comments about central app path validation and pytest-function docstrings were resolved without code changes: this is an intentionally standard-library-only trusted CI/operator tool with runner-temporary paths, and pytest tests are not package public APIs. Replacement workflow run https://github.com/rmusser01/tldw_chatbook/actions/runs/31580179256 passed Linux x86_64, Windows x86_64, and macOS x86_64 against the exact remediation SHA; all three downloaded records validated and regenerated the checked-in aggregate through the repository script. AC6 and TASK-601 are complete again.
<!-- SECTION:NOTES:END -->
