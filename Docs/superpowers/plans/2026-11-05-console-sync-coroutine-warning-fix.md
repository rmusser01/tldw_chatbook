# Fix the unawaited native Console UI synchronization coroutine

Task: TASK-38614.
Goal: identify and fix the original allocation/scheduling cause while preserving refresh, coalescing, cancellation and teardown semantics.
ADR required: no
ADR path: N/A
Reason: routine fix at an existing Textual boundary; no new owner or service contract.

## Global constraints

Use the same dirty dev worktree. Preserve every unrelated file/index change captured in baseline.json. No branch/worktree, provider/network replay, full suite, reset/stash/clean, push/merge or maintenance. Reuse the installed Textual APIs and existing Console synchronization owners. Do not suppress warnings or weaken cleanup/refresh assertions. Root owns Backlog/plan/status and review. One implementation agent and one independent review of the complete small change, with root-cause evidence gate before behavioral edits. Retain source-qualified adverse and positive receipts.

## Task 1: Diagnose and correct the warning

- [x] Trace allocation/callers and reproduce the actual native warning with origin tracing.
- [x] Gate: controller reads concrete origin/evidence before approving behavioral edits.
- [x] Add deterministic RED regression, then make the smallest shared correction.
- [x] Run targeted native/mounted and affected scheduling tests plus scoped static checks.
- [x] Independent review: spec compliance, code quality and connected caller coverage; fix any findings through scoped rereview.
- [x] Record evidence, preserve unrelated changes, mark task Done and commit exact owned files.
