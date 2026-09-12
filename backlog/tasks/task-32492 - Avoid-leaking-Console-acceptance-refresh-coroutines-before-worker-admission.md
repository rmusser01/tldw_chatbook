---
id: TASK-32492
title: Avoid leaking Console acceptance refresh coroutines before worker admission
status: Done
assignee:
  - '@codex'
created_date: '2026-09-11 02:46'
updated_date: '2026-09-11 02:53'
labels:
  - console
  - testing
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ensure accepted Console submissions do not leave an unawaited UI refresh coroutine when Textual cannot start the refresh worker, while retaining normal composer clearing and transcript refresh.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Accepted submission preserves its result and emits no leaked-coroutine warning when no UI worker can start.
- [x] #2 Mounted Console acceptance still clears the composer and refreshes the accepted message.
- [x] #3 The regression fails on coroutine leaks and targeted send/runtime tests and scoped static checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: Routine lifecycle correction at the existing Textual worker call; it preserves interfaces, scheduling policy and UI behavior.

1. Reproduce the existing runtime-gate test with RuntimeWarning and PytestUnraisableExceptionWarning promoted to failures; preserve trace proving eager coroutine creation at submission acceptance.
2. Add those strict warning filters to the existing regression and run it red before the production change.
3. Pass the bound async refresh callable to run_worker so Textual creates the coroutine only when executing admitted work. Keep group, nonexclusive behavior, best-effort errors and composer handling unchanged.
4. Verify the strict regression, affected runtime-gate/agent-swap tests and mounted acceptance/early-echo behavior; compare Ruff to before-images and preserve the existing screen-size ceiling.
5. Review the small before-image diff, record the actual cause and evidence in the orchestration ledger/lesson, and complete task criteria via CLI. No full suite, network, staging or commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed the submission-acceptance coroutine leak by passing the bound async refresh function to Textual instead of constructing a coroutine before worker admission. The real unmounted-screen runtime-gate regression now promotes RuntimeWarning and PytestUnraisableExceptionWarning to failures while preserving its accepted-send/legacy-response assertions. Worker group, nonexclusive scheduling, best-effort error behavior and composer handling remain unchanged.

Evidence: allocation tracing reproduced the leak at the acceptance hook; the permanent strict regression failed before the one-line production change and passed afterward. The affected runtime/composer/next-draft group passed 47 tests, and two separate mounted tests passed for acceptance-time echo with polling disabled/provider output held and safe post-teardown sync. The isolated regression overlaps those 47. No unawaited warning appears in the passing runs. Independent review approved all three criteria with zero findings.

Scoped static comparison found zero introduced Ruff diagnostics (ChatScreen 150→150; swap test 10→10) and no formatter differences overlapping the edits. Diff whitespace checks pass. Screen size remains 17,656 lines/594 methods against the unchanged 17,570/591 ceilings; this routine fix does not expand or complete TASK-3070's separate decomposition scope.

ADR required: no; ADR path: N/A. The existing Textual API and UI scheduling contract are unchanged. Updated the consolidated orchestration ledger, prior messaging-warning disposition and lessons-textual.md with the actual incident. Full commands, before-images, RED/GREEN logs, static comparison and independent review: .superpowers/orchestration-followups-2026-09-10/. Existing Requests/pytest cleanup noise remains. A fresh standalone spawn Lock still fails inside SemLock with ENOSPC, so the six earlier multiprocessing cases remain unverified. No full suite, provider network call, host resource cleanup, staging or commit.
<!-- SECTION:NOTES:END -->
