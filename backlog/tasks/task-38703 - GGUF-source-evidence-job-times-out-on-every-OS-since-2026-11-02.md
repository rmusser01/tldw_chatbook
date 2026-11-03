---
id: TASK-38703
title: GGUF source evidence job times out on every OS since 2026-11-02
status: In Progress
created_date: '2026-11-03 13:47'
labels:
- ci
- flaky-test
priority: medium
references:
- backlog/tasks/task-38206 - GGUF-source-mode-recompose-test-times-out-on-Windows-CI-intermittently.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The "TASK-2062.2 GGUF source evidence" workflow was green through 2026-10-31. From the first run on 2026-11-02 (17:41Z) it fails on PR heads based on current dev, on ubuntu, macOS and Windows alike. The keyboard cases `test_supported_width_keyboard_reaches_each_provider_source_and_actions[llamacpp-llama-cpp]` and `[llamafile-llamafile]` hit pytest-timeout's 60 s limit (one run also raised `console_view_worker_owner_changed` after the timeout), and `test_claim_authority_survives_screen_recompose_and_not_window_selection` fails with "screen recompose did not reach deferred mount". The failure is not limited to Windows, so it is not the executor starvation TASK-38206 diagnosed. The job is the only evidence lane for the GGUF source-mode UI, and while it is red it hides real regressions in that UI.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The root cause is identified, with the dev commit (or PR) that introduced it and evidence from bisection.
- [ ] #2 The job's full-app node set, run exactly as the workflow runs it (one process per node, private profile, --timeout=60), passes at least 5 consecutive times locally on macOS, with per-node timings back in the range of the last green CI runs.
- [ ] #3 If the cause is a product bug, a test that fails without the fix and passes with it guards it.
- [ ] #4 No assertion is loosened, no wait is dropped, and no timeout or settle budget is raised.
- [ ] #5 The job passes on ubuntu-latest and macos-latest in CI.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the failing CI logs (runs 41595165317, 41647054961, 41601553525, 41582510679) and a last-green log (41235218213) for per-node timings and timeout stack dumps.
2. Reproduce on macOS with the job's exact node list, one process per node with TLDW_TEST_PRIVATE_PROFILE_NODE set (the workflow's private-profile mechanism, which also avoids the local raw_source_selection_changed setup error), and --timeout=60.
3. Bisect dev first-parent merges between the last green base and the first red base in scratch worktrees, timing the llamacpp keyboard node; then bisect inside the introducing PR.
4. Find what keeps the process busy (stack sampling, call counters in an uncommitted plugin) and fix it at the shared function every caller routes through, test first.
5. Re-run the node set 5+ times locally; run the touched test files, ruff and preflight; record findings in TASK-38206.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
