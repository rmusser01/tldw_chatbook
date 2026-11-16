---
id: TASK-41248
title: Prevent Library Git row layout failures during shutdown
status: Done
created_date: 2026-11-16 03:08
labels:
- bug
- library
- ui
priority: medium
assignee:
- '@codex'
updated_date: 2026-11-16 08:59
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A deferred Session Git row layout callback can run after Textual starts removing the row's children. Closing the Library or replacing Git status rows can then raise NoMatches during shutdown instead of completing quietly.

Reproduction:
1. Open a file-notes workspace in Library, then Review session changes.
2. Let a Git action fail and finish its postflight status refresh, which rebuilds the visible rows.
3. Close the app while a row's deferred layout callback is pending. The intermittent failure appears during shutdown after the action's assertions have passed.
4. Existing reproducer: Tests/UI/test_library_file_notes_git.py::test_unexpected_action_failure_survives_postflight_refresh. Add a controlled partial-unmount regression so verification does not depend on reproducing the timing by chance.

Evidence:
- PR #3495 hosted UI18: 1 failed, 107 passed. The failure is NoMatches for .file-notes-git-row-secondary in _SessionGitListItem._fit_copy, called during run_test.__aexit__: https://github.com/rmusser01/tldw_chatbook/actions/runs/43781885976/job/133022539499
- The exact case passed locally (1 passed in 16.95s), and the unchanged-head retry passed all 108 cases in 527.53s: https://github.com/rmusser01/tldw_chatbook/actions/runs/43781885976/job/133046617185
- Textual removes child widgets separately before detaching their row. Being mounted and having some children does not establish that both row labels still exist.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A pending Session Git row layout callback exits quietly during actual partial row unmount, allowing removal and shutdown to finish.
- [x] #2 Active rows continue to fit their displayed labels, and unexpected live-row lookup failures remain visible.
- [x] #3 A deterministic regression fails on the original code and passes with the fix, and the original postflight-refresh scenario remains green.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no.
ADR path: N/A.
Reason: routine UI lifecycle correction and regression coverage; no storage, Git authority, service, token, layout or dependency contract changes. Existing ADR-035/038/039 remain unchanged.

1. Trace the real Textual callback and child-unmount lifecycle. Add a deterministic mounted regression and live-row controls; preserve the original secondary-label NoMatches failure evidence.
2. Apply a minimal lifetime guard and verify focused scenarios, then register the regression in the existing UI PR gate census and obtain independent review.
3. Integration update: TASK-41218 / PR #3524 landed a guard for the same race while #3573 waited in the queue. Retain that merged guard, preserve every incoming census entry, append the additional regression module and raise the floor to the measured total.
4. Validate both regression sets and the original postflight-refresh/navigation/projection controls on current dev. Recheck the census, scoped lint/format and backlog guards. Obtain fresh independent review of the integrated coverage and document the overlapping fix accurately.
5. Push the rebased PR with an explicit force-with-lease after disabling auto-merge, re-arm it with no unresolved review threads, follow required pull-request CI, and verify the actual merge commit is present on dev.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Verified the shutdown race and added deterministic partial-unmount coverage plus live-row controls. While PR #3573 waited in the merge queue, TASK-41218 / PR #3524 landed a fix for the same race on dev (merge 71e49709a3). The rebased change preserves that merged is_attached/_pruning guard unchanged rather than replacing it with the independently developed is_running guard. No query exception is caught, so broken live rows still report their errors.

The deterministic regression uses real mounted row/Static widgets. A forwarding layout callback and the actual primary label's Unmount handler are held; the test proves the secondary has detached while the primary remains and the row is mounted/attached but no longer running. The callback then resumes, and finally both holds release and physical removal is joined. A separate live control verifies model-to-label updates and an unmasked missing-label error.

Validation:
- RED on unchanged product: 1 failed, 1 passed in 11.42s; exact original NoMatches for .file-notes-git-row-secondary during run_test.__aexit__.
- Initial GREEN: all 10 focused cases passed in 24.73s. Independent read-only review found no issues and independently passed both new cases in 7.73s. The original PR head passed every required CI lane on attempt 1; its UI18 passed 110 cases in 684.93s: https://github.com/rmusser01/tldw_chatbook/actions/runs/43798739806/job/133078405579
- After rebasing onto the overlapping merged fix: all 11 focused cases passed in 34.05s, including both regression sets, the original failed-postflight scenario, hidden/reopen behavior, five note-intent projections and scrollbar copy fitting.
- All 87 UI census-checker tests passed again in 52.42s. The gate now has 2147 entries/floor2147, preserving all 2146 entries from current dev in order; the appended module maps to UI19. The production file is identical to current dev.
- Scoped Ruff lint, formatting, direct census and whitespace checks passed. Existing pytest-asyncio/Pydantic deprecation warnings remain. All 6679 backlog task files pass readability/marker and ID uniqueness guards. Fresh independent integration review approved the final five-file diff with no findings; its separate isolated run passed both added tests in 8.72s. Current-head hosted CI remains tracked on PR #3573.

Changed files relative to current dev: the new Tests/UI/test_library_git_row_shutdown.py module, the UI gate census and floor, this task, and an addition to the existing partial-prune testing-evidence lesson. All incoming TASK-41218 coverage and documentation are preserved.

ADR required: no; routine UI lifecycle regression coverage with no changes to storage, Git authority, dependencies, UI tokens, or service contracts. Existing ADR-035/038/039 remain unchanged. Plan deviation: keep the fix already merged by TASK-41218 and integrate the additional coverage. Required pull-request CI on the rebased head remains the separate integration gate before merge into dev.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
