---
id: TASK-35589
title: Repair Windows prompt fixture source ownership and Notes initial action focus
status: Done
assignee:
  - '@codex'
created_date: '2026-10-19 04:14'
updated_date: '2026-10-19 04:22'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore eight baseline failures exposed by Windows performance verification while retaining production storage admission and the intended first actionable Notes control.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Internal prompt authoring tests use an independent checked config source and all seven previously failing cases pass.
- [x] #2 The mounted Notes add-from-files projection focuses Import once and preserves existing interaction assertions.
- [x] #3 Targeted related tests and static checks pass with no storage checks bypassed.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace the recorded baseline failures and read current ownership/focus lifecycle contracts.
2. Select an independent real config source for the prompt fixture; correct initial Notes focus at its owning lifecycle boundary.
3. Run the original failing nodes plus related prompt and Notes tests, then static checks and self review.
ADR required: no
ADR path: N/A
Reason: routine fixture ownership and initial-focus bug fixes preserving existing boundaries and storage checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Repaired the eight pre-existing failures exposed by Windows performance verification. Tests/Internal_Prompts/conftest.py now selects one independent real config source with the existing select_config_source helper before writes/reloads; no participant registry or storage guard is bypassed. LibraryNotesCanvas completes its existing initial safe-control focus during on_mount, and LibraryNotesAddFromFilesCanvas completes the focus handoff synchronously with screen.set_focus. No new recompose focus trigger was added.

Diagnosis: notes-focus-diagnostic.log recorded the initial scroll-body autofocus and the delayed safe-control callback; the extra deferred app callback could run only after the test began teardown. Completing only the inner callback was insufficient; completing the original initial lifecycle handoff removes both delays. The existing mounted assertion remains unchanged.

Evidence in Docs/Development/evidence/windows-performance-2026-10-18/: compatibility-baseline.log records 1 failure/7 setup errors; compatibility-repair-final.log records 37 passed authoring/resolver/lasting-sync cases with native Windows owner fixture and unchanged storage checks. compatibility-repair-related.log records 96 passed/14 failed across all Internal_Prompts and Notes lasting/review/root cases; all 22 Notes cases pass. compatibility-related-fixture-baseline.log substitutes exact HEAD scratch_config only in a subprocess and reproduces all 14 wider failing nodes (13 setup errors plus 1 parity failure). The repaired fixture unmasks stale imported config consumers in 13; these inherited failures are outside the original 8 scope and remain documented, with no expanded product edits.

Ruff lint passes for all 3 changed files, formatter check passes for the fixture and both changed widget ranges, compileall and diff check pass. Whole widget files carry pre-existing formatting debt; comparing formatter edit contents against HEAD yields the same 120/9 changed lines respectively, with no new debt. Self review confirms the original focus owner and native source checks remain intact. ADR required: no; ADR path N/A, routine boundary-preserving fixes. No new general lesson beyond the existing source-selection and baseline-attribution guidance.
<!-- SECTION:NOTES:END -->
