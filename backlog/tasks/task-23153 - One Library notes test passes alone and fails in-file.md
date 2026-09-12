---
id: TASK-23153
title: One Library notes test passes alone and fails in-file
status: Done
assignee:
  - '@codex'
created_date: '2026-08-28'
updated_date: '2026-08-29'
labels:
  - tests
  - library
priority: low
dependencies: []
---

## Description

The originally reported `test_library_note_failed_discard_clears_shortcut_lock_status` contract was
removed when Notes-specific Ctrl+S shortcut-lock status was intentionally eliminated. Reconcile the
stale task against the replacement destructive-admission regression and current verification evidence.

## Acceptance Criteria

- [x] Current dev no longer exposes the removed Notes shortcut-lock status contract, and the original named test is absent for that reason
- [x] The replacement destructive-admission regression passes standalone and in the recorded whole-file Library shell run
- [x] Resolution is traced to removal of the obsolete source contract; no cleanup-only victim patch or production change is added

## Implementation Plan

1. Reproduce the original test selection against current dev and trace the test's removal through history.
2. Identify the current regression that preserves the still-supported destructive-admission behavior.
3. Run the replacement regression standalone and compare it with the latest whole-file Library shell evidence.
4. Close the stale task without changing application behavior.

ADR required: no
ADR path: N/A
Reason: This is test-record reconciliation after an intentional removal of an obsolete keyboard/status contract; no architecture changes.

## Evidence

- The original node selection now returns “not found” because the test no longer exists.
- Git history traces the replacement to `7cf89de6c042e3309dbd20bdd2eae59b5160e8fb`, which removed the prohibited Notes Ctrl+S shortcut/status contract and renamed the surviving regression to `test_library_note_failed_discard_releases_destructive_admission`.
- The replacement test passes standalone (`1 passed`); TASK-24195 records the complete `Tests/UI/test_library_shell.py` run passing (`823 passed`).

## Definition of Done

- [x] The stale failure is reconciled against current behavior and source history
- [x] Current supported behavior has targeted and whole-file verification evidence
- [x] No unrelated production or test behavior is changed
- [x] Implementation notes and ADR disposition are recorded

## Implementation Notes

- The suspected cross-test pollution was tied to a test for `_library_note_shortcut_status`, a Notes-specific Ctrl+S refusal/status mechanism removed by TASK-22513's keybinding correction. Commit `7cf89de6c042e3309dbd20bdd2eae59b5160e8fb` narrowed the regression to the supported invariant: a failed discard releases destructive admission.
- Ran `Tests/UI/test_library_shell.py::test_library_note_failed_discard_releases_destructive_admission` on current dev: 1 passed. TASK-24195 supplies the later whole-file result: 823 passed.
- Because the leaking source contract itself was intentionally removed, adding victim cleanup would restore obsolete behavior and hide the actual resolution. No production or test-code change was needed.
- No full application suite was run; verification remained targeted to the affected Library Notes behavior.
