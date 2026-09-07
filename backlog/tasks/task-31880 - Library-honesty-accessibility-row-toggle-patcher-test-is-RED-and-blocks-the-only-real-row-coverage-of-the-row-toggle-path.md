---
id: TASK-31880
title: >-
  Library honesty-accessibility row-toggle patcher test is RED and blocks the
  only real-row coverage of the row-toggle path
status: To Do
assignee: []
created_date: '2026-09-06 18:12'
labels:
  - library
  - tests
  - pre-existing-red
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_library_honesty_accessibility.py::test_row_toggle_patcher_rebuilds_marker_label_both_directions fails on every tree measured, in isolation, with a byte-identical assertion. It is the only real-row Pilot test of the row-toggle marker patcher, so while it is red the targeted-sync row-toggle path has no end-to-end coverage: the wave-7 media series' fix for a computed-attribute-name defect in that exact path (canvas_sync.py's f-string-built selection-object name) had to be guarded by a ~15-line screen double instead of by this test. Fix the label assertion (or the truncation behaviour it is asserting about) so the real-row coverage is live again.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Tests/UI/test_library_honesty_accessibility.py::test_row_toggle_patcher_rebuilds_marker_label_both_directions passes
- [ ] #2 The verdict states whether the marker label or the assertion was wrong, with the evidence that decided it
- [ ] #3 The row-toggle marker-label behaviour the test pins is confirmed unchanged for a passing run (the fix does not make the test vacuous)
- [ ] #4 recipe backlog/docs/library-decomposition-recipe.md section 7's entry for this test is removed once it is green
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Filed by the wave-7 (media) Library-decomposition wave close. Failure, reproduced in isolation on BOTH the wave-7 branch and an isolated baseline worktree at 78186d159, byte-identical each time:

    assert '○ Export' == '○ Export selected'

The assertion fires BEFORE the body under test matters, which is why it voids the test's real coverage rather than merely failing it. Provenance: wave-7 task 3 report section 9.2 (.superpowers/sdd/2026-09-06-library-decomposition-wave7-media/task-3-report.md); the fifth-census-spelling find it blocks is recipe section 3 and section 22. Wave-7 task 3 EDITED this file (the duck-typed app stand-in gained a _media_state) and the failure is unchanged by that edit.

## Id provenance

The CLI auto-assigned TASK-31821 at filing. A sweep of every remote ref
(`git for-each-ref refs/remotes/` + `git ls-tree -r <ref> backlog/`, numeric
sort) shows 31821 is ALREADY taken TWICE on remotes
("Close-remaining-inventory-UI-fixture-owned-database-resources" and
"Route-auth-account-login-bearer-writes-through-the-per-profile-credential-scope"),
and the true repo-wide maximum is 31860 — the exact "never trust the CLI's
auto-assignment" failure `backlog/docs/lessons-backlog-hygiene.md` records.
Renumbered to 31880 (MAX+20) before anything referenced the old id; no
inbound reference existed to move.
<!-- SECTION:NOTES:END -->
