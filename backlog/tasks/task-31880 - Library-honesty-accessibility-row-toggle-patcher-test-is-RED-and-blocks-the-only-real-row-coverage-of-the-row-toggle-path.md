---
id: TASK-31880
title: >-
  Library honesty-accessibility row-toggle patcher test is RED and blocks the
  only real-row coverage of the row-toggle path
status: Done
assignee: []
created_date: '2026-09-06 18:12'
updated_date: '2026-09-08 23:43'
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
- [x] #1 Tests/UI/test_library_honesty_accessibility.py::test_row_toggle_patcher_rebuilds_marker_label_both_directions passes
- [x] #2 The verdict states whether the marker label or the assertion was wrong, with the evidence that decided it
- [x] #3 The row-toggle marker-label behaviour the test pins is confirmed unchanged for a passing run (the fix does not make the test vacuous)
- [x] #4 recipe backlog/docs/library-decomposition-recipe.md section 7's entry for this test is removed once it is green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce in isolation and read the exact assertion.
2. Read the media bulk-action label construction
   (`library_media_canvas._bulk_action_button`) and the patcher
   (`canvas_sync._patch_library_disabled_marker_label`).
3. Date the divergence with git history (`git log -S` on both the canvas
   and the test) to decide LABEL vs ASSERTION.
4. Probe the live harness to confirm what the button actually carries, and
   whether the in-place path or the recompose fallback runs.
5. Fix whichever side the evidence convicts; keep the pinned behaviour
   intact.
6. Mutation-test the row-toggle marker patcher to prove non-vacuity;
   restore and re-green.
7. Remove the recipe §7 entry, noting the closure; run the file, the
   canvas-sync guards, and `./scripts/preflight.sh`.
<!-- SECTION:PLAN:END -->

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

## Closure (2026-09-08) — verdict, evidence, fix

**VERDICT: the ASSERTION was wrong, not the label.** The fix is test-side
only; no file under `tldw_chatbook/` changed.

**Evidence.** (1) `library_media_canvas.py:554-619` passes the literal base
`"Export"` (also `"Review"`/`"Delete"`) and stashes it as
`_library_disabled_marker_base`, with a docstring naming task-30043 and the
reason: at the items pane's ~40-col width the full labels chopped and a
disabled action "collapsed to a bare '○'". (2) `git show
e34b11b23^:…library_media_canvas.py` has `"Export selected"`; the same file
AT `e34b11b23` (2026-09-03, task-30043) has `"Export"` — and that commit
did not touch this test, whose assertions date from `a1a58308e`
(2026-08-10, task-4023 review M-1). So the test was correct until
2026-09-03 and stale after. (3) A live Pilot probe printed
`canvas.compact = False` at 80x24 with base `'Export'` — media's bulk
labels have no width-responsive spelling at all (unlike notes'), so nothing
truncates at runtime; the width story is why task-30043 chose the short
word.

**A second staleness sat behind the first assertion.** With only the
spelling corrected the test still failed: `_apply_library_row_toggle`'s
media leg calls `screen._library_media_analyze_reason()` (added 2026-09-04
by task-28007) and the duck-typed screen stand-in `_SelectModeApp` never
had it, so the dispatcher's blanket `except Exception` swallowed the
`AttributeError` and rerouted EVERY toggle here onto
`screen.refresh(recompose=True)`. Confirmed by instrumenting
`canvas_sync.logger.debug` in a probe: 2 fallbacks, 2 toggles; the host app
recomposed and the test's button references went stale. So this test had
been silently non-covering since 2026-09-04, on top of being red since
2026-09-03.

**Changes** (`Tests/UI/test_library_honesty_accessibility.py`):
`_SelectModeApp` gains `_library_media_analyze_reason() -> ""` (the same
stub the screen/controller double in `test_library_selection_updates.py`
uses); assertions updated to `○ Export` / `○ Delete` disabled and
`LIBRARY_ACTION_LABEL_PAD + "Export"/"Delete"` enabled (the marker-width
reservation from task-31635/task-31959); a widget-IDENTITY guard added
after each toggle so a future silent fallback fails loudly instead of
degrading; docstring records the verdict.

**Non-vacuity (AC#3).** Mutations in `canvas_sync.py`, each restored after:
dropping `_patch_library_disabled_marker_label(export_button)` → RED
`assert '○ Export' == '  Export'` (the patcher reason); dropping the
delete-button patch → RED `assert '○ Delete' == '  Delete'`; forcing the
fallback (bad analyze selector) → RED on the identity guard. Restored:
green, `git status` clean of product files.

**Runs.** Target test 1 passed in isolation. `test_library_honesty_
accessibility.py` + `test_library_selection_updates.py`: 6 failed / 33
passed after vs 7 failed / 32 passed with the change stashed — the delta is
exactly this test; the other six are pre-existing reds byte-identical on
both sides. `./scripts/preflight.sh`: all checks passed.

**Docs.** Recipe §7's entry removed with a closure note carrying the
verdict (AC#4); §22 lesson 6, §23 and §25's phase-C handoff item 3 stamped
closed. Full write-up: `.superpowers/sdd/phase-c/task-31880-report.md`.
<!-- SECTION:NOTES:END -->
