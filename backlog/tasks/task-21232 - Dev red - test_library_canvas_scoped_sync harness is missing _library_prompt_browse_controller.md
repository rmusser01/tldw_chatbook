---
id: TASK-21232
title: >-
  Dev red - test_library_canvas_scoped_sync harness is missing
  _library_prompt_browse_controller
status: Done
assignee:
  - rmusser01
created_date: '2026-08-23'
labels:
  - bug
  - test-health
  - dev-red
  - library
dependencies: []
priority: high
---

## Description

Source: close-out of the 2026-08-22 holistic performance review burn-down; found pre-existing
during TASK-21101's implementation and re-confirmed at close-out.

`Tests/UI/test_library_canvas_scoped_sync.py` builds its screen double from
`types.SimpleNamespace` and never sets `_library_prompt_browse_controller`, which the
production code under test reads. Re-run on dev `b2b1e2e0d`: **4 failed, 5 passed**, with
`AttributeError: 'types.SimpleNamespace' object has no attribute
'_library_prompt_browse_controller'` raised from the freshness check.

The failure is in the harness, not the subject. Those four tests are not measuring the
canvas-scoped sync behaviour they were written for, so the Library canvas seam — the seam
TASK-21116 is converting more sites onto, and the seam TASK-21242 must repair — is
under-covered while looking covered. That is the worst state for a guard to be in while
another task is actively changing its subject.

## Acceptance Criteria

- [x] `Tests/UI/test_library_canvas_scoped_sync.py` passes in full on dev
- [x] Each of the four repaired tests exercises the canvas-scoped sync assertion it was written for, shown by failing when its subject behaviour is mutated — not merely by no longer raising AttributeError
- [x] A missing screen attribute the subject reads fails with a message naming the attribute, or the double is derived from the real screen's attribute surface so it cannot drift again

## Implementation Plan

1. Re-verify the premise at base `ecc0a531c8`: the original
   `_library_prompt_browse_controller` AttributeError is gone (the double at
   line 242 was fixed upstream); classify the current 6 failures.
2. Classify: `test_import_status_lines_patch_the_mounted_static_without_recompose`
   is the same harness-drift class, one attribute later
   (`_patch_library_skills_import_status_line`, introduced by 579135667a /
   task-32055 on 2026-09-08) — live defect, fix red to green. The five
   mounted-app tests fail at setup with `RecoveryRequired:
   raw_source_selection_changed` (ADR-126 admission) — known class; enroll
   those five nodes with `@pytest.mark.bootstrap_profile` (TASK-32873
   per-node precedent; module-wide enrollment stays with the class owner).
3. Repair the skills status double by binding the real production methods
   onto it so its method surface cannot drift from `LibraryScreen` again.
4. Negative controls: mutate the subject (`_sync_library_canvas` forcing a
   screen recompose; `_patch_library_skills_import_status_line` recomposing
   instead of patching) and show the repaired tests fail, proving they
   exercise their canvas-scoped assertions.
5. Full-file run green; targeted runs only; baseline A/B via
   `git checkout HEAD -- <paths>`, never `git stash`.

## Implementation Notes

**Classification.** Premise re-verified at base `ecc0a531c8` (dev tip) before
any edit, because the task was filed 2026-08-23 against dev `b2b1e2e0d`:

- The named `_library_prompt_browse_controller` AttributeError is **fixed
  upstream** — `test_prompt_and_skill_row_handlers_route_to_their_canvas`
  now provides that attribute (line 242) and passes at base unmodified.
- The same harness-drift class was still live one attribute later:
  `test_import_status_lines_patch_the_mounted_static_without_recompose`
  failed at base with `AttributeError: 'types.SimpleNamespace' object has
  no attribute '_patch_library_skills_import_status_line'`. Commit
  `579135667a` (task-32055, 2026-09-08) split
  `_apply_library_skills_import_status` into a status-line builder plus a
  patcher; the harness double was never updated. **Live defect, fixed.**
- The five mounted-app tests (notes per-click, media/RAG toggles, real
  prompt/skill rows, ingest backend switch, notes latency probe) failed at
  base at setup with `RecoveryRequired: raw_source_selection_changed` —
  the ADR-126 config-admission class. They never reached their assertions.
  Enrolled per-node with `@pytest.mark.bootstrap_profile` (the TASK-32873
  precedent already encoded in `Tests/conftest.py`); module-wide enrollment
  for the wider Library mounted family (e.g. `test_library_prompts_canvas.py`,
  202 red at base with the same signature) stays with that class's owner.

**What shipped** (only `Tests/UI/test_library_canvas_scoped_sync.py` changed):

1. `_library_skills_status_screen()` builds the Skills status double with
   its METHOD surface bound straight from `LibraryScreen` via `MethodType`
   (visibility guard, status-line builder, patcher, structural-wait
   accessor); only STATE is faked. The production methods execute against
   the double, so the test measures the real patching/fallback path, and
   method-level drift is impossible; a future state read the double lacks
   still fails with an `AttributeError` naming the attribute (AC3, both
   arms). Mocking the new methods instead would have satisfied the
   AttributeError but tested nothing — the exact trap AC2 forbids.
2. `@pytest.mark.bootstrap_profile` added to the five mounting tests.

**Red -> green evidence.** Targeted runs only, `-p no:randomly` plus one
default-order run (both green):

- BASE (before edits): `6 failed, 3 passed` (1 harness AttributeError +
  5 admission) — command:
  `python -m pytest Tests/UI/test_library_canvas_scoped_sync.py -q -p no:randomly`
- HEAD (after edits): `9 passed` (same command, 25.95s; default plugin
  order 23.20s).
- Negative control 1 — scratch mutation forcing `_sync_library_canvas`
  back to `screen.refresh(recompose=True)`: `4 failed, 5 passed`; the four
  identity-asserting mounted tests (notes per-click, media/RAG, real
  prompt/skill rows, ingest backend) all FAILED on their
  `calls == []`/canvas-identity assertions. Restored via
  `git checkout HEAD -- tldw_chatbook/UI/Library_Modules/canvas_sync.py`.
- Negative control 2 — visibility guard removed from
  `_patch_library_skills_import_status_line`: status-line test FAILED
  (`assert_called_once_with`). Restored the same way.
- Negative control 3 — notes select-toggle handler neutered: latency probe
  FAILED on its documented behavioral assertion (clicks must complete the
  select-mode transition; it deliberately carries no identity/timing
  assertion, per its own docstring). Restored the same way.
- `git status` clean under `tldw_chatbook/` after each restore; no
  `git stash` used at any point.

ADR required: no
Reason: test-harness repair plus per-node admission enrollment following the
existing TASK-32873/ADR-126 precedent already recorded in `Tests/conftest.py`;
no production code, storage, or interface decision changed.

Files changed: `Tests/UI/test_library_canvas_scoped_sync.py`, this task file.

**For the class owner:** the wider Library mounted family
(`Tests/UI/test_library_prompts_canvas.py`, and possibly other
`test_library_*` mounted suites) is red at base with the same
`RecoveryRequired: raw_source_selection_changed` signature (202 failed in
test_library_prompts_canvas.py alone). Module-wide enrollment decisions for
those files belong to that class's owner, not this task.
