---
id: TASK-1764
title: Second "+ New target" press lands on nothing reachable
status: Done
assignee: []
created_date: '2026-08-01 14:55'
updated_date: '2026-08-01 14:55'
labels:
  - evals
  - ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_evals_steering_e2e.py::test_two_ui_authored_targets_one_steered_light_up_column_mode_delta` — task-1611's own closing E2E for the capability "mint an ADDITIONAL, differently-steered target through the UI" — failed at `assert len(created_after_second) == 2, "second Create must mint an ADDITIONAL row"`. Pressing "+ New target" a second time minted no row.

Root cause: `_refresh_targets_section`'s rebuild mounts every new target row (and, once a real `llama_cpp` model exists, `_build_target_add_control`'s Add picker row) ABOVE the "+ New target" mini-form, never below it. Each row added there pushes the just-pressed button down by exactly that many rows. At a realistic 160x45 viewport, `#evals-bench-editor` (the pane's one and only scrollable region — see its own CSS comment) had exactly enough slack for the FIRST create; the second click landed on `#footer-spacer`, not the button — confirmed directly via `Screen.get_widget_at`, the same hit-test a real mouse click and `pilot.click` perform.

This is not a one-off layout bug; it is the same shape of regression for the THIRD time in this pane's history. `_build_create_target_control`'s own docstring already documents two earlier live failures during task-1611 T2's own development, both answered by trimming the mini-form's own shape until it fit. This time it was task-1710's checkbox (`cb0407067`) that tipped the arithmetic over: a `margin-bottom: 1` on `#evals-bench-capture-continuations` spent the pane's one remaining row of 160x45 slack, and it went unnoticed because that row only matters once a target is staged and the Add picker also renders — a state none of task-1710's own tests happened to reach.

Why per-task review missed it each time: every task that touched this pane verified reachability of the ONE control it added or resized, at whatever target count its own tests happened to stage. Nothing tested the general property — that adding a row above the "+ New target" mini-form must not push the mini-form itself out of reach — so each fix was really "buys back enough slack for today's shape," which holds only until the next row is added anywhere above it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A bench author can press "+ New target" repeatedly and mint an additional target row each time, at a realistic viewport, without manual scrolling
- [x] #2 `test_two_ui_authored_targets_one_steered_light_up_column_mode_delta` passes for the real reason (the control is reachable), not because the test scrolls to it or the assertion was weakened
- [x] #3 A regression test pins reachability at MORE staged-target counts than the closing E2E happens to exercise, so the fix is proven count-independent rather than re-tuned slack
- [x] #4 The fix generalizes to any row added above the mini-form (the Add picker, a future field), not just today's target-row count
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm the root cause directly: reproduce with `Screen.get_widget_at` on `#evals-bench-create-target`'s own region after a second stage, showing it resolves to `#footer-spacer`.
2. Add `BenchEditor._keep_create_control_in_view`, called from the two paths that mount a row above the mini-form (`stage_target`, `_on_add_target_pressed`) via `call_after_refresh` (the rebuild's `region` is stale until Textual's next layout pass) — scrolls `#evals-bench-create-target-form` back into `#evals-bench-editor`'s viewport regardless of target count or terminal size. Deliberately not called from remove (list shrank) or a prompt-mode flip (user is at the top of the form).
3. Judge the uncommitted `_evals.tcss` change (removing `#evals-bench-capture-continuations`'s `margin-bottom: 1`) on its own merits now that the general fix exists: empirically verify whether it is load-bearing.
4. Add a regression test driving several presses past the E2E's own two, asserting hit-testability (`get_widget_at` on the control's own region center) after each.
5. Run the full affected suite plus the CSS bundle sync check.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Picked up mid-implementation from a prior session cut off by a quota limit; its diagnosis (documented above) and its `_keep_create_control_in_view` scroll fix were already correct and complete in the working tree — this pass verified, finished, and closed it out.

**The scroll fix (`bench_editor.py`)**: `_keep_create_control_in_view` scrolls `#evals-bench-create-target-form` into view via `self.call_after_refresh(...)` + `scroll_to_widget(..., animate=False)`, called from both `stage_target` (the `CreateTargetRequested` round trip) and `_on_add_target_pressed` (the Add-picker path) — the two places that mount a new row above the mini-form. Left uncalled from `_on_remove_target_pressed` (nothing was pushed anywhere) and `_on_prompt_mode_changed` (would yank the viewport away from the top-of-form control the user is actively looking at).

**The CSS change was reverted, not kept.** The working tree also carried an uncommitted `_evals.tcss`/generated-bundle change removing `#evals-bench-capture-continuations`'s `margin-bottom: 1` to buy back the one row task-1710's checkbox had spent. Verified empirically (bundle temporarily reverted to `HEAD`, scroll fix left in place) that the control stays hit-testable through at least four staged targets with the margin still present — the general scroll mechanism makes that row of slack unnecessary. Keeping a slack-based tweak alongside a fix whose entire point is "stop relying on slack" would reintroduce the exact anti-pattern this task closes out, so both `_evals.tcss` and the generated `tldw_cli_modular.tcss` were restored to their committed `HEAD` state (verified: `git diff` against both is empty; `Tests/UI/test_css_bundle_sync_guard.py` and `test_css_build_integrity.py` both pass with no rebuild needed).

**New regression test**: `Tests/UI/test_evals_bench_editor.py::test_create_target_control_stays_hit_testable_across_many_staged_targets` presses "+ New target" SIX times (the closing E2E only exercises two) and after every press asserts `Screen.get_widget_at(*control.region.center)` resolves to the button itself, at the same 160x45 viewport this module's other reachability checks use. Verified the test actually catches the regression: with the scroll-fix calls temporarily removed, it fails after the FIRST staged target with `landed on Static(id='footer-spacer') instead` — the identical failure mode the bug report describes, not a hypothetical.

Deleted the scratch probe (`Tests/UI/test_zz_probe_1764.py`) the prior session used to characterize the bug before committing.

Full requested suite (`test_evals_steering_e2e.py`, `test_evals_bench_editor.py`, `test_evals_screen.py`, `test_evals_empty_states.py`, `Tests/Evals/character_probe`) — 362 passed. CSS bundle sync guard — 10 passed, no drift.

Modified files: `tldw_chatbook/UI/Evals/bench_editor.py` (fix + module/method docstrings), `Tests/UI/test_evals_bench_editor.py` (new regression test). No CSS files changed in the final diff — the uncommitted `_evals.tcss`/bundle edit was reverted as superseded, not shipped.
<!-- SECTION:NOTES:END -->
