---
id: TASK-1034
title: 'Results grid intermittently drops its column header and [warned] canary marker'
status: Done
assignee:
  - '@claude'
created_date: '2026-07-27 16:00'
updated_date: '2026-07-28 00:16'
labels:
  - evals
  - bug
  - ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during UAT of the Evals screen on `origin/dev` (155574902), driven live against llama.cpp.

The results table renders in **two structurally different ways within the same view**, and which one you get is not under the user's control:

- **Boxed, no header.** Rows render inside a drawn box with no column header at all, e.g. `│The protestors were [neutral]  "mente"  49%`. The target column is unlabelled and the `[warned]` canary marker is absent.
- **Unboxed, with header.** The same grid renders `Snippet | Sample target (llama.cpp) f0fded1f [warned]` above the rows.

Reproduced repeatedly. Immediately after "Create sample bench" completes the grid appears in the boxed/headerless form; at one point during lens interaction it switched to the headed form; a later systematic pass over all five lenses showed `header=0` on every one, including Entropy which had shown a header minutes earlier. Clicking a cell did not restore it. So it is **not lens-dependent** — it is some other state we did not isolate.

Two things are lost in the headerless state, and both matter:

1. **Column identity.** With one target the reader can infer it; with several the numbers become unattributable, which is the whole point of the grid.
2. **The `[warned]` canary marker.** This is the column-level signal that the target preflighted with a degenerate canary. Losing it silently removes the interpretive guardrail the design added on purpose.

Worth investigating whether the boxed form is a different widget (an error/placeholder container) rather than the `DataTable` with `show_header` off — the row prefixes differ (`│ │The…` boxed vs `│  The…` headed), which suggests two render paths rather than one widget in two states.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The results grid always renders its column header, in every lens and every entry path
- [x] #2 The `[warned]` marker is present whenever the run's canary is degenerate
- [x] #3 The two render paths are reconciled to one, or the second is explained and made deliberate
- [x] #4 A test fails if the header is absent after a fresh mount
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the module docstring/results_grid.py to understand ResultsGrid's mount/focus flow.
2. Confirm the outline hypothesis empirically with a real Pilot + compositor render (screen._compositor.render_strips(), the pattern test_lab_mode_strip.py already uses) comparing the DataTable focused vs blurred.
3. If confirmed, check whether DataTable already has its own non-outline focus differentiation (native cursor/header styling) before choosing a fix, so removing the outline does not silently remove the only focus cue.
4. Scope a fix in css/features/_evals.tcss (source, not the generated bundle), regenerate the bundle via build_css.py.
5. Add a Pilot-driven regression test reading real compositor output (not table.columns[].label, which cannot see this defect) asserting the header and [warned] marker survive while focused.
6. Revert-check: temporarily rebuild the bundle from the pre-fix source and confirm the new test fails with a concrete error; rebuild the fixed bundle again afterward.
7. Run the targeted test files plus any focus/CSS contract tests.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause confirmed, not the "two render paths" red herring the description worried about: there is only ever one widget, the `DataTable` at `#evals-grid-table`. `ResultsGrid.on_mount` (results_grid.py) calls `.focus()` on it immediately after every fresh mount -- so the "boxed, no header" state UAT saw right after "Create sample bench" was simply the DEFAULT post-mount state. The global fallback `*:focus { outline: solid $ds-focus-accent; }` (core/_reset.tcss) draws its outline INSIDE a widget's own box, overwriting the widget's own first/last rendered rows rather than sitting outside them (unlike `border`). A DataTable's first rendered row is its header, so a focused grid's outline literally replaced the header (and any `[warned]` suffix in it) with the outline's box-drawing top edge -- reproducing "boxed, no header" exactly. Blurring the table (e.g. focus moves to `#evals-lens-selector`) removes the outline and the header reappears -- reproducing "unboxed, with header" exactly. Confirmed empirically with a real Pilot run reading `screen._compositor.render_strips()` (Textual 8.2.7 has no `App.export_text()`) in both focus states before writing any fix.

Verified DataTable's own built-in cursor highlighting (`.datatable--cursor`) does NOT differ between focused/blurred in this app (components/_lists.tcss restyles it unconditionally, bold+underline either way) -- so the outline was ALSO this table's only focus cue. Simply deleting it would have swapped one defect (lost header) for another (invisible focus), which the task explicitly ruled out.

Fix (css/features/_evals.tcss, scoped to `#evals-grid-table`, not a blanket `DataTable`/`*` change): `#evals-grid-table:focus { outline: none; }` plus `#evals-grid-table:focus > .datatable--header { background: $ds-focus-bg; color: $ds-focus-fg; text-style: bold underline; }`. This recolours the header itself as the focus cue -- geometry never changes (no border added, no row consumed), so there is no layout shift and the header/`[warned]` text is always drawn, never overlaid. Chose grid-scoped over a global `DataTable:focus` rule because ~25 other files construct `DataTable`s across the app that were not exercised or verified here; narrowing the blast radius to the one confirmed-broken widget matches this codebase's own precedent (TASK-383's chip fix, TASK-445's ListView fix, both similarly scoped instead of touching core/_reset.tcss). Rebuilt tldw_cli_modular.tcss via build_css.py; never hand-edited.

Added `test_focused_results_grid_keeps_its_header_and_the_warned_marker` (Tests/UI/test_evals_results_grid.py) plus a `_rendered_text` helper (mirrors test_lab_mode_strip.py's own) that reads the real compositor output instead of `table.columns[...].label`, which is what every earlier warned-header test in this file asserted on -- and which cannot see this defect at all, since the column label was always correct in the DataTable's data model; only its on-screen paint vanished. The new test makes no extra `.focus()` call -- `_select_run_group` already leaves the table focused via `on_mount`'s auto-focus, so it fails on the exact default, no-interaction path UAT hit.

Revert-check: reverted only _evals.tcss, rebuilt the bundle, ran the new test -- failed with `AssertionError: column header missing from the focused grid's rendered output:` followed by the captured boxed/no-header ASCII render (the `┌────┐` box, no "Snippet", no "[warned]"), i.e. byte-for-byte the UAT symptom. Restored the fix and rebuilt; the same test then passes along with the rest of the file (56/56 in test_evals_results_grid.py + test_evals_screen.py combined) and test_focus_accessibility.py / test_console_transcript_selection_contract.py (14/14). test_non_obscuring_focus_contract.py has 9 pre-existing failures (retired `.collapsible--header`, `.preset-button`, and a missing `_chat_tabs.tcss` file) -- confirmed via `git show HEAD:<path>` that the test file is byte-identical to HEAD, i.e. unrelated baseline breakage in this worktree, not caused by this change.

Files touched: tldw_chatbook/css/features/_evals.tcss (source fix), tldw_chatbook/css/tldw_cli_modular.tcss (regenerated bundle), Tests/UI/test_evals_results_grid.py (regression test).
<!-- SECTION:NOTES:END -->
