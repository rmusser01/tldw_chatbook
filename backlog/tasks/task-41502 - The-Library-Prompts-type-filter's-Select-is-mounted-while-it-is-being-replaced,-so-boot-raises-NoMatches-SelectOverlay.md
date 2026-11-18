---
id: TASK-41502
title: "The Library Prompts type filter's Select is mounted while it is being replaced, so boot raises NoMatches SelectOverlay"
status: Done
assignee:
- '@claude'
updated_date: 2026-11-18 07:11
created_date: 2026-11-17 05:27
labels:
- follow-up
- library
- prompts
- ui
priority: medium
references:
- tldw_chatbook/UI/Screens/library_screen.py
dependencies:
- TASK-41127
- TASK-41141
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41127 and re-measured in its fix round (round 15). `#library-prompts-type-filter` mounts a `Select` while the previous one is being replaced; `NoMatches 'SelectOverlay'` / `'#label'` is raised. Boot with the Prompts default tab fails 3 of 3 and the Prompts route 2 of 3. TASK-41141 fixed the exit-time mount fence for stock Select widgets; this is a different, mount-time overlap.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Booting with the Prompts default tab and opening the Prompts route raise no `NoMatches` for the type filter, 5 of 5 runs
- [x] #2 The two failing nodes are named in the Implementation Notes with red-before and green-after evidence
<!-- AC:END -->

## Implementation Plan

1. Reproduce at the base under load and find what prunes the type filter before it composes (probe `App._prune` and `Select._on_mount`), and whether `App._exit` is involved (TASK-41500's lesson).
2. Pin the window deterministically in `Tests/UI/test_library_prompts_canvas.py`: hold the new type filter's Compose during a canvas `sync_state` recompose, replace the canvas (as the Library snapshot reconcile does), release, and assert the app recorded no exception. Show it red.
3. Fix at the cause's seam, then rerun the two named `test_screen_navigation.py` nodes 5 times each under load and record red-before/green-after.
4. Lane the new test (census tail, `MINIMUM_FILES`), run covering tests, ratchets and preflight.

## Implementation Notes

- **Cause: a prune, not the exit fence.** At boot on Prompts (and on the Prompts route) the browse result's `sync_state` recomposes `#library-prompts-canvas`. Meanwhile the Library snapshot reconcile (`_reconcile_library_entry_state`, `library_screen.py:13472`) removes that canvas to mount the source-error callout, which the test harness shows because its Library sources are unavailable. The recompose's new type filter is registered but has not composed yet. `App._prune` marks it `_pruning`, so its compose mounts nothing, and the stock `Select._on_mount` raised `NoMatches 'SelectOverlay'` (or `SelectCurrent`'s `'#label'`) into `App._exception`, which ends the app. A probe of `App._prune` and `Select._on_mount` found `_pruning` True at the filter's Mount in 10 of 10 failing runs. `App._exit` was False in all but one, where the app was already shutting down. The replacement is the reconcile's intended order, so it stays as it is.
- **Fix.** The type filter is a `PruneSafeSelect` (TASK-1960). It skips child setup only while `_pruning`, `_closing` or `App._exit` is set, and catches no exception. It is the filter's only `Select`. `PruneSafeSelect` is already resident at UI-ready and the canvas is not imported by `import tldw_chatbook.app`, so the import-weight and UI-ready censuses are unchanged.
- **Pin.** `Tests/UI/test_library_prompts_canvas.py::test_type_filter_survives_its_canvas_being_replaced_mid_recompose` holds the new filter's Compose during a `sync_state` recompose, removes the canvas as the reconcile does, then releases the Compose.
  - Red with stock `Select`: `assert NoMatches("No nodes match 'SelectOverlay' on Select(id='library-prompts-type-filter', ...)") is None`.
  - Green with the fix. It also passed 8 of 8 parallel runs at load 18-28.
- **AC#2, the two nodes.** `Tests/UI/test_screen_navigation.py::test_boot_with_prompts_default_tab_lands_on_library_with_prompts_row_selected` and `::test_prompts_route_lands_on_library_with_prompts_row_selected`, run as 16 parallel single-process copies with `-p no:xdist --timeout=180`:
  - Red at base, load 9-19: boot failed 14 of 16 and route 7 of 16. Every failure was `NoMatches 'SelectOverlay'` (17) or `'#label'` (4) on the type filter. At load ~8 both passed 6 of 6, so the race needs load.
  - Green at head, load 9-21: 16 of 16 for each node.
- **Lane.** Census tail, `MINIMUM_FILES` 2250 -> 2255, shared with TASK-41503. Serial lane-flag times at load ~15: boot 8.9 s, route 9.5 s, and the pin under 1 s call (2.6 s for the process).
- **Covering runs.** `test_library_prompts_canvas.py`, `test_library_prompt_mixins.py`, `test_library_prompts_reader.py`, `test_library_canvas_scoped_sync.py`, `test_library_entry_compose_once.py`, `test_library_choice_strips.py`, `test_library_prompt_collections.py`, `Tests/Widgets/test_prune_safe_select.py` and `test_library_prompts_wiring.py` at `-n 4`, load 16-28: 302 passed, 296 failed. All 296 fail the same way with the base source restored: 288 are the local `RecoveryRequired: raw_source_selection_changed` trap, and the rest are base reds in `test_library_prompts_canvas.py`, among them the ConfirmationDialog `Container` and the notes-editor CSS-parity checks. `test_library_mixin_create_and_filtered_pagination_totals` sets the filter's value on the real screen and passed.
- **Lesson.** `lessons-textual.md` gains a paragraph under TASK-41500's entry: this `NoMatches` was a prune, not the exit fence.
- Files: `tldw_chatbook/Widgets/Library/library_prompts_canvas.py`, `Tests/UI/test_library_prompts_canvas.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, `backlog/docs/lessons-textual.md`.
