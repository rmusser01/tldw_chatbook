---
id: TASK-42422
title: "Switching from Notes to a resident Media list by press still lands Library focus on the pane grip and fails UI Fast Lane 33"
status: Done
assignee:
- '@claude'
created_date: 2026-11-22 22:10
updated_date: '2026-11-23 04:57'
labels:
- follow-up
- ci
- library
- focus
- flaky
priority: high
references:
- Tests/UI/test_library_rail_switch_keeps_reader_and_note.py
- tldw_chatbook/UI/Screens/library_screen.py
- tldw_chatbook/UI/Library_Modules/screen_constants.py
- scripts/ui_pr_gate_census.txt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 21 of the follow-up burn-down (2026-11-22). Checked on the round-21 stack 77f124a740 (dev ddd3c02052 plus seven ports); paths are under `tldw_chatbook/` unless they start with `Tests/` or `scripts/`.

**Why.** The required "UI Fast Lane (33)" fails on `Tests/UI/test_library_rail_switch_keeps_reader_and_note.py::test_each_library_switch_lands_on_its_first_row_never_a_grip[notes-press-resident-media]` with `library-media-row-0: focus on LibraryAdaptiveReaderPaneGrip(id='library-browse-library-grip', name='Collapse Library pane', ...) (in the focus chain: True)`, after the test's full wait (call 21.9 s). In a scan of the 140 Derived Artifacts PR runs created from 2026-11-20 00:00Z to about 21:50Z on 2026-11-22 (every attempt; times are job start times), it failed 5 of the lane's 6 failed jobs, from 2026-11-21 02:26Z to 2026-11-22 15:09Z: runs 44758099535, 44944358256 (listed in TASK-33620.15.61's notes), 44954347282, 44974765316 and 45031398878. It passes locally at the base.

**Defect class.** Focus landing on a pane grip instead of the first row is TASK-41504's defect class; its description recorded exactly this switch ("Notes then resident Media `press()` lands on a grip about 1 in 3"). TASK-41504 is Done, so this residual is untracked. It is not TASK-32301's ceiling: when a list arrives after the fixed entry window, focus lands nowhere, while here it lands on a focusable grip, which is what a user on a slow machine would be left on.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The path that leaves focus on `#library-browse-library-grip` after a Notes-to-resident-Media press is identified and recorded in the notes
- [x] #2 After that switch, focus lands on the Media list's first row and never on a pane grip, including when the switch's work is slowed (shown with a temporary delay, reverted, that fails before the fix)
- [x] #3 The parametrized test passes in repeated runs of its file under parallel load (at least 20 runs, or a soak recorded in the notes)
<!-- AC:END -->

## Implementation Plan

1. Trace the switch (a `set_focus` / `_reset_focus` / arm / disarm / Media restore trace in a throwaway pytest probe) under parallel load until a natural run lands on the grip; name the path from the trace.
2. Reproduce it on demand with a temporary forced delay at base, and record the red.
3. Fix it where it lives: the Library list-entry focus must not land on a control that a queued or running recompose is about to replace (TASK-40293's rule, so far applied to rows only), in `_focus_library_list_entry`, the seam every list route shares.
4. Pin it in `Tests/UI/test_library_rail_switch_keeps_reader_and_note.py` with a deterministic hold of the Media repaint; show it red at base and green with the fix.
5. Siblings: the other `test_each_library_switch_lands_on_its_first_row_never_a_grip` cases and the other entry seams that land on a fallback control.
6. Soak (20 runs at `-n 4`, 5 serial runs with the UI lane's flags), covering tests, ratchets, ruff F, preflight; notes, ACs, Done.

## Implementation Notes

**Cause (AC#1).** A throwaway probe traced `set_focus`, `_reset_focus`, the list-entry arm and disarm, and the Media focus restore. It ran the switch 40 times at `-n 10` (load ~21) and caught one natural failure ending on `#library-browse-library-grip`:
1. The press shows the resident Media canvas and arms list-entry focus. The canvas recomposes, and the page result queues a second recompose.
2. During the first recompose's mount, the rebuilt toolbar is mounted but its rows are not, so `query(".library-media-row")` finds nothing. The entry's 50 ms retry tick took the empty-list branch of `_focus_library_list_entry` and landed on `#library-media-type-filter`. That branch checked only `_pruning`; TASK-40293's queued-recompose check guarded rows, not the fallback control.
3. The queued recompose removed `type:`. Textual's reset parked focus on the Library grip, and its `DescendantFocus` disarmed the Media entry (only Notes treats a grip landing as the app's own).
4. Nothing re-landed focus:
   - the sync's default restore had captured "nothing focused";
   - its `focus_pending_entry` follow-up is gated on the arm's generation, which the disarm ended;
   - the page result's own `#library-media-row-0` follow-up survives only when no later Media sync queues a follow-up over it (`queue_after_recompose` replaces).

   That last condition is why CI saw the failure only 5 times in 140 runs.

It is a product race, not test timing: the arm is gone and focus stays on the grip for the test's whole 15 s wait, as in all five CI logs (call 20.2-22.1 s).

**Fix.** `_library_entry_target_replaced` in `UI/Screens/library_screen.py` returns true when a widget is `_pruning`, or when any ancestor has a queued (`_recompose_required`) or running recompose. Running is read from the widget's `lock` (held by `batch()` for the whole `Widget.recompose`), which every widget has. The first version read `recompose_in_flight`, which only `RecomposeCaptureGuard` subclasses carry; review replaced it, see "Review fixes" below. `_focus_library_list_entry` now uses it for the first row, the shared empty-list control (Notes, Prompts, Skills, Conversations, Collections) and Media's empty-list control. On a hit it retries within the arm's window, so the entry lands once the repaint has finished. Before, it landed on a control the repaint was about to remove. This is TASK-40293's rule widened with TASK-41703's "the flag clears as the recompose starts", in the one seam every list route shares.

**Red/green (AC#2).**
- **New case** `test_each_library_switch_lands_on_its_first_row_never_a_grip[notes-press-resident-media-rows-held]`: it holds `LibraryMediaRowScroll._compose` until an entry attempt has run with the toolbar live and no rows. It asserts that nothing is focused at that point, then the case's usual first-row, focus-chain and no-grip checks. Base: 8 of 8 failed, `mid-repaint focus on Button(id='library-media-type-filter', ...)`; with the earlier end-only form, 4-6 of 8 failed on the grip. Fix: 8 of 8 passed (load 34-43).
- **Temporary forced delays** in the probe (reverted; the probe is deleted):
  - Rows held, at base: 2 of 8 ended on the grip.
  - Rows held, with the facet result released after the page result, at base: 4 of 8 ended on `type:`.
  - A 0.15 s sleep before the canvas teardown, at base: 2 of 3 ended on the grip.
  - With the fix, each passed 8 of 8.

**Soak (AC#3).**
- The parametrized test (11 cases), 20 runs at `-n 4`: 220 passed in 7 min 36 s, load 27-40.
- The same test, 5 serial runs with `-p no:xdist --timeout=180`: 55 passed, load 35-37.

**Siblings.**
- Every other case of the parametrized test passed in both soaks.
- The guard sits in the shared seam, so the Notes, Prompts, Skills, Conversations and Collections entries get it too. With the first version's `recompose_in_flight`, Prompts and Skills got only the queued half: their list canvases are not `RecomposeCaptureGuard`s. The widget lock removed that dependency (review fixes below).
- Audited and left alone:
  - `_expire_library_list_entry_focus`: one Notes landing at window close. Review fixed this one too; see below.
  - `_land_library_media_emptied_page`: runs after the recompose it requested.
  - `_restore_library_media_focus`: runs after the canvas's recompose has returned and released its lock.
- Media's `on_descendant_focus` still disarms on a grip landing; Notes is exempt. That is no longer reachable from the entry's own landings, so it was not changed.

**Covering runs** (fix applied, `-n 4`).
- The 19 Tests/UI files that reach the entry channel (files naming `_focus_library_list_entry` or `_library_pending_list_entry_focus`, plus the two `test_library_focus_never_lands_on_a_removed_widget*` files, `test_library_collection_reader_journeys.py` and the extended rail-switch file): 167 failed, 1351 passed in 26 min, load 33-37. 163 of the 167 also failed at base with the same command, mostly plain tests hitting the local profile trap.
- The 167 rerun under a temporary local plugin that adds the `bootstrap_profile` marker (as TASK-41504 did; deleted): 8 failed, 159 passed. The same 8 fail at base too, 3 of 3 runs each side. They are three `test_library_screen_reuse.py` timer nodes, two `test_library_skills_canvas.py` nodes, and three `test_library_shell.py` Notes nodes that TASK-41504 also recorded as red or flaky at base.
- `test_library_note_newer_selection_cancels_pending_back_focus` failed 11 of 30 on head and 11 of 30 on base. It presses a row with `press()` inside the entry window; fixed in the follow-up below.
- Two head-only failures from the long run passed 10 of 10 on rerun.
- Library wiring (serial): 44 passed. Size ratchets plus the Library recompose ratchet (serial): 291 passed.
- `ruff check tldw_chatbook --select F` and `Tests/lint/ruff_gate.py` passed. Preflight passed.

**Lane time.** The new case joins a whole-file UI census entry, so the census is unchanged. Serial call time with the UI lane's flags: 9.8-11.3 s (load 35-37, beside a covering run). The unheld case measured 9.7-10.7 s in the same run.

**Size.** `library_screen.py` 32828 -> 32842 lines (1288 methods); ratchet row raised with a note.

**Docs.** No User Guide change: the documented landing (the list's first row) is unchanged, and the fix removes a race. Lesson added to `backlog/docs/lessons-textual.md`.

**Files.** `tldw_chatbook/UI/Screens/library_screen.py`, `Tests/UI/test_library_rail_switch_keeps_reader_and_note.py`, `Tests/Architecture/test_screen_size_ratchet.py`, `backlog/docs/lessons-textual.md`.

**Follow-up (coordinator, same entry window): `test_library_note_newer_selection_cancels_pending_back_focus`.**
- **Cause.** A trace (9 of 16 probe runs failed) showed this order:
  1. The Notes entry lands on note-1.
  2. The test's code-driven `press()` on note-2 opens the editor, and `arm_and_focus_editor` focuses `#library-note-body`.
  3. Before the body's queued `DescendantFocus` reached the screen (where it would disarm), a retry tick put focus back on note-1. The body's event then arrived stale, so the entry stayed armed until it expired.
- **Test timing, not product.** Real input disarms the window before the press:
  - Enter on the entry's row: `on_key` disarmed first in 16 of 16 probe runs, and focus ended in the editor every time.
  - A click on the other row: `get_focusable_widget_at` disarmed first in 8 of 8.
  - The only product code that presses a note row is Enter on the applied filter, which is itself a key.
- **Fix.** The test now waits, with the file's deadline helper, for `_library_pending_list_entry_focus` to clear before its code-driven press. Its assertions are unchanged.
- **Evidence.**
  - Before: 11 of 30 failed at `-n 4`, on both head and base.
  - After: 30 of 30 passed (load 33-43).
  - Negative control, temporary and reverted: with the cancel guards removed (`_supersede_library_notes_navigation`'s pending-focus clears and the generation check in `_release_library_notes_focus_after_snapshot`), the waited test failed 4 of 4 with `'library-notes-tree-note-1' == 'library-note-body'`.
- **Re-soak, parametrized rail-switch test 20 times at `-n 4`.**
  - A first run: 217 of 220 passed at load 37-57. The three failures were Collections cases ending with nothing focused (`media-press-collections` twice, `media-press-collections-empty-scope` once).
  - That is the capture loader outlasting the fixed 2 s entry window, TASK-32301's ceiling, which TASK-41504 recorded at loads above 50.
  - A temporary trace of `_library_entry_target_replaced` across 100 Collections case runs (load 38-46, all passed) never flagged a Collections target. The guard does not delay those landings.
  - A/B of the four Collections cases, 15 repeats each side: head 60/60, base 60/60, head again 58/60.
  - The final-tree run: 220 of 220 passed (load 33-37).
- **Slice and checks.**
  - The `test_library_shell.py` slice `-k "note and (focus or back or selection or entry)"` (60 nodes): 59 passed. The one failure, `test_library_shell_blank_note_autosaved_then_emptied_still_gcs_on_back`, is base-red or flaky, as above.
  - Preflight passed; ratchets 291 passed; ruff F and `ruff_gate.py` passed. No product code changed, so no ratchet row moved.
- **File.** `Tests/UI/test_library_shell.py`.

**Review fixes (review of 3c2b0313d4, "With fixes").**
- **The running half had no pin.** `[notes-press-resident-media-rows-held]` is caught by either half, because the page result has already queued the next repaint when the held attempt runs. With the running half deleted it still passed: 6 of 6, then 8 of 8 (review saw 7 of 7).
  - New case `[notes-press-resident-media-rows-and-page-held]`. It also holds `LibraryMediaBrowseController._search` and `_load_facets` until the rows are released, so nothing queues a repaint. It counts only attempts with no ancestor `_recompose_required`.
  - Queued half only: 8 of 8 red, `mid-repaint focus on Button(id='library-media-type-filter', ...)`.
  - Full guard: 8 of 8 green.
- **The running signal was mixin-only.** `recompose_in_flight` exists only on `RecomposeCaptureGuard` subclasses, and `LibraryPromptsListCanvas` and `LibrarySkillsListCanvas` are not. `_library_entry_target_replaced` now reads `node.lock.is_locked` instead. `Widget.recompose` holds that lock through `batch()` until its new children, nested ones included, have mounted. The held cases confirm it: with only the lock and no queued repaint, both the Media and the Prompts holds wait, so the lock is still held while a nested `_compose` is blocked. `library_entry_canvases.py` already used `lock.is_locked` as "a recompose is active". In Library code, only `batch()` holds a widget lock.
  - New test `test_prompts_list_entry_waits_out_a_running_repaint`, on a non-mixin route. It holds `#library-prompts-list`'s `_compose`, re-arms the entry and recomposes the Prompts canvas.
  - With `recompose_in_flight` (3c2b0313d4): 6 of 6 red, `mid-repaint focus on Input(id='library-prompts-filter')`.
  - With the queued half only: 8 of 8 red.
  - With the lock: 8 of 8 green.
- **Notes window-close fallback.** `_expire_library_list_entry_focus` used to land on the filter even when a repaint was replacing it. It now waits one retry tick at a time when its target is being replaced, up to 20 ticks (1 s), then lands as before. The new timer handle is the one a disarm stops, so a key or click still ends it.
  - New test `test_notes_entry_window_close_waits_for_a_filter_being_repainted`. It sets an empty Notes list, nothing focused, the window open and a Notes canvas repaint queued, then fires the expiry.
  - At 3c2b0313d4: 6 of 6 red, `landed on LibraryRailSearchInput(id='library-notes-filter') mid-repaint`.
  - Fixed: 6 of 6 green. It ends on the new filter, never a grip.
- **Minors.**
  - The held tests release their holds in a `finally`.
  - The helper docstring, the lesson and the notes above now say the running signal is the widget lock and needs no mixin.
  - The lesson's review incident says the CI-shaped held case is caught by either half, and the page-held variant needs the running half.
- **Runs** (final tree):
  - Parametrized test (12 cases), 20 runs at `-n 4`: 240 passed in 8 min 36 s, load 29-38.
  - 5 serial runs with `-p no:xdist --timeout=180`: 60 passed, load 17-31.
  - Whole touched file 3 times at `-n 4`: 20 passed each.
- **Covering run.** 28 Tests/UI files (the 19 above plus nine Prompts and Conversations files), `-n 4`, with the temporary `bootstrap_profile` plugin: 28 failed, 2053 passed (34 min, load 16-62).
  - Rerun of those 28: 27 fail on this tree and 26 on 3c2b0313d4's `library_screen.py`, with no failure unique to 3c2b0313d4.
  - The one difference, `test_library_prompts_canvas.py::test_library_prompt_pager_first_and_filter_failure_states[size1]`, failed 5 of 10 here and 4 of 10 at 3c2b0313d4. It fails on a compositor geometry check, not focus.
- **Checks.** Wiring: 44 passed. Ratchets: 291 passed. Preflight, `ruff --select F` and `ruff_gate.py` passed.
- **Lane time** (serial, UI lane flags): the page-held case 9.4-10.1 s (load 17-31), the Prompts test 3.5 s and the Notes expiry test 2.8 s (load ~17). All are in a whole-file census entry, so the census is unchanged.
- **Size.** `library_screen.py` 32842 -> 32857 lines (1288 methods); ratchet row raised by exactly that, with a dated note.
