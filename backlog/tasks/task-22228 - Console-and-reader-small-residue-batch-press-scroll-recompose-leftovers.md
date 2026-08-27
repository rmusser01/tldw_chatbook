---
id: TASK-22228
title: Console and reader small-residue batch (press/scroll/recompose leftovers)
status: Done
assignee:
  - '@claude'
created_date: '2026-08-24'
updated_date: '2026-08-26 10:10'
labels:
  - performance
  - cleanup
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22228).

Verified small items that do not warrant tasks alone (each with cites in the evidence
doc):
1. `chat_screen.py:18953-18956` — `on_mouse_up` runs an id `query_one` on every Console
   mouse-up before any cheap guard (same physical press 21119 cleaned; cheaper class).
2. `left_rail.py:145-148`, `:1250-1267` — the left rail lacks the TASK-21117 pure-scroll
   split the Inspector rail has (guarded, but 2 query_one + max_scroll_y per frame).
3. `left_rail.py:561-585` — `_focusable_body_controls` does an uncached subtree
   `query("*")` per focus change in the rail.
4. `Workspaces/display_state.py:631-688` — 3+ stat/realpath syscalls per bound folder per
   state build (deliberate ADR-028 posture; frequency drops with TASK-22201 — re-evaluate
   a short-TTL cache for network mounts after it lands).
5. `Widgets/Prompts/prompt_block_editor.py:818-888` — `_sync_footer` fires 3 unguarded
   `Static.update()` (layout=True) + an unguarded tooltip write per keystroke while the
   prompt workbench is open (partly PR #2053).
6. `library_screen.py:32300`, `:32311`, `:32506-32521`, `:33599`, `:33611`, `:14316` — six
   reader button presses still whole-screen recompose; the two delete-confirm presses
   re-parse the document in Read mode.
7. `library_screen.py:5741-5755` + `library_media_reader_shell.py:117-127` — two layout
   resolves per Resize event (screen-level one fires even when Media is not active).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each numbered item is fixed as described or explicitly declined with a reason in the notes
- [x] #2 No behavior change beyond the stated mechanics; touched areas keep their tests green
- [x] #3 Fixes verified by the cheap probe named in the evidence doc where one exists
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-verify each of the seven items on base 6aafb01c0 (the review's cites are stale and most of the burn-down has merged); record which still reproduce.
2. Item 1 (chat_screen on_mouse_up): route through the existing memoized _console_composer_or_none() instead of a per-mouse-up id query_one.
3. Item 2 (left_rail pure scroll): measure the per-scroll-frame cost of _update_outer_hint; memoize the two node lookups / add a cheap stand-down where the evidence supports it.
4. Item 3 (_focusable_body_controls): count the subtree walked per focus change; fix only if the measured walk is material, otherwise decline with the count.
5. Item 4 (display_state stat/realpath): re-measure state-build frequency after TASK-22201; decline if the ADR-028 freshness posture now costs a bounded number of syscalls per interaction.
6. Item 5 (_sync_footer): equality-guard the three Static.update() writes and the tooltip write; pin with a test that bites under mutation.
7. Item 6 (six reader presses): route through the sanctioned _sync_library_media_viewer_or_recompose() seam; report the recompose-ratchet census before/after.
8. Item 7 (double layout resolve per Resize): scope the screen-level leg so it does not resolve the Media layout when Media is not the active destination, and equality-guard the redundant second resolve.
9. Red-first probes per fixed item, before/after measurements, targeted suites + the merged 22203/22207/22208/22209 gate files, --collect-only sweep, preflight, mutation tests.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Seven verified residue items re-measured on base `6aafb01c0` before touching anything.
Four are fixed, three are declined with the measurement that killed them. Every number
below is from a mounted-app probe on this machine (Console harness at 160x45 = 475
widgets; Library harness at LIBRARY_TEST_SIZE = 140 widgets with the reader open).

**The measurement that reshaped the batch.** Textual 8.2.8's `DOMNode.query_one` takes
an id-selector fast path (`walk_breadth_search_id` + a per-node `_query_one_cache`), so
`query_one("#id")` is 0.3 us warm / 5.2 us cold — NOT the DOM walk items 1-2 were filed
as. `query(...)` takes no such path: `DOMQuery.nodes` builds its candidates from
`walk_children(Widget)` and then runs the parsed selector through `match()` for every
node it walked, which for the universal selector is pure overhead. The finding that
looked smallest (item 3) was the only expensive one. Recorded in
`backlog/docs/lessons-testing-evidence.md`.

**Item 1 — Console `on_mouse_up` id query. DECLINED.** Reproduces exactly as described
(10 mouse-ups = 10 `query_one`), but it costs 0.3 us warm / 5.2 us worst-case cold per
press. The prescribed fix — route through the memoized `_console_composer_or_none()` —
measured SLOWER on the hit path (0.7 us) because the memo revalidates by building
`ancestors_with_self`. Implementing it would have been a pessimization with a green test.

**Item 2 — left rail pure-scroll split. DECLINED.** The structural half of TASK-21117 is
already there: `_ContextOuterBody.watch_scroll_y` routes to `_update_outer_hint` only,
never into `request_allocation_reconcile`, so a scroll frame already skips the reconcile,
the refold chain and focus recovery. The named residue (2 `query_one` + `max_scroll_y`)
measures 2.7 us per scroll frame (10.4 us with a cold cache), against a frame that
repaints the rail. Confirmed 9 hint updates for 10 scroll writes.

**Item 3 — `_focusable_body_controls` subtree walk. FIXED, 87.8 us -> 23.4 us per call.**
`bounded.viewport.query("*")` over the 16-node Model body cost 74.1 us; the identical
`walk_children(Widget)` costs 2.2 us. It runs once or twice per focus change in the rail
(4 calls per 6 focus moves, measured). New gate
`Tests/UI/test_console_left_rail_focus_walk.py`: a perf arm (zero selector queries on
every mounted section) plus an equivalence arm that pins the returned tuple against the
`query("*")` result section by section, with a non-vacuity assertion.

**Item 4 — ADR-028 filesystem binding recompute. DECLINED.** TASK-22201 already cut the
frequency to ONE state build per settled tick (pinned by
`test_settled_tick_builds_workspace_state_once`), and the recompute costs 7.1 us per
bound folder warm (is_dir+is_symlink 1.8 us, resolve 5.2 us) — so a typical workspace
pays 7-21 us per tick for the guarantee that a deleted or symlink-swapped root cannot
keep reporting "ready". A short-TTL cache would buy those microseconds by reintroducing
exactly the stale-status window ADR-028 exists to close. The real residual risk is a
BLOCKING stat on an unreachable network mount, and a TTL cache does not fix that either
(the first call in each window still blocks the loop); moving the recompute off the loop
is a design change, not a residue item.

**Item 5 — prompt workbench per-keystroke footer writes. FIXED, 23.0 us -> 6.0 us per
`_sync_footer`, 15 no-op `Static.update` calls per 5 keystrokes -> 0.** Every one of the
15 wrote text byte-identical to what was already rendered, each with the `layout=True`
default. Guarded by text equality (module-level `_present_static`, mirroring
`ConsoleLeftRail._present_header_title`) rather than by `layout=False`: this copy really
does re-wrap at narrow widths, so its geometry is not fixed and the layout flag must stay
true for the writes that change something. The tooltip writes are guarded too — the
`Widget.tooltip` setter is not a plain assignment, it calls `screen._update_tooltip`. The
probe also caught the same class one layer down (`PromptBlockCard.sync`'s issue/status
Statics and its Duplicate tooltip, also per keystroke), so those are guarded by the same
helper. Two new arms in `Tests/UI/test_prompt_block_editor.py`; three mutations
(unguarded helper, unguarded footer tooltip, unguarded card tooltip) each turn them red.

**Item 6 — six reader presses. FIXED, 1 whole-screen recompose per press -> 0 for all
six.** Edit metadata / Cancel, Move to trash / Cancel, Edit analysis / Cancel now take
`_sync_library_media_viewer_or_recompose()` — the task-21116 seam Escape already used for
the identical three flips. Recompose census 80 -> 74; the ratchet pin was re-based 97 ->
74 in the same change (the 2026-08-24 reader burn-down had taken it to 80 without
lowering the pin, leaving 23 sites of headroom a ratchet cannot bite through). New gate
`Tests/UI/test_library_reader_press_scope_t22228.py` asserts zero screen recomposes,
rail and viewer node identity, and that each flip actually happened.

NOT converted, deliberately: the residual in-viewer document re-parse on the
delete-confirm flip. The 22207 cure (a persistent, display-gated widget) would leave
`#library-media-delete-confirm` permanently in the DOM, which turns three existing
presence assertions in `test_library_shell.py` (including `assert not
screen.query("#library-media-delete-confirm")`) into statements that cannot fail. Arming
a destructive confirm is a deliberate, rare gesture, not a per-keystroke path; trading a
real contract check for it is the wrong side of that bargain.

**Item 7 — double layout resolve per Resize. HALF FIXED.** Measured 3 resolves per
terminal resize with Media active (the screen gets 2 `Resize` events, the shell 1), and
2 per resize on the Conversations route — where the mounted shell does not exist and the
call could only walk the whole Library DOM to find nothing (16.0 us per FAILED
`query_one`, which takes no id-cache fast path). The screen-level leg is now scoped to
the Browse Media row, the one route that mounts the shell: off-route resolves 2 -> 0,
behaviour identical (the skipped call was already a no-op).

The on-route duplication is DECLINED. Collapsing it needs either a coalescing scheduler
across the shared adaptive-shell resize contract (all three destinations) or an equality
guard on `sync_layout` — and that guard is unsafe: `shell.library` IS `#library-rail`,
whose `display` is also written by `_apply_library_notes_stage_visibility`, so the
unconditional re-apply is what keeps the shell's view of its own pane authoritative.
50 us per resize frame does not buy that risk.

### Modified

* `tldw_chatbook/UI/Console_Modules/left_rail.py` (item 3)
* `tldw_chatbook/Widgets/Prompts/prompt_block_editor.py` (item 5)
* `tldw_chatbook/UI/Screens/library_screen.py` (items 6, 7)
* `Tests/UI/test_library_recompose_ratchet.py` (pin 97 -> 74)
* `Tests/UI/test_prompt_block_editor.py` (+2 arms)
* `Tests/UI/test_console_left_rail_focus_walk.py`, `Tests/UI/test_library_reader_press_scope_t22228.py` (new)
* `backlog/docs/lessons-testing-evidence.md`

### Verification

Targeted suites (counts read from tee files, this worktree, `.venv/bin/python -m pytest`):

* merged gates kept green: `test_library_media_reader_flow.py`,
  `test_library_media_reader_no_change_sync_t22208.py`,
  `test_library_media_reader_traversal_t22207.py`,
  `test_library_media_reader_match_nav_t22209.py`,
  `test_console_run_tick_workspace_reads.py`,
  `test_console_rail_reflow_hover_budget.py`,
  `test_console_avatar_geometry_offloop.py` -- **71 passed**.
* Console rail + prompts + prompt canvases (9 files incl. both new gates) --
  **1345 passed, 1 failed**; the one failure
  (`test_library_prompts_canvas.py::test_library_prompt_delete_receipt_undo_restores_row_and_count`)
  passes in isolation and again in a clean 348-test re-run of its file plus the
  ratchet and both new gates -- it fell over while a base-comparison checkout
  was swapping source files under a parallel run.
* Library media/adaptive-shell batch (9 files) -- **272 passed, 20 failed**.
  Nineteen of those twenty are PRE-EXISTING on base `6aafb01c0`: proven by
  reverting the three changed source files to the base blob in this worktree
  and re-running the same ids (19 failed, 108 passed, same names).
  `test_library_multiselect_media.py::test_single_delete_arm_supersedes_stale_receipt`
  was the one genuine break -- a `SimpleNamespace` double that stubs only
  `refresh`, so the seam call raised `AttributeError`. Repaired by stubbing the
  seam alongside `refresh` and asserting the arm now repaints through the
  viewer, not the screen.
* `--collect-only` sweep clean: 16,104 tests collected in `Tests/UI`, 3,699 in
  `Tests/Prompt_Management` + `Tests/Library` + `Tests/Workspaces`, no
  collection errors.
* `./scripts/preflight.sh` -- all derived-artifact checks passed.

Mutation tests (each restored afterwards): item 3's walk reverted to
`query("*")` -> perf arm red; `_present_static` forced to always write -> both
footer arms red; footer tooltip guard removed -> both footer arms red; card
tooltip guard removed -> the keystroke arm red; two of the six presses put back
on `refresh(recompose=True)` -> the press gate red; item 7's row guard replaced
with `if True` -> the off-route gate red. Deleting item 7's leg outright first
SURVIVED the control arm (the shell's own resize message resolves the layout
too, so a resolve-count assertion cannot see this leg); the arm was rewritten to
count calls carrying the pre-refresh `focus_intent`, which is unique to it, and
that kills the mutant.

### Dev reds found, not this task's to fix

Pre-existing on base `6aafb01c0`, all from the #2064 reader redesign moving
`#library-media-edit`/`#library-media-delete` behind the Reader's "More" region
and the rail's starter disclosure:

* `Tests/UI/test_library_per_click_recompose_t21116.py` -- **7 failed, 2 passed**
  before any edit in this branch; its `_boot_media_library` helper presses
  `#library-rail-explore-all`, which this harness no longer composes. This is
  the file that most directly covers item 6's subject, which is why the new
  gate carries its own boot helper.
* `Tests/UI/test_library_media_side_by_side.py` (4), `test_library_entry_compose_once.py` (13),
  `test_library_media_reader_shell.py` (1), `test_library_canvas_scoped_sync.py` (1).
* `Tests/UI/test_library_shell.py`, the `-k "media_edit or media_delete or
  analysis or confirm"` slice that covers every handler item 6 touched: **15
  failed, 10 passed on the BRANCH and the identical 15 failed, 10 passed with
  the three changed source files reverted to the base blob** -- same names, no
  difference (each failure costs a 30 s `_wait_for_selector` timeout, which is
  why the whole file takes over an hour here). Item 6 adds none of them.
<!-- SECTION:NOTES:END -->
