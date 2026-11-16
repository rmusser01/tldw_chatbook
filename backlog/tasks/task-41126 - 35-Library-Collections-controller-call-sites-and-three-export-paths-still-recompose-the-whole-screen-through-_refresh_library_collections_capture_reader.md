---
id: TASK-41126
title: "35 Library Collections controller call sites and three export paths still recompose the whole screen through _refresh_library_collections_capture_reader"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-16 21:24'
created_date: 2026-11-15 02:20
labels:
- follow-up
- library
- performance
priority: low
references:
- tldw_chatbook/UI/Library_Modules/library_collections_controller.py
- tldw_chatbook/Widgets/Library/library_collections_capture_reader.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** TASK-40328 made the Collections entry load repaint its panes in place (one recompose per press instead of four, about 0.4-0.6 s less UI-loop CPU per press). Every other Collections action still rebuilds the whole Library screen, with the same cost and the same focus and stale-rail risks that task had to fix.

**Evidence.** `library_collections_controller.py:424-432` (`_refresh_library_collections_capture_reader`) is a whole-screen `refresh(recompose=True)`; it is called from 35 places in that file (`:448` through `:1656`; the 36th match is the def itself). It has three more callers in `library_file_export.py`, the legacy recovery export's completion, keep and failure paths (`:725`, `:731`, `:752`), and `library_screen.py:36244-36247` is a shim that forwards to it (2 matches: the def and its call; checked after the restack onto dev ecad404bd4, review r14cl-rev1). TASK-40328 routed only the entry loader through `_sync_library_collections_capture_panes` (`recompose_capture_panes`) and left the rest: "Other Collections actions still use the whole-screen helper (out of scope)."
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Collections actions that change only the panes' presentation repaint the panes in place, including the three legacy recovery export paths in library_file_export.py; each remaining whole-screen recompose is listed with its reason
- [x] #2 A test counts compose_content passes for a representative set of actions (capture, annotate, scope change, delete), at most one each, red before the change
- [x] #3 UI-loop CPU per action is measured before and after
<!-- AC:END -->

## Implementation Plan

1. Measure at base cc6006fcf5: a probe (pytest plugin outside the worktree, run through the wrapper) presses each Collections action with Enter and records Library `compose_content` passes, UI-loop `time.thread_time()` from press to settled, and focus afterwards. Base (load ~50): one whole-screen recompose per action (two for a capture), 322-558 ms; focus stays on the same control, or lands on `library-ingest-top-button` when the control left the tree (capture, delete).
2. Red test first: one bootstrap-profile node in `Tests/UI/test_library_collection_reader_journeys.py`, parametrized over capture, annotate, scope change, delete (and one legacy-recovery export path), counting `compose_content` from the Enter until settled.
3. Fix it once in the shared helper: `_refresh_library_collections_capture_reader` queues the in-place pane sync on the screen with `call_next` (the same slot `refresh(recompose=True)` used, so it still runs once the handler returns), instead of recomposing the screen. Every caller, the export paths included, goes through it unchanged. The whole-screen recompose stays only as the fallback when the panes are not mounted.
4. Serialize pane syncs (the entry worker's and the actions') with one lock on the controller, so two recomposes of one pane never interleave.
5. Keep focus as today: when the focused control's pane is rebuilt, focus the same id if focusable, else the content's first focusable control (the whole-screen restore's rule).
6. Re-measure with the probe, run the covering Collections files, size ratchets, ruff, preflight; append the census node with its serial time.

## Implementation Notes

- **Premise held at base cc6006fcf5** (2026-11-15). A probe (a pytest plugin outside the worktree that replaces a bootstrap-profile node's body, run through the wrapper) pressed 15 Collections actions with Enter and counted Library `compose_content`: one whole-screen recompose per action (two for scope-all or a capture in some launches), since each handler's several `refresh(recompose=True)` calls coalesce into one recompose after it returns.
- **Fix, in the shared helper.** `_refresh_library_collections_capture_reader` now captures the annotation draft and queues `_sync_library_collections_capture_panes` (TASK-40328's in-place repaint) on the screen with `call_next`. That is the slot `refresh(recompose=True)` used, so the repaint still lands once the current handler returns, on the screen's pump, where `pilot.pause()` sees it. All 35 controller call sites, the three legacy recovery export paths in `library_file_export.py` and the screen shim go through it unchanged. The old whole-screen body is `_recompose_library_collections_screen`, which the sync calls only as its fallback. One `asyncio.Lock` on the controller serializes pane syncs, so the entry worker's and an action's never recompose one pane at the same time: two interleaved `pane.recompose()` calls would mount duplicate ids.
- **Focus kept as today.** When a rebuilt pane held focus, `recompose_capture_panes` focuses the same id if it is focusable, else `screen._first_focusable_in_content()`, the whole-screen restore's rule. Probe, base against head, 15 actions: the same focus after every one (the same control, or `library-ingest-top-button` after a capture or a delete removed the pressed control). A pane whose presentation did not change is no longer rebuilt, so an unrelated action no longer resets an unsubmitted filter box or focus on a grip.
- **Remaining whole-screen recomposes** (AC#1): (1) `_recompose_library_collections_screen`, the sync's fallback when the Items or Work pane is not mounted. There is nothing to patch in place, and it is still limited to the Collections route. (2) The route change into Collections (`_select_library_rail_row_after_source_admission`'s `recompose()`, one per rail press). It swaps the canvas and is not a Collections action. `LibraryRail.sync_state` can still recompose the rail itself, not the screen, when the rail's tree shape changes.
- **Test** (AC#2): `test_collections_actions_repaint_their_panes_without_a_library_recompose` (`Tests/UI/test_library_collection_reader_journeys.py`, bootstrap profile) runs a scope change, annotate, capture, delete and a legacy recovery export (the FileSave callback's completion path) in one launch. For each it counts `compose_content` until the panes show the result and checks focus as above. Base already made one recompose per action, so "at most one" would have been green there; the test pins zero instead, which still meets the criterion. Red at base: the parametrized first version failed 5 of 5 with "1 Library recomposes for one <action>", and the single-launch version fails at its first action. With only the compose assertion disabled at base, its focus checks passed 5 of 5.
- **Measured** (AC#3): the same probe, base and head back to back (UI-loop `time.thread_time()` from Enter to settled, `gc.collect()` first). At load 66-88 (2 base and 3 head launches): base 318-515 ms per action (median 358), head 123-177 ms (median 147), 0 recomposes. Capture went from 406-415 to 150-152 ms, delete 366-390 to 158-167, annotate 349-350 to 140-148, scope change 348-361 to 127-141. At load 9-39: base median 329 ms, head median 156 ms. One of three base launches failed inside the probe in each set (base only; not investigated).
- **Covering runs** (`-n 4`; the plain Library files ran under a temporary `bootstrap_profile` marker that a probe plugin adds at collection, the TASK-40328 recipe without editing the files, for the local RecoveryRequired trap): the seven Collections UI files, `test_library_entry_compose_once.py`, `test_library_rail_profile_admission.py`, `Tests/Widgets/Library/test_library_rail.py`, `test_library_export_characterization.py`, `Tests/Library/test_library_file_export.py`, `test_base_app_screen_tab_region_routes.py` and `Tests/Architecture/test_base_app_screen_recompose_focus_seam.py`. Head (load ~100): 14 failed, 295 passed, 1 error. 10 failures and the error are also red at base with the same command: the four 80x24 compact journeys, crit8 `test_rail_count_never_falls_back...`, three entry-compose nodes, export-selected-media, the legacy-export FileSave test (`SelectOverlay`) and the MeetingsScreen error. Head-only failures, rerun: `[WorkflowsScreen]` tab routes passed; entry-compose `[pending-conversations-size1]` failed 3 of 4 at head and 2 of 4 at base (a pre-existing flake); export-selected-conversations passed; characterization `test_quick_capture_refresh_hard_delete_cancel_and_legacy_recovery_close` passed 13 of 14 at head (its one failure was in the 4-wide full run at load ~100, an assertion one `pilot.pause()` after `Button.press()`) and 5 of 5 at base. Also: `test_library_shell.py -k "reconcile or snapshot or graduat or lifecycle or collections or restore_focus or focus_after"`: 48 passed, 2 lifecycle-graduation failures that also fail at base; the whole of `test_library_crit9_shell.py` after TASK-41130's fix: 25 passed. Ratchets and `test_library_collections_wiring.py`: 84 passed plus the controller pin below. Ruff gate OK; preflight passed.
- **Lane.** New census node (tail), `MINIMUM_FILES` 2145 -> 2146: 40.5 s call / 65.6 s serial at load 84-86, 21.9 s call at load ~46 (`-p no:xdist --timeout=180 --durations=0`). One launch instead of five parametrized ones: the five-launch version took 86.8 s of call at the same load.
- **Size.** `library_collections_controller.py` 1720 -> 1733 (measured); pin raised by exactly that with a dated comment.
- Files: `tldw_chatbook/UI/Library_Modules/library_collections_controller.py`, `tldw_chatbook/Widgets/Library/library_collections_capture_reader.py`, `Tests/UI/test_library_collection_reader_journeys.py`, `Tests/Architecture/test_library_modules_size_ratchet.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.
