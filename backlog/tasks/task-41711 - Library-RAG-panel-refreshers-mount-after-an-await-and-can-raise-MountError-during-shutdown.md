---
id: TASK-41711
title: "Library RAG panel refreshers mount after an await and can raise MountError during shutdown"
status: Done
assignee: ['@claude']
created_date: 2026-11-18 06:57
updated_date: 2026-11-19 16:39
labels:
- follow-up
- ui
- library
priority: medium
references:
- tldw_chatbook/UI/Library_Modules/library_rag_search_controller.py
- tldw_chatbook/UI/Screens/library_screen.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41500 (round 16). TASK-41500 guarded the RAG scope recovery mirror against mounting after the app's exit fence (`app._exit` makes every node read as detached, so Textual raises "Can't mount ... before X is mounted"). `_refresh_search_rag_panel_state_widgets` carries on after the scope block's early return into the answer, results and history remove/mount calls, and the query-status, answer, history and results refreshers have the same window between an await and a mount.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No RAG panel refresher mounts once the app's exit fence is set, and none swallows a mount error while the app runs
- [x] #2 A deterministic test sets the fence between the await and the mount and is red before the fix
<!-- AC:END -->

## Implementation Plan

1. In each RAG panel refresher that mounts after an await (query status, answer, history, results), check the mount target's `is_attached` before each mount, as TASK-41500 did for the scope recovery block. `is_attached` is False for every node once `app._exit` is set, and a pruned target already mounts nothing, so no mount error is caught or swallowed.
2. In `_refresh_search_rag_panel_state_widgets`, stop after the scope block once the screen reads detached, so it does not go on to remove the answer, results and history rows.
3. Add a deterministic test next to `test_scope_recovery_mirror_mounts_nothing_once_the_app_is_exiting`: hold an awaited removal in each refresher, set the fence, release, and assert no `MountError`. Show it red before the fix.

## Implementation Notes

Each RAG panel refresher now checks its mount target's `is_attached` right before it mounts, the TASK-41500 guard on the scope block. `is_attached` reads False for every node once `app._exit` is set, so past the exit fence a refresher mounts nothing. While the app runs nothing is caught: a pruned target already mounts nothing in Textual, and no `MountError` is swallowed.

- **Shape.** The query-status, answer, history and results refreshers used to mount their new children one at a time, each mount awaited, so the fence could land between any two of them. Each now mounts its children in one `mount(*children)` call (with the same `after=`/`before=` anchor, which keeps their order), behind a single check. That closes the window with one check per refresher and keeps `library_rag_search_controller.py` inside its size ratchet: it was at its pin, and the batching is line-neutral (2061 lines at base and head).
- **Plan step 2 dropped.** `_refresh_search_rag_panel_state_widgets` still goes on after the scope block. Past the fence the later refreshers only remove rows and then skip their mount, and removal past the fence works (the TASK-41500 test removes there), so a separate stop added nothing the AC asks for.
- **Test.** `Tests/UI/test_screen_navigation.py::test_rag_panel_refreshers_mount_nothing_once_the_app_is_exiting`, one case per refresher, next to TASK-41500's test. Each case stubs the region's builder so the region has one widget to remove and one to mount, mounts once while the app runs (the positive half), then holds the removal, sets `app._exit`, releases, and asserts no `MountError`.
- **Red/green.** With the source at base: 4 of 4 failed on the assertion, each naming its container ("Can't mount widget(s) before Vertical(id='library-rag-results' ...) is mounted", `library-rag-query-controls`, `LibrarySearchRagPanel`, `Contents()`). With the fix: 4 of 4 passed, with the TASK-41500 test (`-n 4`, load 34-37).
- **Covering runs.** The RAG test files (`Tests/UI/test_library_rag_*.py`, `test_product_maturity_gate16_library_search_rag.py`, `Tests/Architecture/test_library_search_rag_wiring.py`): the same 259 nodes failed at base and at head, the local `RecoveryRequired: raw_source_selection_changed` trap. Ten of those plain files were then run under a temporary module `bootstrap_profile` marker (removed afterwards): 194 passed, 4 failed, and the same 4 (`test_ready_disclosure_is_fully_painted_without_moving_run[rag-*-size1]`, a truncated quiet line) fail at base under the same marker. The two `test_library_rag_answer_names_model*` files, which mount real answers, results and history, passed.

- **Lane.** Like its TASK-41500 sibling, the node is a tail entry in `scripts/ui_pr_gate_census.txt` (floor 2279, shared with TASK-41715's node). Serially with the lane's flags at load 63-65 the two new nodes took 41.5 s for 5 cases (calls 8.4, 5.2, 4.9 and 4.6 s for this test's four).

**Files.** `tldw_chatbook/UI/Library_Modules/library_rag_search_controller.py`; `Tests/UI/test_screen_navigation.py`; `scripts/ui_pr_gate_census.txt`; `scripts/check_ui_pr_gate_census.py`.
