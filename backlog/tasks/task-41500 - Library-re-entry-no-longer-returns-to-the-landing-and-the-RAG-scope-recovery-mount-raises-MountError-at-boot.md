---
id: TASK-41500
title: "Library re-entry no longer returns to the landing and the RAG scope recovery mount raises MountError at boot"
status: Done
assignee: ['@claude']
created_date: 2026-11-17 05:27
updated_date: 2026-11-18 03:06
labels:
- follow-up
- library
- ui
priority: high
references:
- tldw_chatbook/UI/Screens/library_screen.py
- Tests/UI/test_screen_navigation.py
dependencies:
- TASK-41127
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41127 (libtests, round 15) and left red as product defects. Library screen reuse (7992124e8d) skips `restore_state` (`library_screen.py:10650`), so re-entering Library no longer returns to the landing. Separately `_mirror_library_rag_scope_recovery` (`library_screen.py:37292`) mounts into `#library-rag-source-scope` before it is mounted and the worker raises `MountError`.

**Evidence.** `test_generic_reentry_returns_to_library_landing` and `test_library_screen_round_trip_returns_to_landing_with_rag_draft` fail 3 of 3 (twice on `'browse-search' == ''`, once on the RAG `MountError`). `test_boot_with_search_default_tab_lands_on_library_rag_canvas` fails on the `MountError`. Turning reuse off makes the first test pass (41127 notes). Re-confirmed in the libtests fix round.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Leaving Library and returning through any route lands on the Library landing, not on the previous search or rail state, unless a route explicitly asks to restore
- [x] #2 Booting with the Search default tab reaches the RAG canvas with no `MountError` from the scope recovery worker
- [x] #3 The three named nodes pass, with the red-before evidence recorded
- [x] #4 A reuse path that skips `restore_state` is either removed or has a test pinning what it intentionally keeps
<!-- AC:END -->

## Implementation Plan

1. Confirm both defects at base 45b1949ca3 (the three named nodes, through the wrapper).
2. Reuse (AC#1, AC#4): 7992124e8d (TASK-31521) made Library reusable for speed (warm switch 33 ms CPU vs 340-1,540 ms fresh), so keep reuse. On a reused instance with no navigation context, the app calls a new optional screen hook before the switch; Library's hook takes the landing through the same rail seam Esc-to-hub uses, so every switch rule and veto still applies. Deep links keep their own canvas.
3. MountError (AC#2): probe the worker to find what detaches the container, then fix the order where the scope recovery block mounts, without catching MountError.
4. Tests: one new reuse-contract node pinning what reuse keeps (same instance, RAG draft) and that re-entry shows the landing; one deterministic node for the MountError window. Red at base, green after. Re-point covering reuse tests that relied on generic re-entry keeping the row.
5. Lane the new nodes and the three named nodes at the census tail, measured serially with the lane flags. Size ratchets, preflight, User Guide check.

## Implementation Notes

**Approach.** Both defects reproduced at base 45b1949ca3: the two re-entry nodes failed on `'browse-search' == ''`, and all three raised the scope recovery `MountError` at teardown.

- **Reuse is kept (AC#4).** 7992124e8d (TASK-31521) made Library reusable for speed: a warm switch cost 33 ms CPU against 340-1,540 ms for a fresh one, and per-visit config reads and SQLite connections stopped. Generic Library entries need that speed most, so removing reuse was the wrong trade. The app (`app_navigation.py`) now calls an optional `apply_generic_reentry()` on a reused screen when the route has no context, in the same place a deep link gets `apply_navigation_context`, before the switch. Library's hook (`library_screen.py`) takes the landing through `_select_library_rail_row("")`, the seam Esc-to-hub uses, so every switch rule and veto applies. `on_screen_resume` then sees the landing row and kicks only the landing's surfaces. Deep links keep their own canvas. Home and Console have no hook and are unchanged.
- **What reuse keeps on purpose**, pinned by `test_reused_library_reentry_shows_the_landing_and_keeps_its_live_state`: the same instance and its live state (the RAG draft). The test also checks that the landing canvas is on screen and the RAG panel is gone, not only the row id. The round-trip node pins that the draft comes back in the reopened Search canvas.
- **MountError (AC#2): the brief's diagnosis did not hold.** A probe in the mirror showed `#library-rag-source-scope` and every ancestor mounted and attached when the worker started. At the `mount` call, after the awaited `remove()` calls, `app._exit` was True. `TldwCli._shutdown` sets that fence first and then spends seconds in lifecycle drains. Textual's `mount` raises "Can't mount widget(s) before X is mounted" whenever `is_attached` is False, and `_exit` makes every node read as detached. So waiting for the container to mount would change nothing: it was mounted. The fix is a liveness check after the awaits in `_apply_library_rag_scope_recovery_block`, the code both the mirror and the full refresh run: `if not scope_container.is_attached: return` before each mount. Nothing catches `MountError`; any other mount failure still raises. A pruned container never raised here, because `mount` already returns without error while pruning. Lesson added to `backlog/docs/lessons-textual.md`.
- **Re-pinned covering test.** `Tests/UI/test_library_screen_reuse.py::test_suspended_library_gates_ingest_dom_work_until_resume` returned with ctrl+3 and relied on that bare return resuming on the Ingest row. It now returns through the explicit Import route (`LIBRARY_NAV_CONTEXT_INGEST`). That file's async tests cannot run locally, because their own `_scratch_env` trips `RecoveryRequired` (red at base too, and not laned). I checked the journey with a temporary bootstrap-profile copy, since deleted. After the fix, the bare return left the ingest reconciliation at 0 calls, and the Import route reached it and cleared the latch. The copy could not check the test's snapshot assertion: `_build_test_app` has no source services, so the error-severity load failure skips the snapshot refresh on both routes, at base as well.

**Red/green.**
- `test_reused_library_reentry_shows_the_landing_and_keeps_its_live_state` was red at base on `assert 'browse-search' == ''`.
- `test_scope_recovery_mirror_mounts_nothing_once_the_app_is_exiting` was red at base on `AssertionError: the mirror mounted after the exit fence: Can't mount widget(s) before Vertical(id='library-rag-source-scope', ...) is mounted`.
- The three named nodes were red 3 of 3 at base (load 36-63). After the fix, the five nodes passed together at `-n 4` (load 79) and again serially (below).

**Covering runs (`-n 4`, load 60-78).** The run covered `test_screen_navigation.py` (whole file), both `test_console_runtime_ownership.py` Library round trips, `test_library_shell.py::test_library_shell_reused_screen_reentry_serializes_draining_retrieval` (run only), and these files whole: `test_library_entry_compose_once`, `test_library_phase_c_switch_residency`, `test_library_phase_c_resident_canvas`, `test_screen_residency`, `test_screen_reuse`, `test_screen_reuse_helpers`, `test_library_rag_scope_recovery`, `test_library_note_autosave_after_dialog`, `test_library_inspection_admission`, `test_library_artifact_navigation`, `test_character_navigation_screen_reuse`, `test_app_keep_alive_dead_screen`, `test_home_console_resume` and `test_library_screen_reuse`.
- Result: 214 passed, 124 failed, 43 errors. All 167 red nodes were rerun at base source.
- 164 were red there too. They are mostly the local `RecoveryRequired` trap, plus `_FakeMediaRowButton` (TASK-23149) and `test_on_screen_suspend_stops_every_timer_in_isolation` (`AttributeError: _artifacts_navigation`).
- The other three are known races outside the reuse path, and they flake at base as well. 3 rounds each:
  - `test_boot_with_prompts_default_tab...`: 2/3 red at base, 3/3 after (Prompts `Select` `NoMatches 'SelectOverlay'`).
  - `test_overlapping_navigate_requests_complete_in_fifo_order`: 1/3 red at base, 2/3 after (Workflows `NoMatches`).
  - `test_navigation_keypress_during_splash_is_safely_ignored`: 1/3 red at base, 0/3 after (timing).
- Size ratchets and the Library architecture pins (`test_screen_size_ratchet`, `test_module_size_ratchet`, `test_library_modules_size_ratchet`, `test_library_search_rag_wiring`, `test_library_cluster_membership_census`): 289 passed, serial.

**Census.** The five nodes above went at the tail (`MINIMUM_FILES` 2248). Serial lane-flag times (`-p no:xdist --timeout=180 --durations=0`), one process, load 97 falling to 51:

| Entry | s |
|---|---|
| `test_generic_reentry_returns_to_library_landing` | 18.1 |
| `test_library_screen_round_trip_returns_to_landing_with_rag_draft` | 17.9 |
| `test_boot_with_search_default_tab_lands_on_library_rag_canvas` | 10.1 |
| `test_scope_recovery_mirror_mounts_nothing_once_the_app_is_exiting` | 9.4 |
| `test_reused_library_reentry_shows_the_landing_and_keeps_its_live_state` | 11.0 |

Setup and teardown each took under 1 s.

**Ratchets (owner rule 2026-10-04, measured, dated notes).**
- `library_screen.py`: 38282 -> 38280 lines, 1379 -> 1380 methods.
- `app_navigation.py`: 1208 -> 1219.
- `library_rag_search_controller.py`: 2056 -> 2061.

**Diagnostic inventory.** One added statement, reviewed with `--statements` before `--write`: `logger.warning("Generic re-entry failed (route={}, exception_category={}).", current_tab_value, type(exc).__name__)`. It writes the canonical route id and the exception class name only, so no user content, secrets or paths reach the log. Files: `Docs/security/production-diagnostic-inventory.json`.

**Docs.** In `Docs/User_Guide/library.md`, a bare return (Ctrl+3, the nav bar, the palette) opens the landing and keeps the Search / RAG draft. `library/search-and-rag.md` loses its "if the landing page is shown" hedge.

**Files.**
- Source: `tldw_chatbook/app_navigation.py`, `tldw_chatbook/UI/Screens/library_screen.py`, `tldw_chatbook/UI/Library_Modules/library_rag_search_controller.py`.
- Tests: `Tests/UI/test_screen_navigation.py`, `Tests/UI/test_library_screen_reuse.py`, the three ratchet files.
- Census: `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.
- Docs: the two User Guide pages, `backlog/docs/lessons-textual.md`.

