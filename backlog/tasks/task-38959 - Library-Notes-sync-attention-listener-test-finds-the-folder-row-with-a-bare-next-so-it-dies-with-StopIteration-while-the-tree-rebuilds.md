---
id: TASK-38959
title: "Library Notes sync-attention listener test finds the folder row with a bare next(), so it dies with StopIteration while the tree rebuilds"
status: Done
assignee:
- '@claude'
created_date: 2026-11-05 21:35
updated_date: 2026-11-06 16:13
labels:
- follow-up
- tests
- flaky
- library
- notes
priority: low
references:
- Tests/UI/test_library_notes_sync_attention_listener.py
- Tests/UI/test_library_notes_sync_attention.py
- Tests/UI/test_library_notes_sync_delete_restore.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05.

The reported flake was the held-folder test, `Tests/UI/test_library_notes_sync_attention.py::test_a_held_sync_folder_reads_needs_attention_until_recovery_heals_it`, and it no longer reproduces. It passed 23 of 23 runs: 3 alone at load 20-34, 10 whole-file `-n 4` runs at load 34-70, and 10 runs alongside the readiness and listener files at load 48-82. It was fixed on 2026-10-28 by 67406398ad ("wait for the exact live sync folder", which replaced a bare `next()` with `_live_button`) and 0024e7f27d.

Its sibling file still has the bug that commit removed:
- **The test:** `Tests/UI/test_library_notes_sync_attention_listener.py::test_a_watcher_driven_hold_turns_the_three_surfaces_without_a_keypress`.
- **Failures:** 1 of 10 combined `-n 4` runs and 1 of 6 serial runs alone, at load 48-61. The traceback runs `listener.py:120: in <lambda>`, `:52: in _folder_row`, `StopIteration`, ending in `:116 RuntimeError: coroutine raised StopIteration`.
- **Cause, from the traceback:** `_folder_row` (`:51-56`) still uses a bare `next(... if "VSync" in label)`. It runs inside the `_until` predicates (`:79`, `:120`) and directly (`:84`), and raises while the tree rebuilds and the row is briefly gone. `:95` has the same bare `next()` for the note row.
- **Lanes:** the file runs on no PR lane; its docstring says the lane is near its cap. The listener file last changed on 2026-10-04 (e25cb5f557).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The listener test passes 20 of 20 runs at a recorded load of at least 40, and its row lookups tolerate a briefly missing row the way the sibling file's have since 67406398ad
- [x] #2 It still fails on an assertion when the runtime status listener is disabled (a temporary source edit making ensure_library_notes_sync_attention_listener a no-op)
- [x] #3 If the file joins a PR lane, its measured whole-file time is recorded in the notes
<!-- AC:END -->

## Implementation Plan

1. Replace the listener test's bare `next()` row lookups (the folder row and the note row) with the sibling file's `_live_button` helper (67406398ad), so a predicate polled while the tree rebuilds reads None instead of raising `StopIteration`, and each press waits for a live row.
2. Negative control: a temporary source edit making `ensure_library_notes_sync_attention_listener` a no-op must turn the test red on an assertion (a timed-out wait), then revert.
3. 20 of 20 runs at a recorded load of at least 40. The file stays off the PR lanes (AC#3 is conditional; its docstring keeps the reason).

## Implementation Notes

- **Fix.** The listener test now finds its rows the way the sibling file has since 67406398ad:
  - `_folder_row` is `_live_button(screen, LIST, ".library-notes-folder-row", _is_vsync)`, so while the tree rebuilds it reads None instead of raising `StopIteration`.
  - The two polled predicates go through `_folder_row_reads`, which treats None as "not yet".
  - The folder-row and note-row presses go through the sibling's `_press`. It waits for a displayed, enabled row on a settled canvas with no Notes mutation fenced.
  - No bare `next()` remains in the file.
- **AC#2 (negative control).** A temporary `return` at the top of `ensure_library_notes_sync_attention_listener`, since reverted, gave 1 failed in 35.7 s: `AssertionError: timed out waiting for the tree row to say the folder needs attention`.
- **AC#1.** 20 of 20 at `-n 4`, with a scratch `-p` repeat plugin (not tracked). There are four 20-run batches, 80 of 80 in all. Loads are 1-minute averages on 18 CPUs, sampled every 5 s.
  - 60.1 s at load 53.3 falling to 37.3. The load was 41.4 or more for the first 50 of the 62 s.
  - 89.8 s at load 31.6 rising to 55.7. It was 40.7 or more from 25 s in.
  - 68.3 s at load 19.1-39.0.
  - 44.8 s at `-n 8`, at load 33.5-37.5.
- **AC#3.** The file stays off the PR lanes, so there is no lane time to record. Its docstring keeps the reason.
- Files: `Tests/UI/test_library_notes_sync_attention_listener.py`.
