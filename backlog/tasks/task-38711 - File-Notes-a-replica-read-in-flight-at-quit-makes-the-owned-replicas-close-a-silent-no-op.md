---
id: TASK-38711
title: "File Notes: a replica read in flight at quit makes the owned replica's close a silent no-op"
status: Done
assignee:
- '@claude'
created_date: 2026-11-04 01:05
updated_date: 2026-11-04 01:19
labels:
- library
- file-notes
- bug
priority: medium
references:
- tldw_chatbook/Widgets/Library/library_file_notes_workspace.py
- tldw_chatbook/Notes/file_notes_replica.py
- tldw_chatbook/Backup_Recovery/participants.py
- Tests/UI/test_library_file_notes_workspace.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-36531 made File Notes close its owned replica at quit, on the thread that opened it. The workspace still reads that replica directly in two places, outside the service's operation lock: the Search box (`replica.search`) and the deleted-files list (`replica.list_deleted`). Both run on default-pool threads.

A file-backed replica's close refuses, without raising, while any core operation on it is in flight (`_core_closing` in `Backup_Recovery/participants.py`). So one of those reads in flight at quit turns the close into a silent no-op, and the SQLite connection leaks: the same leak class TASK-36531 fixed for the normal path. The reverse also matters. A closed file-backed replica reopens on its next use, so a read that starts after quit has closed it opens a new connection that nothing will close.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Quit waits for, or routes around, a direct replica read in flight, so the owned replica ends closed: its connection attribute is None and the native handle it had raises on use
- [x] #2 A deterministic test holds a direct read inside its replica operation across quit and is red on the base build
- [x] #3 No direct read that starts after shutdown has begun can reopen the owned replica; pinned by a test that is red on the base build
<!-- AC:END -->

## Implementation Plan

1. Write the test first: an owned file-backed replica, a default executor that runs each call on a new thread, a search held inside its replica operation across `shutdown()`, then the TASK-36531 proof (connection None, the retired native handle raises). Add a second case: after quit, a deleted-files read with the captured replica must not reopen it. Show both red.
2. Route both direct reads through one helper. When the workspace owns its replica, the read runs on the replica's one owner thread, so the close queues behind any read in flight. A read that starts once shutdown has begun is skipped instead of run, so it cannot reopen the closed replica.
3. Run the new tests green, plus the covering File Notes tests, and compare any failure against base.

## Implementation Notes

**Approach.** The two direct reads, `replica.search` in `_run_search` and `replica.list_deleted` in `_load_deleted_paths`, now go through one helper, `_read_replica`.
- When the workspace owns its replica, the read runs on the replica's one-thread owner executor, the same thread that opens and closes it. Quit's close is submitted to that thread too, so it queues behind any read in flight, and the read's core operation has always ended before `_core_closing` looks.
- A read that starts once `shutdown()` has begun returns None without touching the replica. It checks before submitting (so it never meets the shut-down executor) and again on the thread (so a read queued before quit but run after the close is skipped).
- An injected replica (no owner thread) keeps `asyncio.to_thread`.
- I did not take the service lock instead. That would have queued every search behind scans and reconciles, including a scan wedged on a dead mount.

**Tests** (`Tests/UI/test_library_file_notes_workspace.py`):
- `test_quit_closes_owned_replica_behind_a_search_in_flight` uses `_ThreadPerCallExecutor` and the TASK-36531 proof: the connection is None and the retired native handle raises. The search is submitted before quit and enters its replica operation once quit's retention pass is done (that pass needs the replica's lock), and retention waits until the search is inside. The read is released once the close has returned, or once it has been submitted while the read holds the owner thread. So neither ordering depends on the pool.
- `test_read_started_after_quit_cannot_reopen_owned_replica`: after quit, `_load_deleted_paths` with the captured replica (as `_initialize` holds it across its scan) leaves the connection None.
- Red on base: both failed at `assert owned_replica._connection is None`. In the first the close was refused and the connection stayed open; in the second the read reopened it. Green: 2 passed.
- Covering run of 12 File Notes / Library Notes files (`-n 4`, load average ~26): 56 failed, 578 passed. All 56 are in the set that fails on base with the same command (TASK-32185's authority/task-return reds, `test_prepare_session_fixed_regions_remain_visible_at_40_by_20`, 12 push-file tests and 37 Files-sync journey tests).

**Files.** `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py`, `Tests/UI/test_library_file_notes_workspace.py`, `Tests/Architecture/test_module_size_ratchet.py` (9599 -> 9615, dated).
