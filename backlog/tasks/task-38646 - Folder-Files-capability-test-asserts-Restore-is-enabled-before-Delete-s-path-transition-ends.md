---
id: TASK-38646
title: Folder Files capability test asserts Restore is enabled before Delete's path transition ends
status: To Do
created_date: 2026-11-02 21:52
labels:
- follow-up
- tests
- library
priority: low
references:
- Tests/UI/test_library_file_notes_workspace.py
- tldw_chatbook/Widgets/Library/library_file_notes_workspace.py
dependencies:
- TASK-36383
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while filing the round-3 follow-ups of the 2026-10-23 burn-down (dev 88af9f9379, 2026-11-02). `Tests/UI/test_library_file_notes_workspace.py::test_folder_files_capability_inventory_and_modes` failed 2 of 6 serial runs on dev. Each run was its own pytest call through the scratch-profile wrapper. Both failures were at :1886 with `restore.disabled=True`. That is the same rate TASK-36383's notes recorded on 2026-10-24, and TASK-36386 noted this test too.

**Cause.** The test waits only for `_selected_deleted_path == "working.md"` (:1880), then asserts Restore is shown and enabled (:1886). Delete sets that field inside `_hold_path_transition()` before it awaits the rescan (library_file_notes_workspace.py:8927, :8933). Restore stays disabled while the transition runs (:5770, :5843-5845), and the transition's `finally` re-enables it (:6141-6150). TASK-36383 fixed this race in the file's sibling tests by waiting for a pressable Restore, but not here.

No PR lane runs the file: it has no UI census entry, and Nightly Deep last ran on 2026-09-27. So far it only fails local whole-file runs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test passes 10 of 10 serial runs on dev, each its own pytest process, because it waits for a shown and enabled Restore
- [ ] #2 It still fails if Restore stays hidden or disabled after Delete settles (shown under a temporary source edit)
<!-- AC:END -->
