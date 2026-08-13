---
id: TASK-15781
title: Verify-then-trim NotesSyncService's SyncProfile CRUD surface
status: To Do
assignee: []
created_date: '2026-08-13 12:31'
labels:
  - cleanup
  - notes
priority: low
---

## Description

Verify-then-trim candidate surfaced by task-15481's disclosure (input-latency
burn-down's dead-code sweep). Task-15481 trimmed `Notes/sync_service.py`'s
dead auto-sync-loop machinery (`create_profile`, `start_auto_sync`,
`stop_auto_sync`, `stop_all_auto_syncs`) but explicitly left `SyncProfile`,
profile load/save, `get_profile`, `list_profiles`, and `sync_with_profile` in
place, on the grounds that they were "not named dead by the task and not a
crash-on-touch landmine" — i.e. task-15481 did not investigate whether
these remaining methods have production callers, only that they were out of
its own AC's scope.

Re-verified here by a repo-wide grep: `library_screen.py` is the sole
production importer of `NotesSyncService`, and it calls exactly one method
on the instance it constructs — `service.sync_folder(...)`. No production
code anywhere calls `NotesSyncService.get_profile`, `.list_profiles`,
`.sync_with_profile`, or constructs a `SyncProfile`. (Several *other*,
unrelated `get_profile`/`list_profiles` methods exist elsewhere in the
codebase — TTS profile managers, RAG profile managers, MCP local control —
none of which touch `Notes/sync_service.py`; they are not evidence against
this finding.)

## Acceptance Criteria

- [ ] Re-verify at implementation time that `SyncProfile`, `get_profile`,
      `list_profiles`, and `sync_with_profile` in `Notes/sync_service.py`
      have zero production callers (grep, not trusting this task's finding
      blind)
- [ ] If still dead: remove the CRUD surface (with git-log provenance
      recorded) and its now-orphaned test-only coverage, keeping
      `NotesSyncService.sync_folder` and everything it depends on untouched
- [ ] If a live caller is found: the task closes as "not dead," with the
      caller documented, and no removal happens
- [ ] `Tests/Notes/` and `Tests/Sync_Interop/` stay green; a final grep
      sweep for the removed names returns no production hits
