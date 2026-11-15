---
id: TASK-41110
title: "Actor Pack import runs on asyncio.to_thread and never closes that thread's cached ChaChaNotes connection, which may hold a later backup drain"
status: To Do
created_date: 2026-11-15 02:20
labels:
- follow-up
- actor-packs
- backup
priority: medium
references:
- tldw_chatbook/Actor_Packs/import_controller.py
- tldw_chatbook/Backup_Recovery/participants.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** A capture or restore pause waits for every native ChaChaNotes connection lease. If an idle executor thread keeps the connection an import opened, a backup started later can wait on it.

**Evidence.**
- `import_controller.py:246-248` runs inspect and activate through `asyncio.to_thread(execute)`, and `:350` runs cleanup the same way. Neither closes the thread's cached connection after the work.
- TASK-40304's notes: "the worker thread's own ChaChaNotes connection lease keeps a drain waiting"; its in-flight test closes that connection by hand in its publish hook.
- Other services close theirs after the thread work: `Writing_Interop/writing_scope_service.py:183-192` and `Research_Interop/research_scope_service.py:235-244` run `close_settled_core_cache` (`Backup_Recovery/participants.py`) after `to_thread`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A test completes an Actor Pack import, then begins a capture pause and records whether the drain completes
- [ ] #2 If it waits, the import releases its thread's cached connection when the work ends, as the interop services do, and the test is red before the fix
<!-- AC:END -->
