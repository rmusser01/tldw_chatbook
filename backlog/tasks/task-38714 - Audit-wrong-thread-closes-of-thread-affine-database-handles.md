---
id: TASK-38714
title: "Audit wrong-thread closes of thread-affine database handles"
status: To Do
assignee: []
created_date: 2026-11-04 00:59
labels:
- follow-up
- database
- threading
- bug
priority: medium
references:
- tldw_chatbook/Backup_Recovery/participants.py
- tldw_chatbook/Notes/file_notes_replica.py
- tldw_chatbook/Workflows/authoring.py
- tldw_chatbook/DB/Workflows_DB.py
- tldw_chatbook/DB/store_kernel.py
dependencies:
- TASK-36531
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 6 of the 2026-10-23 follow-up burn-down (TASK-36531's fix). Checked on dev c2cdb6b82e, 2026-11-03.

**The trap.** `FileNotesReplica.close()` (`Notes/file_notes_replica.py:130-139`) closes through `_core_closing` (`Backup_Recovery/participants.py:464`), which b5251e9a6e (TASK-32628) added. For a registered repository type, `_core_closing` refuses a close made from any thread other than the connection's resource thread (:477-493). The close then returns with the connection still open. It raises nothing and logs nothing. TASK-36531 traced a replica left open at quit to this: the replica was opened on one default-executor thread (`asyncio.to_thread`) and closed on another. It fixed that one seam by giving the workspace a one-thread executor for both. Nobody has checked the other closes.

**Thread-affine handles.** Besides the replica, `_core_closing` guards the closes in `ChaChaNotes_DB.py`, `Client_Media_DB_v2.py`, `Prompts_DB.py`, `Library_Ingest_Jobs_DB.py`, `Evals_DB.py`, `Dreams_DB.py`, `Subscriptions_DB.py`, `store_kernel.py`, `app_collections_wiring.py` and `Chat/console_change_review_read.py`. Thread-local handles have a related problem: `store_kernel.py`'s `close()` (:287-298) closes only the calling thread's connection, so a close made from a pool thread leaves the opener's connection open.

**Closes routed through a pool thread.** The audit starts from these sites:
- `asyncio.to_thread(<x>.close)` in `app_lifecycle.py` (:213, :400, :427, :1112), `Chat/console_runtime.py` (:6846), `Chat/console_voice_attempts.py` (:858), `Chat/console_send_diagnostics.py` (:235) and `Audio/voice_transcription.py` (:190, :270);
- `Workflows/authoring.py`, which is one candidate. It opens `WorkflowsDB` with `asyncio.to_thread(WorkflowsDB, ...)` (:57) and closes it with `asyncio.to_thread(self._db.close)` (:168). On a read, `WorkflowsDB.close()` (`DB/Workflows_DB.py:76`) is a plain close under an `RLock` on a `check_same_thread=False` connection, and `WorkflowsDB` is not among `_repository_types()`. So it may turn out not to be thread-affine; the audit should record that either way.

`Dreams_DB.close_all()` (:410) is the explicit all-threads close for a handle its caller owns outright, so it is out of scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every close of a thread-affine database handle that can run on a thread other than the one that opened it now runs on the opening thread, or the task notes show why that handle is not thread-affine; the notes list every call site checked
- [ ] #2 Each close that is fixed has a test that fails on the base build when the open and the close land on different threads (for example with a thread-per-call executor, as TASK-36531's test does)
<!-- AC:END -->
