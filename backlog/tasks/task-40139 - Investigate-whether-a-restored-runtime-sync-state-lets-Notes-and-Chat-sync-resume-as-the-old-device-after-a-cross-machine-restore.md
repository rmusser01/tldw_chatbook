---
id: TASK-40139
title: "Investigate whether a restored runtime.sync_state lets Notes and Chat sync resume as the old device after a cross-machine restore"
status: To Do
created_date: 2026-11-10 17:13
labels:
- follow-up
- security
- backup
- recovery
priority: medium
references:
- tldw_chatbook/Sync_Interop/recovery.py
- tldw_chatbook/Sync_Interop/sync_state_repository.py
- tldw_chatbook/Backup_Recovery/activation.py
- Docs/superpowers/specs/2026-11-10-personal-context-backup-key-custody-design.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 13 of the 2026-10-23 burn-down from TASK-39148's owner-approved design, `Docs/superpowers/specs/2026-11-10-personal-context-backup-key-custody-design.md`. Owner decision 2026-11-10 (AskUserQuestion): "Approve all 11". Checked on dev ae9f9e8d21, 2026-11-10; line numbers are dev's.

**Why.** TASK-39148's design requires that a restored device never act as the old device towards the server (spec constraint 1). It could not trace whether general Sync honours that.

What is captured:
- Backup captures `tldw_chatbook_sync_state.db` as `runtime.sync_state` (`Sync_Interop/recovery.py:79-90`).
- That includes the shared `sync_profile_state.device_id`, dataset cursors, and the Sync V2 outbox and remote heads (`:39` and the schema at `:21-46`).

What restore does and does not do:
- Every restore mints a new `installation_id` (`Backup_Recovery/isolated_restore.py:96-98`, `replacement.py:1629-1630`).
- But no consumer gates sync on `execution_allowed(("runtime.sync_state",))`. The current consumers are `app.py`, `app_lifecycle.py`, `app_service_wiring.py`, Dreams and RAG; none is for sync.
- Server credentials come back into fresh recovery scopes when remappable (`credentials.py:1231-1302`), or the user enters them again.

**Unknown.** After a cross-machine replace restore with credentials entered again, do Notes and Chat Sync V2 push and pull with the old device id and old cursors? If so, the restored device acts as the old one, and two devices may share one identity.

TASK-40137 rebinds the Personal Context side of that shared row, and its notes say what that does to Notes and Chat.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The task notes record, with file:line evidence and a test or probe run through the scratch wrapper, whether Notes and Chat sync after a cross-machine replace restore send the old device id and cursors
- [ ] #2 If they do, the notes propose the guard (gate, rebind or reset after restore) for the owner to decide, and the proposed follow-up is listed; if they do not, the guard that prevents it is named and a test pins it
- [ ] #3 The finding is checked against TASK-40137's Personal Context device rebind, and any conflict is recorded
<!-- AC:END -->
