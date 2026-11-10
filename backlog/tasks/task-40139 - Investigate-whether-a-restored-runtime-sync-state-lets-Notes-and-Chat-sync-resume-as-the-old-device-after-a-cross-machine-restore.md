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

What restore does:
- Every restore mints a new `installation_id` (`Backup_Recovery/isolated_restore.py:96-98`, `replacement.py:1629-1630`).
- Every restore binds all activation-required owners (`Backup_Recovery/publication.py:592-604`), including `runtime.sync_state`.
- Server credentials come back into fresh recovery scopes when remappable (`credentials.py:1231-1302`), or the user enters them again.

What is gated:
- Server Sync V2 calls are gated after a restore. `_sync_call` holds `execution_scope(("config", "runtime.sync_state"), …)` and raises `sync_activation_required` (`Sync_Interop/server_sync_service.py:51-95`, `:85-88`), checked with `git show ae9f9e8d21:…`.
- It wraps `send_changes`, `get_changes`, `run_v2_dry_run`, `push_v2_envelopes`, `pull_v2_envelopes`, the Notes organization bootstrap, the Personal Context link calls, and the conflict and recovery-bundle calls.
- `Sync_Interop/local_first_sync_service.py` uses the same decorator.

**Unknown.**
- What approves `config` and `runtime.sync_state` for a restored generation. No approver naming either was found; the approvers found are Notes, RAG, Skills, MCP and models, each for its own owner.
- Whether that approval is a plain approval. After a cross-machine replace restore, with credentials entered again, Notes and Chat sync would then push and pull with the restored device id and cursors. The restored device would act as the old one, and two devices could share one identity.
- Whether any Notes or Chat sync path reaches the server without going through `_sync_call`.

TASK-40137 rebinds the Personal Context side of that shared row, and its notes say what that does to Notes and Chat.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The task notes record, with file:line evidence, which Notes and Chat sync paths go through the `_sync_call` activation gate and which do not, and what approves `config` and `runtime.sync_state` after a restore
- [ ] #2 The task notes record, with a test or probe run through the scratch wrapper, whether Notes and Chat sync send the old device id and cursors once that gate is approved after a cross-machine replace restore
- [ ] #3 If they do, the notes propose the guard (gate, rebind or reset after restore) for the owner to decide, and the proposed follow-up is listed; if they do not, the guard that prevents it is named and a test pins it
- [ ] #4 The finding is checked against TASK-40137's Personal Context device rebind, and any conflict is recorded
<!-- AC:END -->
