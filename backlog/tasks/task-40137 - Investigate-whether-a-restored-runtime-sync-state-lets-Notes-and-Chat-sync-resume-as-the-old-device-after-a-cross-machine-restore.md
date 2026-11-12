---
id: TASK-40137
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
- `Sync_Interop/local_first_sync_service.py` uses the same decorator. `LocalFirstSyncService.sync_once` (`:70-71`) syncs Notes, Chat and Personal Context in one call.
- Activation approval is per owner and generation (`Backup_Recovery/activation.py:351-384`), so every Sync V2 call needs both `config` and `runtime.sync_state` approved.
- That includes the Personal Context link calls (`Personal_Context/link_service.py:170`, `:730` → `server_sync_service.py:710-712`, `:811-813`).

**Unknown.**
- What approves `config` and `runtime.sync_state` for a restored generation. No approver naming either was found; the approvers found are Notes, RAG, Skills, MCP and models, each for its own owner.
- Whether that approval is a plain approval. After a cross-machine replace restore, with credentials entered again, Notes and Chat sync would then push and pull with the restored device id and cursors. The restored device would act as the old one, and two devices could share one identity.
- Whether any Notes or Chat sync path reaches the server without going through `_sync_call`.

**Why a ruling is needed.** A fresh Personal Context first link after a restore cannot pass `_sync_call` unless something approves `config` and `runtime.sync_state`. Approving them must not let Notes and Chat resume as the old device.

**Scope.** This task produces findings and a ruling. It does not implement a gate, a rebind or a review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The task notes record, with file:line evidence, which Notes and Chat sync paths go through the `_sync_call` activation gate and which do not, and what approves `config` and `runtime.sync_state` after a restore
- [ ] #2 The task notes record, with a test or probe run through the scratch wrapper, whether Notes and Chat sync send the old device id and cursors once that gate is approved after a cross-machine replace restore
- [ ] #3 The notes give the ruling: who may approve `config` and `runtime.sync_state` after a restore, and through what review, such that Notes and Chat sync never resume as the old device after a cross-machine restore, and how a fresh Personal Context first link then passes `_sync_call`; where the ruling needs an owner decision, that decision is recorded verbatim with its date
- [ ] #4 If current behaviour lets Notes and Chat resume as the old device, the notes propose the guard (gate, rebind or reset after restore) as a follow-up; if not, the guard that prevents it is named and a test pins it
<!-- AC:END -->
