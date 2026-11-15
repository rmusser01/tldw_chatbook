---
id: TASK-40711
title: "A restored profile's settings get their own review, which turns model-catalog refresh and change tracking back on"
status: To Do
assignee: []
created_date: 2026-11-13 03:35
labels:
- follow-up
- backup
- recovery
- security
priority: medium
references:
- tldw_chatbook/Backup_Recovery/activation.py
- Tests/Architecture/test_backup_owner_inventory.py
- backlog/decisions/126-complete-local-backup-and-recovery.md
dependencies:
- TASK-40137
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down from TASK-40137's findings and owner ruling. Checked on dev 26b2e677dd, 2026-11-12.

**Why.** Every restored generation requires the shared `config` owner to be approved before its consumers run, and nothing in the app approves it (TASK-40137's notes list the six `ActivationStore.approve` call sites: MCP, `notes.file_notes`, plugins, RAG, models and skills). So after any restore, even onto the same device or after a rollback, model-catalog auto-refresh and native workspace change tracking stay off for good.

**Owner ruling, 2026-11-12 (TASK-40137):** "Rebind review + separate config review (Recommended)". `config` gets its own simple review, which brings those consumers back. Approving `config` this way must not approve `runtime.sync_state`; server sync identity is approved only by the sync-identity review in TASK-40139.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After a restore that publishes `config`, the user can approve `config` on its own through a reviewed step in the existing post-restore review surface; until then model-catalog auto-refresh and native workspace change tracking stay off with a named reason; tests pin both states
- [ ] #2 That review approves only `config`: server sync calls stay refused with `sync_activation_required` until the sync-identity review approves `runtime.sync_state`; a test pins it
- [ ] #3 After approval, model-catalog auto-refresh and native workspace change tracking run again; a test pins each
- [ ] #4 `test_no_reviewer_approves_the_shared_sync_owners` (or its successor) allows exactly this review and the sync-identity review to approve `config`, and nothing else; it is shown red with a stray approver added
- [ ] #5 The User Guide's backup and restore page says what this review turns back on and that it does not resume sync
<!-- AC:END -->
