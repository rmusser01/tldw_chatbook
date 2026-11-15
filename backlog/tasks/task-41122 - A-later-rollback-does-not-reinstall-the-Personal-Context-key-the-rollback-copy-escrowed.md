---
id: TASK-41122
title: "A later rollback does not reinstall the Personal Context key the rollback copy escrowed"
status: To Do
created_date: 2026-11-15 02:20
labels:
- follow-up
- personal-context
- backup
priority: medium
references:
- tldw_chatbook/Backup_Recovery/later_rollback.py
- Docs/superpowers/specs/2026-11-10-personal-context-backup-key-custody-design.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** The approved design (spec Q4, option A) says the rollback copy escrows the live key "That covers a keyring that is lost between restore and rollback." If the original account is gone when the user rolls back later, the rolled-back database cannot be opened.

**Evidence.** Spec `:268-282`. `later_rollback.py` has no Personal Context handling at all (no reference to the owner, its key or `credentials.age`). The restore-time rollback paths delete only the restore's own account (TASK-40135, `recovery.forget_restored_keys`); nothing reinstalls the original key from the copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A later rollback whose original Personal Context account is missing reinstalls the escrowed key into a fresh account (set-if-absent) and points the rolled-back database at it
- [ ] #2 A rollback whose original account is still present leaves the keyring unchanged
- [ ] #3 Tests cover both, the first red before the change
<!-- AC:END -->
