---
id: TASK-41124
title: "Restore review says credentials 'need manual recovery' for a backup whose Personal Context key a replace restore installs itself"
status: To Do
created_date: 2026-11-15 02:20
labels:
- follow-up
- backup
- personal-context
- ux
priority: low
references:
- tldw_chatbook/UI/Screens/backup_restore_screen.py
- tldw_chatbook/Backup_Recovery/replacement.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** Since TASK-40135 a replace restore installs a readable Personal Context key into a fresh account. The review still warns that credentials need manual recovery, which tells the user to do work the restore does for them.

**Evidence.** A Personal Context credential record still triggers `credential_isolated_retention_required` (`replacement.py:1497-1499`; TASK-40135's notes, "Not changed"). Its copy, `backup_restore_screen.py:2039` and `:2095`, reads "Some (incoming) credentials need manual recovery and will remain in the encrypted archive." (The other half of this follow-up, a rolled-back restore leaving a `recovery_` keyring entry, was fixed in TASK-40135 review r14pccA: rollback deletes the restore's own account at all three sites.)
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 When the only retained credential is a Personal Context key the restore installs, the review does not say it needs manual recovery
- [ ] #2 Other retained credentials keep the current copy
- [ ] #3 Tests cover both, red before the change
<!-- AC:END -->
