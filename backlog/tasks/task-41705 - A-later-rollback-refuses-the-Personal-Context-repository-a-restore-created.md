---
id: TASK-41705
title: "A later rollback refuses the Personal Context repository a restore created"
status: To Do
assignee: []
created_date: 2026-11-18 06:11
labels:
- follow-up
- backup
priority: medium
references:
- tldw_chatbook/Backup_Recovery/later_rollback.py
- tldw_chatbook/Personal_Context/recovery.py
- Tests/Backup_Recovery/test_created_owner_rollback.py
dependencies:
- TASK-41521
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from TASK-41521 (review r16rs-rev2). A cross-profile restore into a profile without a Personal Context database creates `tldw_chatbook_personal_context.db` (read from the code, not run). `later_rollback._created_destination_target` does not admit `personal_context.repository`, so the later rollback refuses with `local_snapshot_absence_unclassified`. TASK-41521's created-owner rollback test could not cover it: a source fixture needs the encrypted profile and its key material in a keyring, which that harness does not build.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A later rollback of a restore that created the Personal Context repository retires it, or refuses by a named reason the notes justify, with a created-file rollback test that is red before the fix
<!-- AC:END -->
