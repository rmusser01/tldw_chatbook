---
id: TASK-42010
title: "The RAG retained roundtrip test fails at capture preview"
status: To Do
assignee: []
created_date: 2026-11-19 19:39
updated_date: 2026-11-19 19:39
labels:
- follow-up
- tests
- backup
- rag
priority: low
references:
- Tests/Backup_Recovery/test_rag_retained_roundtrip.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41718 (round 17), red at base. `test_native_rag_retained_roundtrip_with_fresh_passive_reads` fails `assert preview.complete` with `('missing_required',)` before capture runs. It is the only end-to-end RAG roundtrip and runs in the backup-recovery-qualification workflow. TASK-42000 fixed a similar `missing_required` in the pet roundtrip by seeding the required dependencies.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause is named and fixed, and the test passes
<!-- AC:END -->
