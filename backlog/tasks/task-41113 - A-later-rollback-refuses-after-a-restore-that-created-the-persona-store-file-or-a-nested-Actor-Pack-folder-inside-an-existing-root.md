---
id: TASK-41113
title: "A later rollback refuses after a restore that created the persona store file or a nested Actor Pack folder inside an existing root"
status: To Do
created_date: 2026-11-15 02:20
labels:
- follow-up
- backup
- actor-packs
priority: low
references:
- tldw_chatbook/Backup_Recovery/later_rollback.py
- Tests/Backup_Recovery/test_created_persona_subtree_rollback.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** The rollback copy promises the user can undo the restore later. These two shapes cannot be undone.

**Evidence (TASK-40305's notes, review r14act-rev1).**
- A restore from another profile that creates `tldw_chatbook_personas.json` refuses its later rollback with `local_snapshot_absence_unclassified` (observed; the test gives its target a persona of its own to avoid it).
- A restore that creates a nested pack folder inside an existing root refuses the same way (read from the code, not run).
- `later_rollback.py:634` (`_CREATED_TREE_ROOT_OWNERS`) admits created roots only for whole tree owners; the refusals are at `:1449`, `:1459` and `:1888`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A later rollback of a restore that created the persona store file completes and removes it, with a test red first
- [ ] #2 The nested-folder case is run; if it refuses, it is fixed the same way, with a test red first
<!-- AC:END -->
