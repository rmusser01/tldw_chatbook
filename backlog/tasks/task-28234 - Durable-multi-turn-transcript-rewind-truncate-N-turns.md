---
id: TASK-28234
title: Durable multi-turn transcript rewind (truncate N turns)
status: To Do
assignee: []
created_date: '2026-09-02 06:39'
labels:
  - console
  - chat
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred row C2's residue, promoted by TASK-26041: /rewind exists (regenerate-from-snapshot) and file rollback shipped via change-review + per-turn snapshots; the missing piece is durably truncating the persisted conversation back N turns as a unit (messages + linked artifacts), with a confirm step.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A rewind-to-turn action durably removes later turns from the persisted conversation after an explicit confirm
- [ ] #2 Linked rows (attachments, tool records, traces) are handled per their existing deletion semantics, not orphaned
- [ ] #3 The action is refused mid-run
<!-- AC:END -->
