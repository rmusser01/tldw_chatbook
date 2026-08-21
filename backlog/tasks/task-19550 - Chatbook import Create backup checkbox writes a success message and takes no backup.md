---
id: TASK-19550
title: >-
  Chatbook import "Create backup" checkbox writes a success message and takes no
  backup
status: To Do
assignee: []
created_date: '2026-08-21 20:00'
labels:
  - chatbooks
  - data-loss
  - honesty
priority: high
dependencies: []
---

## Description

Source: 2026-08-21 holistic review, Lane 6 (UX coherence / error handling /
honesty) — its **P0**. CONFIRMED, and re-verified at this branch base.

`tldw_chatbook/UI/Wizards/ChatbookImportWizard.py` composes a
`Checkbox("Create backup", value=True)` (line 553) under a heading offering to
back the database up before importing. The value **is** read at import time,
and the import path then writes a completed status row:

```
751:            # Create backup if requested
755:                # TODO: Implement actual backup functionality
757:                self._update_status("status-backup", "completed", "✓ Created backup")
```

No backup is taken. This is a default-ON safety control that displays a
spinner, then asserts an outcome it did not produce, immediately before a
**database-mutating import that has no rollback**. A user who loses data during
or after an import will look for the backup the wizard told them it made.

This is the sharpest instance of the theme Lane 6 identified as its most
actionable output: *the app asserts outcomes it did not produce* — a backup
that was not taken, an export that dropped items and said "successfully", a
sync that discarded an edit and said "no changes" (see also TASK-19554, the
Notes-sync `DISK_WINS` silent no-op, which converges with this).

**Disposition (the lane's own, and it matters):** REMOVE the checkbox and the
fake status row. Do not merely disable it — a greyed-out "Create backup" box
reads as "backup already handled". Per the owner's standing ruling
(durable/pragmatic over clever/unstable): removing a lying control is
strictly better than shipping a hurried backup implementation behind it. If a
real backup is wanted, it is a separate, deliberately scoped piece of work.

## Acceptance Criteria

- [ ] The import wizard no longer presents any control that offers to back up
      the database unless a backup is actually performed
- [ ] The wizard never emits "✓ Created backup" (or any equivalent success
      claim) for work it did not do
- [ ] The remaining status rows in the import flow are audited: every one that
      reports "completed" corresponds to work that actually ran
- [ ] A test fails if a status row can report success on a code path whose
      implementation is a `TODO`/no-op
- [ ] The user-facing risk is stated honestly somewhere the user sees it
      before confirming: this import mutates the database and cannot be
      rolled back
- [ ] `Docs/User_Guide/` is updated wherever it describes the import wizard's
      backup behaviour
