---
id: TASK-32637
title: >-
  Library Notes: Import once on an already-synced folder still duplicates it
status: To Do
assignee: []
created_date: '2026-09-15 10:15'
labels:
  - library
  - notes
  - critique-4
  - rider
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rider off task-32605, and the mirror of the bug 32605 fixed. 32605 taught
**lasting sync** to recognise what **Import once** had already made. The
opposite order is still broken: running **Import once** on a folder that is
already under lasting sync creates a second note for every file, because the
importer does not consult the sync bindings the way the reconciler now
consults the importer's receipts.

The critique exercised the import-then-sync order, so that is the order 32605
fixed and the only one its AC#3 claims. This is the other one, named in PR
#2691's body as the honest remaining gap rather than left for a user to find.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Import once on a folder already under lasting sync produces no duplicate notes — a file already bound is recognised, not re-created.
- [ ] #2 The user is told what was skipped and why, in the import's own result, in their terms.
- [ ] #3 Pinned in both orders in one test file, so the two directions cannot drift apart again.
- [ ] #4 A note the user deleted after binding is still importable — "bound once" is not a permanent veto.
<!-- AC:END -->
