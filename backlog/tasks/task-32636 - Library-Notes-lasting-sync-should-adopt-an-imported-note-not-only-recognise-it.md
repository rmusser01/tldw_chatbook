---
id: TASK-32636
title: >-
  Library Notes: lasting sync should adopt an imported note, not only recognise it
status: To Do
assignee: []
created_date: '2026-09-15 10:15'
labels:
  - library
  - notes
  - critique-4
  - rider
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rider off task-32605. 32605 stopped lasting sync planning a fresh create for
every note Import once had already made from the same vault: the reconciler
now reads Import once's receipt ledger and recognises the prior import.

Recognising is not adopting. The general case — an imported note becoming a
first-class member of a lasting-sync root, with a binding, so that later edits
on either side flow — is broader than the critique's order and was deliberately
left out of 32605's scope. The honest gap statement is in PR #2691's body.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A note created by Import once, inside a folder later put under lasting sync, ends up bound — edits on either side reach the other without a re-import.
- [ ] #2 The adoption is idempotent: running it twice binds once.
- [ ] #3 A note the user has since deleted is not resurrected by adoption (32605's existing behaviour — sync creates it again only because it is genuinely absent — must survive).
- [ ] #4 The user can tell from the screen that adoption happened, without reading a log.
<!-- AC:END -->
