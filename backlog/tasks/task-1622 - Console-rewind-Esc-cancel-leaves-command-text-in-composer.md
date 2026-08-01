---
id: TASK-1622
title: 'Console /rewind: cancelling the menu leaves "/rewind" in the composer'
status: To Do
assignee: []
created_date: '2026-07-31'
labels: [console, polish]
dependencies: []
---

## Description (the why)

Dismissing the Rewind menu with Esc (or "Never mind") leaves the literal
`/rewind` text sitting in the composer draft, so the next thing the user
types continues after it (e.g. `/rewind/bogus` → unknown-command hint). The
other slash commands that clear the draft on successful dispatch behave the
same way on cancel, but `/rewind` is the one whose menu is routinely
opened-and-cancelled while browsing. Observed live on dev @ ff435772c
(G1 user-guide session, 2026-07-31); documented as a quirk in
`Docs/User_Guide/console/branching-and-rewind.md`.

## Acceptance Criteria (the what)

- [ ] Cancelling the Rewind menu (Esc or "Never mind") leaves the composer
      draft as it was before `/rewind` was typed (empty if it only
      contained the command).
- [ ] Choosing "Restore to here" still replaces the draft with the restored
      prompt text (existing behavior preserved).
- [ ] The User Guide quirk note is updated/removed to match.
