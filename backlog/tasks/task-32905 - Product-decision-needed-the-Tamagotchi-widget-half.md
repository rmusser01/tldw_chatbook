---
id: TASK-32905
title: "Product decision needed: the Tamagotchi widget half"
status: To Do
assignee: []
created_date: '2026-09-21 23:05'
labels:
  - tier2-review
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Widgets/Tamagotchi/` is 2,181 lines with no screen, no route and no `compose()` -- unreachable as a
widget. But the **storage** half is wired into backup/recovery and the private-SQLite allowlist, and CSS
and timer-inventory rows reference the widget half.

So this is not a dead-code sweep: deleting the widgets while the storage stays leaves orphaned persisted
state and inventory rows, and deleting both is a feature removal. Someone who knows whether this feature
is coming back has to say which.

Source: tier-2 code review 2026-09-21 -- `qa/tier2-code-review-2026-09-21/report.md` (26 slices, 890,356 lines: the surface tier 1 never reached). Per-slice evidence in `qa/tier2-code-review-2026-09-21/slices/`, reproductions in `phase4-verification.md`, and per-finding re-validation against `origin/dev d0face3ebe` in `validation/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A decision is recorded: ship it, remove it entirely, or keep the storage and delete the widgets
- [ ] #2 Whichever is chosen, the CSS and timer-inventory rows are reconciled in the same change
- [ ] #3 If the storage is kept, its backup/recovery and allowlist wiring is documented as deliberate
<!-- AC:END -->
