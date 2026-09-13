---
id: TASK-32546
title: >-
  Library Notes: the Library landing's "From your Library" rows show no focus
  mark while Tab walks them, and Enter opens whichever is focused
status: To Do
assignee: []
created_date: '2026-09-13 06:47'
labels:
  - library
  - notes
  - critique-3
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 (dev 5fd502dbac), assessor B, persona Sam, the Library landing on the way into Notes (Library-wide, not Notes-specific; not owned by 32210–32237 or 32341–32393 — #2603 fixed rail-row focus shape only). D16.

**What happened.** From a cold landing (palette "Switch to Library") twelve Tabs never reached a rail row: the landing's "From your Library" rows come first and show no focus mark — an ANSI diff over the twelve captures changes only the nav-bar line — yet Tab×1 + Enter opened the Media item "Meeting recording 2026-09-05" (B 25-tabwalk-*.ansi, 28). F6 on the landing toggles between the rail search field ("typing in field") and an unmarked target (B 27). Captures: B 25-tabwalk-*.ansi, 27, 28.

**Cause.** INFERRED.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Landing "From your Library" rows show a shape-based focus cue when focused by Tab, and the footer names the focused row
- [ ] #2 F6 on the landing lands on a visibly marked target
- [ ] #3 A test pins the landing row's focus class and the footer label
<!-- AC:END -->
