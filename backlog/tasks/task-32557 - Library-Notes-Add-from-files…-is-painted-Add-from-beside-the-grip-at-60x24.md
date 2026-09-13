---
id: TASK-32557
title: >-
  Library Notes: "Add from files…" is painted "Add from" beside the grip at
  60x24
status: To Do
assignee: []
created_date: '2026-09-13 06:48'
labels:
  - library
  - notes
  - critique-3
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 (dev 5fd502dbac), assessor B, compact terminal. D14. Task-32360 fixed mid-word clipping on the rail-only stage below 64 columns; task-32127's whole-word pin is at 235 wide.

**What happened.** At 60x24 the Notes list toolbar's first row paints `New  Sort: Newest  Select  Add from   s` — "Add from files…" cut to "Add from" against the grip (B 52 line 10). The footer compacts correctly and the status drops its prefix as documented. Captures: B 52.

**Cause.** INFERRED. Docs contradicted: "Nothing is ever painted as half a word."
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At 60x24 every Notes toolbar label paints whole or is elided with an ellipsis
- [ ] #2 A test at 60 columns pins the toolbar labels
<!-- AC:END -->
