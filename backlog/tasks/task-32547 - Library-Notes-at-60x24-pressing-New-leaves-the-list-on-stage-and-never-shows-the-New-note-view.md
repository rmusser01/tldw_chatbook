---
id: TASK-32547
title: >-
  Library Notes: at 60x24 pressing "New" leaves the list on stage and never
  shows the New note view
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
Critique #3 (dev 5fd502dbac), assessor B, persona Jordan on a compact terminal. D10 (workaround: Ctrl+N).

**What happened.** 60x24, Notes list → "New": the footer changes to "enter create | esc notes" but the stage still shows the list and the work pane stays collapsed to its `Notes` grip — Blank note / From a template… are never on screen (B 54). Captures: B 54.

**Cause.** INFERRED: single-stage routing does not promote the New note view. Not covered by 32202 / 32455 (compact surplus rows, ctrl+n region scoping) or 32391 (Ctrl+N transitional frame).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At 60x24 pressing New promotes the New note view (Blank note / From a template…) to the stage with Blank note focused and the footer reading enter create note
- [ ] #2 Escape from that view returns to the list at the same size
- [ ] #3 A test at 60x24 pins the promotion
<!-- AC:END -->
