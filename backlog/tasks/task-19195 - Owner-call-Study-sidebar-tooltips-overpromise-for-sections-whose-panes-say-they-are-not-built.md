---
id: TASK-19195
title: 'Owner call: Study sidebar tooltips overpromise for sections whose panes say they are not built'
status: To Do
assignee: []
created_date: '2026-08-20'
labels:
  - owner-decision
  - study
  - ux-copy
dependencies:
  - TASK-19041
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-19041 made the unbuilt Study panes honest: `UI/Study_Window.py` now
states "Course creation is not available yet in this build." (:477), "Study
guides are not available yet in this build." (:527), and "The learning map is
not available yet in this build." (:575). But the sidebar that leads users to
those panes still overpromises. Verified at dev `7877defba`,
`UI/Screens/study_screen.py` `_SECTION_TOOLTIPS` (:106-117) advertises:

- "Generate or open study guides from your material." (:110) — Guides button
  composed at :229-232
- "Create course outlines and study sequences." — Course button at :238-242
- "Open the learning map for relationships across study material." — Map
  button at :243-247

A user hovering the sidebar is promised a working feature; clicking lands on a
pane that says the feature does not exist. Screen-IA decision for the owner:
hide the unbuilt sections from the sidebar until they ship, or keep them
visible with softened, honest tooltip copy (e.g. naming them as
not-yet-available). Per the standing ruling, prefer the durable option — the
one that will not need re-litigating each time a pane ships or slips.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 The owner has chosen between hiding unbuilt Study sections and softening their sidebar copy, and the decision is recorded in this task.
- [ ] #2 After implementation, no Study sidebar affordance promises functionality its pane then denies: tooltip copy and pane state agree for every section.
- [ ] #3 The chosen mechanism keys off the same source of truth as the panes' built/unbuilt state, so a section shipping later does not require re-synchronizing copy by hand in two places.
- [ ] #4 The Study page in Docs/User_Guide is updated (or its "Verified against" stamp refreshed) per the repo's UI-change rule.
<!-- AC:END -->
