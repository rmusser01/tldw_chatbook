---
id: TASK-37621
title: >-
  Check whether a click-outside text selection leaves an invisible Esc layer in
  the Console transcript
status: To Do
assignee: []
created_date: '2026-10-29 12:28'
labels:
  - console
  - keyboard-focus
  - ux-review-2026-09-29
dependencies:
  - TASK-33622.4
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Raised by the TASK-33622.3/.4 review and not yet verified. The transcript's _remove_selection_menu deliberately does not cancel the selection manager (a drag's own release click is still queued). If a mouse text selection's menu closes by a click elsewhere in the transcript, selection_manager.state.selection may stay set with no highlight shown; TASK-33622.4's Esc ladder counts that as a layer, so one Esc would clear something invisible before the next Esc returns to the composer.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A Pilot probe on the real ChatScreen makes a mouse text selection, closes its menu with a click elsewhere in the transcript, and records whether the selection manager still holds a selection
- [ ] #2 If it does, the first Esc after that returns focus to the composer (no invisible layer), pinned by a test that fails on the pre-fix code; if it does not, the task records the probe and closes
<!-- AC:END -->
