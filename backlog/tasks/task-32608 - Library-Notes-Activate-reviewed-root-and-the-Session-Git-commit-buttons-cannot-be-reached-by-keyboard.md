---
id: TASK-32608
title: >-
  Library Notes: Activate reviewed root and the Session Git commit buttons
  cannot be reached by keyboard
status: To Do
assignee: []
created_date: '2026-09-15 06:38'
labels:
  - library
  - notes
  - critique-4
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 (dev 77eb2601a6), assessor B D2, P1, persona Sam (keyboard-only), Obsidian workflow. Two of B's six cells are not completable without a pointer.

What happened. (a) Sync review: Tab runs past 'Activate reviewed root' into the rail, and Enter there navigates the app to Conversations -- the root was never activated (B cap 33, K17). (b) Session Git commit form: the subject field and the Cancel commit / Review commit buttons are 30 rows apart, Tab x4 reaches no visible stop, and the buttons were only reachable by a computed mouse click at row 48 (B caps 42, 43, K18). A independently reports the same shape for the import review: no visible keyboard route to 'Import selected items' or 'Activate reviewed root', both reached with the mouse (A section 9).

Keystroke result: create, edit and Import once are fine (1, 4 and ~12 keys); lasting-sync activation and Session Git commit are not completable by keyboard.

Cause INFERRED -- screen-wide Tab order with no containment for these panes; not traced. No pinning test found. Adjacent open rider: 32585 (reaching the Manage sync folders controls takes about 23 Tabs), which names the same missing containment for the roots list.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every action in the sync review pane is reachable by Tab from that pane's first control, with a visible focus indicator, and Tab does not leak into the rail
- [ ] #2 Every action in the Session Git commit form and commit review is reachable by Tab from the subject field
- [ ] #3 A keyboard-only walk completes activate-a-root and commit-a-session-change end to end, captured
- [ ] #4 Tests pin the tab route into each pane's terminal action, so a later layout change cannot silently strand it
<!-- AC:END -->
