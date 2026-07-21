---
id: TASK-434
title: Roleplay workbench preserves selection across navigation
status: To Do
assignee: []
created_date: '2026-07-21 09:38'
labels:
  - roleplay
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the RP/character-card UX review (Docs/superpowers/qa/rp-ux-review-2026-07-21/report.md). Observed live: navigate Personas > Console > back and the workbench resets to "Selected: none" with a blank center pane, "Console blocked: select an item", and a collapsed preview - the working context is lost on every round-trip of the workbench-to-Console loop the design itself encourages.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Returning to the Personas screen restores the previously selected item, mode, and center view within a session
- [ ] #2 Preview conversation contents survive the round-trip (until Reset or selection change)
<!-- AC:END -->
