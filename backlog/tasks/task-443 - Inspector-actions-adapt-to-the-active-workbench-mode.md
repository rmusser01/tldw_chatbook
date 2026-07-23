---
id: TASK-443
title: Inspector actions adapt to the active workbench mode
status: To Do
assignee: []
created_date: '2026-07-21 09:38'
labels:
  - roleplay
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the RP/character-card UX review (Docs/superpowers/qa/rp-ux-review-2026-07-21/report.md). The inspector's action block is identical in every mode: Start Chat and Export PNG render in Dictionaries/Lore modes where they cannot apply, while Duplicate exists in the Dictionaries/Lore library rails but not Characters. Buttons that can never apply to the selected kind should not render; parity gaps (Duplicate for characters) should be closed or justified.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Inspector shows only actions applicable to the selected item kind (or clearly disabled with a reason)
- [ ] #2 Duplicate is available for characters or its absence is an explicit decision
<!-- AC:END -->
