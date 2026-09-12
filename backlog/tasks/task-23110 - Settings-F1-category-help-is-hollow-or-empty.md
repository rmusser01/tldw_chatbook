---
id: TASK-23110
title: Settings F1 category help is hollow or empty
status: Done
assignee: []
created_date: '2026-08-28 14:06'
updated_date: '2026-08-29 02:24'
labels:
  - ux
  - settings
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
F1 help for 'Settings: Schedules' opened a completely empty scroll body (live-verified); for Appearance it lists only three shortcut keys. The content that would make it useful already exists - the Scope Inspector's contract rows and each category's state-scope copy. P3 from the 2026-08-28 critique (.impeccable/critique/2026-08-28T06-32-49Z__tldw-chatbook-ui-screens-settings-screen-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 F1 help for every settings category shows non-empty, category-relevant content (at minimum the save contract, ownership, and available verbs)
- [ ] #2 No category opens an empty help body
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
F1 help for every settings category now renders a 'How this category works' section built from the persistence badge, scope line and ownership matrix; a test iterates all 26 SettingsCategoryId members. Two fixes came out of review: the help body rendered through a markup-enabled Static, so the Agents boundary copy's literal '[tools]' was parsed as a markup tag and silently deleted from the sentence it existed to make, and the promoted ownership strings needed a copy pass (Overview was a subject-less run-on, read-only domain pages restated one sentence four times). PR #2170.
<!-- SECTION:NOTES:END -->
