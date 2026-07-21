---
id: TASK-419
title: Skills editor - make description backfill from body transparent
status: To Do
assignee: []
created_date: '2026-07-21 15:19'
labels:
  - skills
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
P2 from the 2026-07-21 Skills UX/NNG review (verified live). Saving a new skill with an empty Description silently copies body-derived text into the Description field; the user typed only a body and finds the description populated after save with no explanation. Silent field mutation. NNG heuristics 1 and 3.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 When a save would derive the description from the body the user can tell (field left empty with the derived value clearly marked as automatic or an inline note explaining the backfill),No silent mutation of user-visible fields on save,Covered by tests
<!-- AC:END -->
