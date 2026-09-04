---
id: TASK-31273
title: An explicit open bypasses review-set auto-resume
status: To Do
assignee: []
created_date: '2026-09-04 13:54'
labels:
  - library
  - media-ux
  - review-sets
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User ruling at the critique #4 close. Task-31234 made auto-resume open the set's cursor item on every Media entry; that now overrides explicit opens — a deep link, open-by-id from another surface, or Enter on a different row while a set is active pulls the user back to the cursor item (library_screen.py:39683-39692). The ruling: an explicit open wins; plain rail entry with no target still resumes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A deep link, open-by-id, or Enter on a row that is not the cursor item opens that item even while a set is active
- [ ] #2 The review banner states the off-set state honestly (e.g. `Reviewing paused: <name> — X of M · this item is not in the set`) and ] resumes the walk from the cursor
- [ ] #3 Plain rail entry with no explicit target still auto-resumes to the cursor item
- [ ] #4 Tests pin all three paths; live-verified
<!-- AC:END -->
