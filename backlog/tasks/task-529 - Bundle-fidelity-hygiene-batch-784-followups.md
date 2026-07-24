---
id: TASK-529
title: >-
  Skills bundle-fidelity hygiene batch (PR #784 deferred minors)
status: To Do
assignee: []
created_date: '2026-07-24 14:10'
updated_date: '2026-07-24 14:10'
labels:
  - skills
  - hygiene
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred minors from the bundle-fidelity final review (PR #784): export_skill raises raw FileNotFoundError instead of a domain error; get_skill walks the skill tree twice; validate_supporting_file_path rejects skill.md by basename only (a DIRECTORY segment named skill.md passes); lowercase-skill.md handling differs across case-sensitive/insensitive filesystems; integration test misnamed _tamper_detection; depth-cap boundary (depth exactly 8) untested; unused 'import stat' in export_skill.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 export_skill surfaces a domain error for missing skills.
- [ ] #2 get_skill performs a single tree walk.
- [ ] #3 Path validator treatment of skill.md segments (file vs directory, case) is reconciled and tested on both filesystem types.
- [ ] #4 Test naming/coverage nits fixed (tamper-detection name, depth-8 boundary, unused import).
<!-- AC:END -->
