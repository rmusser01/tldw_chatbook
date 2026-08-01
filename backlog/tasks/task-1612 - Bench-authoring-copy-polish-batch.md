---
id: TASK-1612
title: >-
  Bench authoring copy polish batch
status: In Progress
assignee: []
created_date: '2026-07-31 15:10'
labels:
  - evals
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Copy findings from the task-1482 whole-branch review, none blocking: (1) a rename collision renders the DB's "Task name already exists" — wrong vocabulary ("Task" not "bench") and, worse, after deleting a bench its name stays reserved (UNIQUE has no deleted_at exemption) so the message can appear with no visible bench of that name — explain the trap ("a deleted bench may still hold this name"); the pinned test only asserts "already exists" so copy can change freely. (2) The zero-target blocked reason still reads "This bench has no targets yet" right after the user STAGES one — nothing says Save is the arming step; append "…and Save". (3) llama_targets() silently uses list_models' default limit=100 unlike the documented _LIST_LIMIT reads — align for consistency.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] Rename-collision copy speaks bench vocabulary and explains the deleted-name reservation
- [ ] The zero-target blocked reason names Save as the arming step
- [ ] llama_targets() uses the documented list limit
<!-- AC:END -->
