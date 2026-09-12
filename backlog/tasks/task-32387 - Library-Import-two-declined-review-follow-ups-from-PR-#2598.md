---
id: TASK-32387
title: 'Library Import: two declined review follow-ups from PR #2598'
status: To Do
assignee: []
created_date: '2026-09-11 10:30'
labels:
  - library
  - import
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two suggestions from the review of task-32351 (PR #2598) were declined in-round as out of that task's scope and recorded for later. First, a batch is named after the folder it came from by inspecting paths at naming time; the batch's own stored metadata is the more durable source and would survive a move. Second, the lifecycle provenance key written by the onboarding apply is not the key a later reader looks for, so provenance for that step is written and never read back. Neither is user-visible today; both are cheap while the code is fresh.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Folder-derived batch naming reads the batch's stored metadata rather than re-deriving it from paths at naming time
- [ ] #2 The lifecycle provenance key written by the onboarding apply is the key its reader queries, proven by a round-trip test
- [ ] #3 Neither change alters the batch names or lifecycle behaviour a user already sees
<!-- AC:END -->
