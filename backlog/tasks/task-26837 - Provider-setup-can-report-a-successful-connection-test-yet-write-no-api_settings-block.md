---
id: TASK-26837
title: >-
  Provider setup can report a successful connection test yet write no
  api_settings block
status: To Do
assignee: []
created_date: '2026-08-31 20:05'
labels:
  - console
  - ux-review
  - follow-up
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
During the 2026-08-30 Console review a first-run setup completed for a local provider whose connection test had just reported success, and the finished config contained no api_settings entry for it at all -- the summary screen reported 'no credentials or saved endpoint' for the provider it had just verified. TASK-25817 originally attributed this to the resume checkpoint dropping the endpoint; that was traced to code and disproved (the checkpoint is not the applied configuration, and the endpoint is deliberately scrubbed from step memory). The commit path itself was never examined, so the behaviour that produced an unusable configuration is still unexplained.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Completing the provider step for a local provider writes an api_settings entry that can reach that provider
- [ ] #2 A successful connection test cannot be followed by a summary reporting the same provider as unconfigured
- [ ] #3 The conditions under which provider commit is skipped are covered by a test
<!-- AC:END -->

## Renumbering

Filed as TASK-25887 on 2026-08-31 20:05. `dev` merged its own TASK-25887 on
2026-09-01 05:15, and the backlog guard flagged the duplicate.

Deviation from the 2026-08-21 owner rule (TASK-19601), stated so it is not read
as an oversight: by that rule the OLDER arrival keeps the id, which would be
this task. It moves the other way because dev's task is already MERGED and may
carry references an unmerged PR cannot see, while this one has no blast radius.
Renumbered to 26837 (next free after sweeping all refs; max was 26836).

This is the fourth such collision for this review's tasks in two days -- the
backlog CLI mints against the local checkout only, so a branch that sits open
across other merges will keep losing the race.
