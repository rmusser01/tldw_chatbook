---
id: TASK-32351
title: >-
  Library Import: a batch is labelled after its subdirectory, and the landing
  card under-reports a failed import
status: To Do
assignee: []
created_date: '2026-09-11 06:16'
labels:
  - library
  - import
  - critique-10
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Importing <profile>/inbox produced a queue group named 'nested — 6 files' (the subdirectory) though 5 of the 6 files were in inbox/ itself (B D1 caps 06/10); after 4 failed and 2 skipped the landing's Needs-attention card reads only 'An import needs review.' (B D2 cap 14). Evidence: critique #10 snapshot (.impeccable/critique/2026-09-11T06-13-11Z__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 1f3184655b, captures under the session scratchpad crit10/A and crit10/B).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A folder import's group is named after the folder the user imported
- [ ] #2 The landing card states the outcome counts ('4 failed, 2 skipped') not a neutral 'needs review'
- [ ] #3 Both pinned
<!-- AC:END -->
