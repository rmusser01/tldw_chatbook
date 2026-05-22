---
id: TASK-69
title: Console native chat core
status: In Progress
assignee: []
created_date: '2026-05-22 08:12'
updated_date: '2026-05-22 08:13'
labels:
  - console
  - chat
  - ux
dependencies: []
priority: high
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Console uses native chat core for send and streaming
- [ ] #2 Local llama.cpp streaming path is verified
- [ ] #3 Transcript message selection and actions work
- [ ] #4 Unavailable action paths show WIP reasons
- [ ] #5 Actual CDP screenshots are captured and approved
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add native chat contracts
2. Add provider gateway
3. Add stores and controller
4. Wire Console send/streaming
5. Replace transcript rendering
6. Add selected-message actions
7. Capture CDP QA evidence
<!-- SECTION:PLAN:END -->
