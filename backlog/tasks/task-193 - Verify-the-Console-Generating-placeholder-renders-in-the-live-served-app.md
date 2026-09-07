---
id: TASK-193
title: Verify the Console Generating placeholder renders in the live served app
status: To Do
assignee: []
created_date: '2026-07-12 12:44'
labels:
  - ux
  - console
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Residual from PR #606/#608 QA: the empty-streaming-row 'Generating…' placeholder passes its transcript unit test but was not visible in any served capture during long local-model reasoning phases (assistant row renders empty for 30-90s). Either the live render path differs from the unit-tested one or the state gating skips the placeholder.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 During a live send the assistant row visibly indicates generation before the first token,Covered by a test exercising the same render path the served app uses
<!-- AC:END -->
