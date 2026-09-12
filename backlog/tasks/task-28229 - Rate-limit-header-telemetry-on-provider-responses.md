---
id: TASK-28229
title: Rate-limit header telemetry on provider responses
status: To Do
assignee: []
created_date: '2026-09-02 06:39'
labels:
  - providers
  - observability
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred row C21's cheap residue, promoted by TASK-26041: the display surface now exists (console cost tracker + cost chip), but zero x-ratelimit-* headers are read anywhere in LLM_Calls/. Parse the standard rate-limit headers off provider responses and surface remaining-budget alongside cost.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Rate-limit remaining/reset from provider response headers is captured per call where the provider sends them
- [ ] #2 The Console cost surface shows remaining budget when known, absent otherwise (no fake zeros)
- [ ] #3 Providers without the headers behave exactly as today
<!-- AC:END -->
