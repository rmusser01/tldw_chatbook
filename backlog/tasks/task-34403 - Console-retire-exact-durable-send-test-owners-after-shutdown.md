---
id: TASK-34403
title: 'Console: retire exact durable-send test owners after shutdown'
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-05 19:29'
updated_date: '2026-10-05 19:39'
labels:
  - console
  - testing
  - resource-ownership
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_chatbook/pull/3024'
documentation:
  - Docs/QA/task-33620.9/README.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Durable-send qualification accumulates controller.sqlite handles from test helpers that never retire their own controllers or database registries. Make the affected module own its teardown so physical resource evidence is meaningful without changing production lifetime policies.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every database created by this module’s controller and ready-store helper calls is retired only after its associated live controller shutdown succeeds.
- [x] #2 Direct module-owned evidence controllers use the same exact-owner teardown; foreign databases and shared configuration owners are untouched.
- [x] #3 Original send, recovery, token-fencing, queue and retention assertions remain unchanged, with raw warnings and strict descriptor results recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-run the twelve exact durable-send controls with the unchanged read-only strict descriptor gate to preserve RED. 2. Give only this importing test module opt-in ownership of its returned controller/ready-store helper objects and explicitly register its direct evidence owner; await the existing controller shutdown before the existing database quiescence boundary. Keep failed drains loud and retain their files. 3. Re-run affected controls and the module without reducing original counts/timeouts, static guards and independent scoped review; record emergency warnings separately from ordinary retirement. ADR required: no. ADR path: N/A; existing ADR120/198 apply. Reason: test-only exact-owner teardown follows established lifecycle APIs; no production runtime, storage, shared-config, global cache or GC policy.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented opt-in exact ownership in the affected durable-send test module only: local helper wrappers capture returned DB/controller owners, the direct evidence case explicitly registers, and existing awaited shutdown precedes same-file quiescence. Failed drains retain their own files and rethrow while independent owners still retire; no production/shared-config/global constructor/GC or warning-filter changes. Strict RED: same12 bodies pass8.34s but process exits1 with48SQLite descriptors. GREEN: same12 pass8.16s no warnings strict0 zeroDBfiles. Complete module25pass336.58s no warnings strict0/zeroDBfiles at every teardown; original1000-send count/timeout unchanged, case323.71s. Combined25 shutdown/postcommit/maintenance controls pass9.31s strict0/zeroDBfiles but retain2 raw intentional closed-loop ContextVar warnings. Independent review has no actionable scoped finding. All11 artifact guards pass; test format clean,2 inherited Ruff findings unchanged/no additions. Existing ADR120/198 apply; no new ADR. QA receipt Docs/QA/task-33620.9/README.md. Keep In Progress while broader PR review/qualification and inherited static debt remain unmet; do not claim native/scale or warning-free emergency qualification.
<!-- SECTION:NOTES:END -->
