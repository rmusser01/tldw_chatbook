---
id: TASK-3.8
title: 'Phase 3.8: Launch RAG search result from Search/RAG into Console'
status: Done
assignee: []
created_date: '2026-05-04 01:54'
updated_date: '2026-05-04 02:02'
labels: []
dependencies: []
parent_task_id: TASK-3
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Search/RAG results a real Phase 3 Console live-work source by allowing a selected retrieved result to be staged into Console with clear recovery and runtime-policy behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Search/RAG result cards expose a distinct Console launch action separate from Use in Chat.
- [x] #2 Selecting a RAG result Console action stages a typed Console live-work launch with result identity score source and recovery context.
- [x] #3 Runtime policy blocks server-backed RAG Console launch with recovery copy before staging when the active runtime is incompatible.
- [x] #4 Console source readiness marks RAG as connected while leaving still-unwired sources honest.
- [x] #5 Focused automated tests and Phase 3 QA evidence verify the RAG Console launch workflow roadmap and task updates.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing regressions for the Search/RAG result Console action, policy-blocked server RAG launch, Console readiness copy, and Phase 3.8 tracking evidence.
2. Extend the SearchResult event/button contract with a separate Use in Console request without changing Use in Chat.
3. Reuse existing RAG handoff payload construction and policy checks to stage a typed ConsoleLiveWorkLaunch through app.open_console_for_live_work.
4. Update Console source readiness, roadmap, Backlog, and QA evidence to reflect RAG as connected.
5. Run focused UI/Console tests plus diff hygiene verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a separate Search/RAG Use in Console action that reuses existing RAG handoff payloads and runtime-policy blocking before staging a typed Console live-work launch. Updated Console source readiness to mark RAG connected, added focused mounted-window regressions, and recorded Phase 3.8 roadmap and QA evidence.
<!-- SECTION:NOTES:END -->
