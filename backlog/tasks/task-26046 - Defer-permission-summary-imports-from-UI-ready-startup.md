---
id: TASK-26046
title: Defer permission-summary imports from UI-ready startup
status: Done
assignee: []
created_date: '2026-09-01 02:24'
updated_date: '2026-09-01 03:01'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the existing UI-ready module ceiling after permission-request summaries made their LLM and trace dependency graph resident before the first interactive frame.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current dev's UI-ready module census passes without raising its ceiling.
- [x] #2 Permission summaries retain their existing behavior.
- [x] #3 Focused permission-summary and startup-budget tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm the eager import path against the failing dev census
2. Defer permission-summary service imports to the actions that use them
3. Add focused regression coverage for the lazy boundary
4. Run focused behavior, import-budget, lint, and artifact checks
5. ADR required: no; ADR path: N/A; reason: direct regression fix implementing existing ADR-090 and ADR-097 boundaries
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Deferred permission-summary, terminal support, trace disclosure, and briefing TTS helpers from UI-ready startup; added focused lazy-boundary coverage; preserved runtime type-hint resolution with non-eager Any fallbacks; documented and tested the lazy TerminalBackend export; moved the deferred permission-summary import inside the advisory failure boundary with a red-green import-failure regression; repaired the Backlog plan formatting; kept the 972-module macOS ceiling; removed the two Linux-only eager TTS modules identified by CI; and passed focused behavior, inspector, terminal, briefing-audio, performance, lint, compile, diff, and diagnostic inventory checks. One batched Textual timing test timed out under load and passed immediately in isolation (4.15s). ADR required: no; implements ADR-090 and ADR-097.
<!-- SECTION:NOTES:END -->
