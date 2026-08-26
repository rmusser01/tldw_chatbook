---
id: TASK-16228
title: Align ProductionApp Ollama tooltip evidence
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 08:57'
updated_date: '2026-08-14 08:59'
labels:
  - testing
  - llm
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep ProductionApp destination-action evidence aligned with the service-aware Ollama control gate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Ollama executable browsing retains its actionable tooltip while the service is down
- [x] #2 Service-dependent Ollama controls are asserted disabled with the bounded service-required tooltip
- [x] #3 The focused LLM destination-action integration test passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Characterize the service-down button states produced by the real LLM destination.
2. Update only the stale ProductionApp expectation while retaining exact tooltip assertions.
3. Run the focused integration test and static checks.

ADR required: no
ADR path: N/A
Reason: This reconciles test evidence with the existing UX-091 Ollama service gate.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Updated the real ProductionApp LLM destination test to distinguish Ollama executable browsing from service-dependent controls: the executable tooltip remains actionable, while Modelfile browsing is disabled with the exact service-required tooltip and retains its pre-gate tooltip for restoration. Worker assertions now filter by the ollama_serve group so unrelated app workers cannot perturb lifecycle counts. Focused integration, Ruff lint/format, and diff checks pass. ADR required: no.
<!-- SECTION:NOTES:END -->
