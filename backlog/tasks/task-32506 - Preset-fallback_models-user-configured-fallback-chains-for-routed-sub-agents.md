---
id: TASK-32506
title: 'Preset fallback_models: user-configured fallback chains for routed sub-agents'
status: To Do
assignee: []
created_date: '2026-09-12 01:06'
labels:
  - agents
  - console
  - llm-routing
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up to TASK-32477 (agent provider routing, ADR-147). Let an AgentDefinition preset carry an ordered fallback_models list (provider/model entries) used when the primary target fails with a retryable provider error (rate-limit, overload, unavailable model, provider-reported timeout) BEFORE any tool activity. The chain is user-authored, so it does not violate the no-silent-fallback rule of ADR-147. Must define interaction with run budgeting, provider continuation, fleet admission, and the resolved-target snapshot; reference pi-subagents docs/models.md fallbackModels semantics as a starting point. Explicitly out of scope there: mid-run fallback after tool activity.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Preset schema gains validated fallback_models list
- [ ] #2 Fallback triggers only on retryable pre-tool-activity provider failures
- [ ] #3 Budget/continuation/snapshot interactions specified and tested
<!-- AC:END -->

## Renumbering provenance

Renumbered from TASK-32479 during PR #2645 rebase on 2026-09-12 because the older task on dev retains that ID. New ID checked across remote refs and registered worktrees.
