---
id: TASK-18920
title: 'Console approval cards: deny with an optional reason'
status: To Do
assignee: []
created_date: '2026-08-19 09:55'
updated_date: '2026-08-19 09:55'
labels:
  - console
  - agents
  - approvals
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Port of hermes-agent's `/deny <reason>` interaction (identified in the 2026-08-19 hermes-release review). Today the Console approval card's Deny decision is silent — the model receives a bare refusal and often retries the same denied call, burning turns. Add an optional free-text reason to the Deny path (row decision select, the single-call fast Deny button, and Deny all): the reason is delivered to the model as part of the denied tool result, clearly labeled as user-authored, so the agent course-corrects instead of retrying. Purely additive — an empty reason must behave exactly as today.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every Deny decision path (per-row select, fast Deny button, Deny all) accepts an optional bounded free-text reason; an empty reason produces today's denied-result behavior byte-for-byte
- [ ] #2 A provided reason reaches the model inside the denied tool result with an explicit user-authored label (e.g. "Denial reason (from user):") that cannot be confused with tool output or an approval
- [ ] #3 Reason text is length-bounded with an honest truncation note, sanitized as untrusted input, and never written anywhere the denied result itself does not already go (no new logs/exports)
- [ ] #4 Sub-agent approval cards keep per-card scoping: a deny-with-reason resolves only its own card, exactly as Deny does today
- [ ] #5 UI tests cover reason entry on each deny path, empty-reason equivalence, bounded input, and the model-visible denied-result content
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no.
ADR path: N/A.
Reason: additive UX over the existing approval flow; no schema, sync, or boundary change (Deny is transient and the reason rides the existing denied-result path).

1. Extend the approval decision model with an optional reason field (transient, not persisted to the permission store)
2. Bridge/tool-executor: inject the labeled reason into the denied ToolResult the model sees
3. Approval card widget: optional reason affordance on deny actions, bounded input
4. Tests (decision model, result content, UI paths) + User Guide console/agent-runs-and-tools.md update
<!-- SECTION:PLAN:END -->
