---
id: TASK-9.2
title: 'Product Maturity Phase 2.2: Console Chatbook Artifact Save Contract'
status: Done
assignee: []
created_date: '2026-05-05 21:30'
updated_date: '2026-05-05 21:36'
labels:
  - product-maturity
  - phase-2-core-agentic-loop
dependencies: []
parent_task_id: TASK-9
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prove Console can create a local Chatbook artifact record from a completed assistant response with source provenance while deferring full Artifact reopen and Home resume behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Completed assistant messages expose a visible save-to-artifact action in basic and enhanced chat message widgets
- [x] #2 The save action creates a local Chatbook artifact through the app local_chatbook_service with bounded provenance metadata
- [x] #3 Missing local Chatbook service or create failure surfaces an honest recovery notification without deleting the message
- [x] #4 Focused regressions verify success and failure paths plus the widget affordance
- [x] #5 Repo tracked QA evidence records automated evidence residual risk and Phase 2.2 exit decision
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing regressions for assistant message save-to-artifact affordance and action handler success and failure paths
2. Implement the smallest ChatMessage and ChatMessageEnhanced save action affordance
3. Route the action through local_chatbook_service.create_chatbook with bounded Console provenance metadata
4. Record Phase 2.2 QA evidence and update the product-maturity tracker
5. Run focused chat event/widget regressions and adjacent Product Maturity checks
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented assistant-only save-to-artifact actions in ChatMessage and ChatMessageEnhanced. Routed the action through app.local_chatbook_service.create_chatbook with bounded Console provenance metadata and recoverable notifications for missing service or create failures. Added focused red-green regressions for widget affordance, success, missing-service, and failure paths, plus Phase 2.2 QA evidence and tracker updates.
<!-- SECTION:NOTES:END -->
