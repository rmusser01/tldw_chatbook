---
id: TASK-32749
title: Reconcile current dev with the reviewed design-system workstream
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-17 19:00'
updated_date: '2026-09-17 19:01'
labels:
  - design-system
  - ui
  - integration
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the saved component-review branch reviewable against current dev while preserving the token migration and incoming Console, Library and picker behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The candidate includes the recorded current-dev commit with no unresolved conflicts and preserves both sides of the reviewed behavior.
- [ ] #2 Canonical stylesheet ownership, zero-literal and Python-style floors, generated artifacts and unchanged boot budget pass after integration.
- [ ] #3 Targeted affected Console, Library, picker and Settings checks pass; any pre-existing failures are separately demonstrated.
- [ ] #4 The candidate has unique Backlog IDs, passes applicable static checks and independent review, and has private native rendering and clean lifecycle evidence.
- [ ] #5 The integration report, full completion ledger and draft PR reflect the verified candidate without claiming complete feature coverage.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: existing ADR-150/161. Reason: reconcile existing contracts, with no new architecture boundary. Follow Docs/superpowers/plans/2026-09-17-component-current-dev-integration.md: record parents and IDs; merge exact current dev without automatic commit; reconcile source behavior and canonical style owners; rebuild generated artifacts; run targeted affected and governance checks; complete independent review and private native lifecycle evidence; update ledger/report and save to draft PR #2704. No full suite or merge into dev.
<!-- SECTION:PLAN:END -->
