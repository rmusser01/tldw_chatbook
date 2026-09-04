---
id: TASK-30012
title: Recompose Conversation Settings around connection-first disclosure
status: Done
assignee:
  - '@codex'
created_date: '2026-09-02 18:04'
updated_date: '2026-09-03 18:20'
labels:
  - console
  - ux
  - accessibility
dependencies:
  - TASK-30010
  - TASK-30011
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restructure the full modal around the user’s connection task while retaining fast, searchable, keyboard-first access to advanced per-conversation controls and explicit global/default save scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The full surface is named Conversation settings and presents Provider, applicable credential/endpoint, Model, verify/discover, and readiness before advanced controls
- [x] #2 Advanced generation defaults closed for first-time or blocked setup, opens when explicitly targeted, and preserves disclosure state for the current Console session
- [x] #3 Controls are hidden only when existing authoritative capability evidence says they have no effect; unknown support remains available under Advanced with neutral copy
- [x] #4 Enumerated controls use constrained widgets, provider selection is searchable/grouped, model/custom-ID speed paths remain available, and hidden controls are absent from keyboard traversal
- [x] #5 Exactly one completion action is primary, every disabled completion action has a persistent reason, and immediate Context and memory operations do not compete with completion primacy
- [x] #6 Save labels and adjacent scope copy distinguish conversation-only application from provider/model/generation defaults used by future conversations
- [x] #7 Provider default, Not estimated, singular model counts, Base URL visibility, and provider display-name copy are accurate across all modal entry points
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR
ADR paths: backlog/decisions/006-provider-aware-generation-settings.md; backlog/decisions/011-chatbook-workbench-ui-system.md; backlog/decisions/033-application-session-state-ownership.md
Reason: This changes presentation and save affordances within the existing Settings/Console ownership split and does not introduce a new long-lived UI or state owner.

Execute the red-green checklist in Docs/superpowers/plans/2026-09-02-task-30012-conversation-settings-connection-first-modal.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the connection-first Conversation settings redesign across four reviewed commits. Added a searchable grouped provider picker; capability-derived Advanced disclosure; constrained generation controls; snapshot-safe disclosure/focus restoration; explicit conversation-versus-future-default actions; single-flight defaults persistence; persistent readiness blockers; canonical provider/endpoint/context copy; and compact size-matrix coverage. Unsafe required-endpoint persistence remains deliberately routed to Settings until TASK-30013 adds the typed conflict-aware persist-and-apply transaction. Updated modal, summary/readiness projection, source/generated TCSS, and focused Chat/UI/widget tests.

Verification: provider/picker/rail slice 117 passed; modal/summary bounded slice 110 passed; Task-4 root slice 22 passed; rail 52 passed; compaction lifecycle 1 passed; native credential pointer/return 4 passed; 80x24, 100x30, and 160x40 geometry cases pass. Ruff, py_compile, CSS build, and git diff --check pass. Full suite was not run per repository policy. Independent review closed all P0-P2 findings.

ADR required: no new ADR. Existing ADR-006, ADR-011, and ADR-033 continue to own provider settings, workbench UI, and session state boundaries.
<!-- SECTION:NOTES:END -->
