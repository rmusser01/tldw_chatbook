---
id: TASK-33664
title: Keep Console Resend out of empty startup imports
status: Done
assignee:
  - '@codex'
created_date: '2026-10-02 20:15'
updated_date: '2026-10-02 21:38'
labels:
  - agents
  - console
  - integration
dependencies: []
documentation:
  - Docs/superpowers/plans/2026-09-29-agent-orchestration-burndown.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The required Resend merge-base integration loads its new module during empty Console startup and exceeds the unchanged UI-ready module budget by one. Resend should load when its actual action or transcript projection is needed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The original UI-ready module census passes its unchanged 1033-module ceiling with the empty Console behavior and expected mount members preserved.
- [x] #2 Real Resend click and keyboard, duplicate-worker, custody polling and selected-row action checks pass after deferring the imports.
- [x] #3 App import, storage, CSS and source artifact guards remain unchanged and pass; focused independent review approves.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/097-boot-budget-ratchets.md; existing ADR-199 unchanged.
Reason: defer an existing optional action module to its actual consumers without changing UI, authority or module budgets.
1. Keep the original UI-ready RED1034/1033 and matching incoming-dev baseline as evidence; identify all three eager import owners and callers.
2. Move the imports into message resend, refused-echo dispatch and transcript action projection. Update the existing test's mock to the owning Resend module, preserving behavior assertions.
3. Run the unchanged original startup/storage/import/CSS guards and actual click/keyboard/duplicate-worker/custody/selected-row consumers; preserve inherited AST guard failures separately.
4. Run focused static/artifact checks, obtain independent source and mounted review, record evidence and close through CLI.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Deferred the three existing eager Resend imports to actual message resend, refused-echo dispatch and transcript action projection consumers; updated the existing legitimate test mock to the owning module. No new owner, dependency, UI behavior or budget. ADR097 directly governs the repair; no new ADR required.
Original UI-ready RED1034/1033 also reproduces on exact incoming Resend source. Final original five-case tested/untested storage/import/UI/CSS selection passes85.975s with681/686imports and1033/1033UI-ready, unchanged performance-source bytes and original drift warnings/no UI headroom. Real click/keyboard/duplicate worker, held-preflight publication and task33663 authority controls pass; final independent UI/runtime reviews approve2cfbb01c76. Static94patchPython/tennewRuff+format/whitespace and diagnostic/worker/index/UI/timestamp/CSS artifacts pass. Evidence and inherited limits are retained in the final review; no raw-suite/full-suite/live-provider/Windows result is claimed.
<!-- SECTION:NOTES:END -->
