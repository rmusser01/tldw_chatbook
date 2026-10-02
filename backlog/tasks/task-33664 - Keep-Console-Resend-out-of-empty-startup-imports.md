---
id: TASK-33664
title: Keep Console Resend out of empty startup imports
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-02 20:15'
updated_date: '2026-10-02 20:23'
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
- [ ] #1 The original UI-ready module census passes its unchanged 1033-module ceiling with the empty Console behavior and expected mount members preserved.
- [ ] #2 Real Resend click and keyboard, duplicate-worker, custody polling and selected-row action checks pass after deferring the imports.
- [ ] #3 App import, storage, CSS and source artifact guards remain unchanged and pass; focused independent review approves.
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
