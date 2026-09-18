---
id: TASK-32778
title: Preserve Tool Profiles action identity and shared initialization
status: Done
assignee:
  - '@codex'
created_date: '2026-09-18 09:08'
updated_date: '2026-09-18 09:20'
labels:
  - settings
  - ui
  - tool-packs
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep Tool Profiles refresh and screen navigation from changing the target of a queued action or cancelling app-owned initialization.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Queued actions from replaced controls never target the profile now occupying that row; fresh controls still emit their exact revision and digest.
- [x] #2 Refreshing or leaving Settings during Tool Profiles initialization preserves the shared app worker, and a current or reopened panel loads the completed service.
- [x] #3 Targeted mounted lifecycle regressions, existing Tool Profiles workflows, scoped static checks and independent review pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce queued row-action retargeting and shared-worker cancellation with mounted Textual controls and workers.
2. Key captured action context by the originating control and reject detached controls; shield observation of app-owned initialization from Settings cancellation.
3. Verify refreshed and reopened Settings, fresh action contexts, and existing Tool Profiles workflows with targeted tests and scoped static checks.
4. Record evidence and remaining export-recovery findings; obtain independent review and save the bounded fix to PR 2707.
ADR required: no
ADR path: backlog/decisions/107-portable-tool-use-packs.md
Reason: This repairs existing captured-context and app/service lifecycle contracts; no new storage, authority, publication, or UX boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Tool Profiles captures row actions by the originating Button object and rejects detached controls, so queued Export/Edit/Bind/Remove events cannot acquire a replacement profile or newer revision. Settings shields observation of app-owned initialization; refresh and unmount cancel only the observer, with eventual shared failure consumed safely.

78 distinct targeted cases pass: 12 lifecycle, 10 cold provisioning, 26 token/component governance and 30 existing Tool Profiles workflows. Fifteen existing app tests now use the established private-profile process wrapper; two scroll real click targets into view. AST comparison preserves all 137 original assertions. Corrected pre-fix tests failed all ten initial cases. No new Ruff diagnostics; existing Settings/panel/test baselines are 114/1/3. New tests, changed small files and the changed Settings method pass formatting; backlog and diff guards pass. Independent review found no introduced blocker.

ADR-107 applies, with no new storage or authority contract. Evidence: Docs/superpowers/qa/2026-09-18-tool-profiles-lifecycle/README.md. The review ledger records misleading existing-file export recovery and remaining native/management journeys. This task makes no new native visual claim and does not close the wider Tool Profiles review. No full suite or provider requests were run.
<!-- SECTION:NOTES:END -->
