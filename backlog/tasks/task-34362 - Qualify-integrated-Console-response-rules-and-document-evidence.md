---
id: TASK-34362
title: Qualify integrated Console response rules and document evidence
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-04 05:56'
updated_date: '2026-10-04 14:36'
labels: []
dependencies:
  - TASK-34361
references:
  - Docs/superpowers/plans/2026-10-03-console-response-rules.md
documentation:
  - backlog/decisions/219-console-learned-response-rules.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Enable the approved Console response-rule behavior for plan task 9. Preserve existing work and make failure states honest.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Integrated route tests cover learning, checking, cancellation, shared repair and inert reopen.
- [ ] #2 Provider behavior and native UI evidence are qualified separately from scripted tests.
- [x] #3 Targeted regression and static checks pass and independent whole-branch review findings are addressed or recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/219-console-learned-response-rules.md. Reason: integrated response-rule qualification and private lifecycle boundaries. Test actual composer learning/repair/reopen/exclusion and preserve original actions, run targeted feature and touched-owner/static gates, attempt separately-labelled provider/native-terminal qualification, perform one fresh whole-branch review and document evidence and limitations before task closure.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the integrated Console response-rule journeys and resolved all seven Important whole-branch review findings in one native TDD pass. Fresh targeted feature gate: 233 passed in 1027.37s; touched-owner gate: 62 passed, one deselected in 57.49s after the unchanged sparse-context-policy promotion failure reproduced with exact product-base ConsoleChatStore. Black, scoped ruff, mypy and diff checks passed. Native repair retains completed calculator work, original answers, prompt-history privacy and inert reopened task ancestry. Settings works before Console construction and after last Chat close. Promotion retains body-free calibration, deleted temporary sources no longer break Save, and effective-rule capacity refuses before publication. Updated source/UI/tests, ADR-219, user guide, implementation evidence and the concrete Windows scratch-ACL lesson. One independent review, no re-review, no deferred minors. Actual configured provider probe returned ConnectError and visible native-terminal tooling is unavailable: acceptance criterion 2 stays unchecked and this task remains In Progress. Managed worktree retained for handoff; no push, merge or PR.
<!-- SECTION:NOTES:END -->
