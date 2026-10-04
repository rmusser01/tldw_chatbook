---
id: TASK-34362
title: Qualify integrated Console response rules and document evidence
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-04 05:56'
updated_date: '2026-10-04 11:52'
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
- [ ] #1 Integrated route tests cover learning, checking, cancellation, shared repair and inert reopen.
- [ ] #2 Provider behavior and native UI evidence are qualified separately from scripted tests.
- [ ] #3 Targeted regression and static checks pass and independent whole-branch review findings are addressed or recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/219-console-learned-response-rules.md. Reason: integrated response-rule qualification and private lifecycle boundaries. Test actual composer learning/repair/reopen/exclusion and preserve original actions, run targeted feature and touched-owner/static gates, attempt separately-labelled provider/native-terminal qualification, perform one fresh whole-branch review and document evidence and limitations before task closure.
<!-- SECTION:PLAN:END -->
