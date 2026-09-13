---
id: TASK-31677
title: Defer vLLM handoff target types until validation
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:14'
updated_date: '2026-09-05 18:26'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep verified-target implementation dependencies off the first Console frame while preserving handoff validation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 UI-ready module count satisfies the existing 972 ceiling
- [x] #2 Exact target and readiness checks retain their behavior on first use
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/097-boot-budget-ratchets.md
Reason: Existing first-use import discipline with no contract change.
1. Record the current 973-module census breach and trace vLLM target imports.
2. Defer only annotation and exact-type implementation imports to their validation calls.
3. Run handoff validation tests and a fresh warm-boot census without raising caps.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Moved vLLM target/readiness type imports from module scope into exact-target validation calls, with annotations under TYPE_CHECKING. Existing strict equality, token, provider and endpoint predicates are unchanged. The fresh-process guard failed before the change and now validates absent boot imports, first-use exact target creation and stale-token rejection. State and handoff-specific behavior selection: 116 passed; final import/modal/census gate:147 passed in109.48s. Deferral removes UI.LLM_Management and vllm_setup; scheduler heartbeat/emergency-stop residency consumed that initial headroom, so the separately tracked video-capacity modal deferral was also needed for final972/972. No cap raised. Ruff lint/format and self-review passed. Existing ADR-097; no new ADR.
<!-- SECTION:NOTES:END -->
