---
id: TASK-13.5
title: 'Phase 6.5: Recovery setup and documentation alignment'
status: To Do
assignee: []
created_date: '2026-05-16 00:00'
labels:
  - product-maturity
  - phase-6-release-hardening
dependencies:
  - TASK-13.1
parent_task_id: TASK-13
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Align provider/model/runtime/server/optional-dependency recovery copy with current documentation so users can diagnose and recover from blocked release paths.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 QA walkthrough verifies recovery and setup states for provider/model, server, ACP, MCP, optional dependency, and missing-source blockers in the running app.
- [ ] #2 Focused regression evidence exists for recovery-copy and documentation-alignment seams changed by this task.
- [ ] #3 Repo-tracked QA evidence exists under Docs/superpowers/qa/product-maturity/phase-6/.
- [ ] #4 P0/P1 findings are fixed or explicitly accepted according to the spec severity policy.
<!-- AC:END -->
