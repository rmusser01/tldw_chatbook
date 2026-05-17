---
id: TASK-13.5
title: 'Phase 6.5: Recovery setup and documentation alignment'
status: Done
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
- [x] #1 QA walkthrough verifies recovery and setup states for provider/model, server, ACP, MCP, optional dependency, and missing-source blockers in the running app.
- [x] #2 Focused regression evidence exists for recovery-copy and documentation-alignment seams changed by this task.
- [x] #3 Repo-tracked QA evidence exists under Docs/superpowers/qa/product-maturity/phase-6/.
- [x] #4 P0/P1 findings are fixed or explicitly accepted according to the spec severity policy.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a failing Phase 6.5 regression that requires recovery/setup evidence, QA index tracking, roadmap tracking, and a Done task state.
2. Exercise the mounted running app to verify release recovery copy for provider/model setup, server status, ACP, MCP, optional dependency, and missing-source blockers.
3. Compare the visible recovery labels/messages against current setup/help documentation and record any P0/P1 decisions.
4. Add durable Phase 6.5 QA evidence and update the Phase 6 QA index plus product-maturity roadmap.
5. Run focused Phase 6 verification and diff hygiene before opening the PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified running-app recovery copy for provider/model setup, server/local mode, ACP runtime setup, MCP management, optional dependency recovery, and missing-source recovery. Added a release recovery/setup guide, durable Phase 6.5 QA evidence, and focused regression coverage that keeps evidence, docs, QA index, roadmap, and task state aligned. No visible UI code changed, so screenshot approval was not required. No P0/P1 blockers were found.
<!-- SECTION:NOTES:END -->
