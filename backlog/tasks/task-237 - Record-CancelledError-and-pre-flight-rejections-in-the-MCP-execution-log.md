---
id: TASK-237
title: Record CancelledError and pre-flight rejections in the MCP execution log
status: Done
assignee: []
created_date: '2026-07-16 15:19'
labels:
  - mcp
  - phase5
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
test_hub_tool records success/failure/timeout but not cancellations (sibling _run_local_lifecycle does) and raises unknown-prefix ValueError before the recording try, so those attempts never reach the log. Audit mode (Phase 5) reads these records; decide and implement consistent every-attempt semantics (PR #639 Task-3 review).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Cancelled tool tests write an ok=False record before re-raising,Unknown-prefix rejections either record or are documented as pre-flight validation,Log semantics documented in execution_log module docstring
<!-- AC:END -->


## Closeout audit 2026-10-11

Board-hygiene flip (cohort A, Jun-Jul 2026 status sweep). **Verdict: DONE** — evidence verified against origin/dev (a3c47c6c72).

Cancellation and pre-flight refusals now reach the execution log: LocalHubStatus includes 'cancelled' (MCP/hub_test_execution.py:37) and UnifiedMCPControlPlaneService records status='cancelled'/error_category='cancelled' (unified_control_plane_service.py:2757-2758); the owned-lifecycle rework (1dc7b702be 'own complete local Hub execution lifecycle', 376b1b9618 'finalize prepared admission refusals') replaced the old pre-try ValueError path.
