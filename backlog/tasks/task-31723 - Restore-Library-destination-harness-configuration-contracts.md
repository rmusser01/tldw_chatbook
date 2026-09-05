---
id: TASK-31723
title: Restore Library destination harness configuration contracts
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 17:57'
updated_date: '2026-09-05 18:03'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Align Library-only destination test fixtures with current app configuration and disabled-action contracts so service timeout and recovery assertions execute against valid harnesses.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both minimal Library timeout fixtures provide the required application configuration and construct successfully.
- [x] #2 Original timeout and stable recovery assertions remain unchanged and execute beyond construction; separately discovered runtime failures are recorded without weakening those assertions.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the four reported Library nodes and trace fixture versus runtime ownership before changing expectations.
2. Supply required app configuration on the two deliberately minimal timeout fixtures; preserve service timing, payload and recovery assertions.
3. Diagnose the retained rail tooltip mismatch against the production action-state policy. Keep existing assertions intact; request runtime-scope expansion if the failure is an actual projection defect rather than a fixture precondition.
4. Verify focused Library nodes and related Library selections in both files; preserve unrelated Scheduling edits.
ADR required: no
ADR path: N/A
Reason: Test fixture repair and verification of existing configuration/recovery contracts; any discovered runtime projection fix will preserve current boundaries and be explicitly authorized before implementation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added app_config={} to the two intentionally minimal Library timeout fixtures, satisfying the current lifecycle/configuration identity contract. No product fallback or behavior assertion was changed. Both fixtures now construct and reach their intended checks: the async-sleep case passes; the blocking-async case exposes a pre-existing UI-loop isolation defect (0.2075seconds against the unchanged0.05second ceiling). The two tooltip failures also revealed a retained-rail projection omission rather than stale policy expectations. Those runtime fixes were explicitly authorized and split into separate atomic tasks; no expected copy or timeout was relaxed.
Verification: four reported nodes reproduced red before correction. The two-node rerun passes async timeout and reaches the original elapsed-time assertion for blocking async. The two-line fixture diff is reviewed, syntactically valid and git diff --check passes. Existing declaration already imports SimpleNamespace.
ADR required: no; test-only required fixture configuration. Runtime diagnosis is recorded in the corresponding approved follow-up work, not hidden by fixture changes.
<!-- SECTION:NOTES:END -->
