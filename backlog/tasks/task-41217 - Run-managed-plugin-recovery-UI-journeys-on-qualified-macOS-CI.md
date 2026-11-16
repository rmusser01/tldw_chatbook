---
id: TASK-41217
title: Run managed-plugin recovery UI journeys on qualified macOS CI
status: Done
assignee:
  - '@codex'
created_date: '2026-11-15 23:04'
updated_date: '2026-11-16 00:35'
labels:
  - ci
  - plugins
  - platform
dependencies: []
references:
  - >-
    https://github.com/rmusser01/tldw_chatbook/actions/runs/43750308318/job/132962026523
  - >-
    https://github.com/rmusser01/tldw_chatbook/actions/runs/43750308318/job/132962026536
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two plugin recovery UI journeys invoke the real managed runtime but were assigned only to Linux UI lanes, where the runtime intentionally refuses qualification. Execute these journeys on the supported macOS platform and make their result required without weakening runtime qualification or losing census coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The two native plugin recovery UI journeys execute on qualified macOS in pull-request CI and unsupported platforms declare the existing managed-runtime capability.
- [x] #2 A missing, skipped or failing macOS recovery job cannot pass the existing required aggregate; both exact node IDs remain covered.
- [x] #3 Linux UI census ownership and floor remain intact and CI routing contracts reject target, runner or aggregation regressions.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes (amend existing ADR before implementation)
ADR path: backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md
Reason: add required platform-specific evidence for two managed-plugin UI recovery journeys while preserving the current aggregate context and fail-closed gate.
1. Read ADR-103, ADR-162, ADR-359 and TASK-40308; amend ADR-103 with the Linux CI failures and the qualified macOS execution requirement before changing code.
2. Declare managed-runtime capability on the two UI recovery tests and execute exactly those cases serially in a minimal macOS job in Derived Artifacts under the existing lane admission expression.
3. Include the macOS result in the required aggregate without changing its context, retaining Linux census entries and floor.
4. Extend CI contracts for exact targets, runner, timeout, admission, dependencies and fail-closed aggregation; verify negative controls and scoped census/lint checks.
5. Verify both real macOS UI journeys in PR CI, obtain independent review and record evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the qualified macOS recovery job and managed-runtime declarations. The existing required aggregate fails unless that job succeeds, and its JUnit check requires both exact recovery journeys to execute and pass without skips. Linux census entries and floor remain unchanged. ADR-103 was amended before implementation and is linked from this task plan.

Validation: current-head PR run 43778537684 at 1e34ea961d87e161d029da656f37a396d30666cd executed Managed plugin recovery UI on macOS and passed, including the two-executed-case/no-skips JUnit check (2026-11-16 00:19:18Z to 00:21:01Z). Evidence: https://github.com/rmusser01/tldw_chatbook/actions/runs/43778537684/job/133013024205 . RED confirmed the routing contract fails without the lane. All four targeted CI contract modules pass (328 checks), including wrong-platform, missing-target, missing-aggregate and nonpassing-JUnit controls. UI census, scoped Ruff and independent review pass.

Files: derived-artifacts.yml, two CI contract files, Tests/UI/test_plugins_browse.py and ADR-103. Existing ADR-162 qualification is unchanged; ADR-103 records the new required platform evidence.
<!-- SECTION:NOTES:END -->
