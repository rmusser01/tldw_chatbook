---
id: TASK-3.9
title: Phase 3.9 Launch latest Chatbook artifact from Artifacts into Console
status: Done
assignee: []
created_date: '2026-05-04 02:39'
updated_date: '2026-05-04 02:46'
labels:
  - unified-shell
  - phase-3
  - console
  - artifacts
  - chatbooks
dependencies: []
parent_task_id: TASK-3
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Artifacts participate in the Console live-work model through a real Chatbook payload instead of a generic chat handoff, so users can move from generated or portable artifacts into the primary agentic Console when a local Chatbook exists and get an honest unavailable state when none exists.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Artifacts shows Console launch as available only when a latest local Chatbook can be resolved
- [x] #2 Launching from Artifacts stages a typed Console live-work payload containing Chatbook identity and metadata
- [x] #3 Artifacts remains non-actionable with clear recovery copy when no Chatbook is available
- [x] #4 Console source readiness marks Artifacts connected after the Chatbook launch path is wired
- [x] #5 Focused automated tests cover available unavailable readiness and tracking evidence paths
- [x] #6 QA walkthrough evidence documents functional behavior visual usability residual risks and focused verification output
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing UI tests for Artifacts unavailable and latest Chatbook Console launch behavior.
2. Add failing readiness and tracking evidence tests for Artifacts as a connected Console source.
3. Implement the smallest Artifacts screen state loader that reads the latest local Chatbook off the main thread and only enables Console launch when a real Chatbook record exists.
4. Update Console source readiness copy and route Artifacts button presses through open_console_for_live_work with Chatbook metadata.
5. Add QA evidence and roadmap/readme/task links.
6. Run focused regression tests and git diff checks before completing the task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a local Chatbook-backed Artifacts Console launch path. Artifacts now loads the latest local Chatbook through the existing local_chatbook_service seam, disables Console launch with recovery copy when no Chatbook exists, and stages a typed Console live-work payload when a Chatbook is available. Updated Console readiness, Phase 3 roadmap/readme evidence, and focused UI regression coverage.
<!-- SECTION:NOTES:END -->
