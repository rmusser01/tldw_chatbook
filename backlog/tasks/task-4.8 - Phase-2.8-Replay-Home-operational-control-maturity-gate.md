---
id: TASK-4.8
title: Phase 2.8 Replay Home operational-control maturity gate
status: Done
assignee: []
created_date: '2026-05-05 00:19'
updated_date: '2026-05-05 01:07'
labels:
  - unified-shell
  - phase-2
  - home
  - qa
  - closeout
dependencies: []
parent_task_id: TASK-4
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replay Phase 2 Home operational-control workflows in the running app and decide whether the phase can be verified or must remain blocked by real-service gaps.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 running-app QA walkthrough verifies Home active-work controls and open-detail paths are functional or explicitly recoverable
- [x] #2 QA evidence proves approve reject pause resume retry and open-detail states are not render-only or click-only behavior
- [x] #3 Phase 2 roadmap README and parent task status are updated from the walkthrough result
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-read Phase 2 child evidence and Home active-work tests to define the maturity-gate checklist.
2. Run the focused Home adapter/dashboard/screen/navigation tests that exercise approve reject pause resume retry open-detail and recoverable unavailable states.
3. Record a Phase 2 closeout QA artifact with exact verification output, workflow matrix, defects, and residual risks.
4. Update Phase 2 README, roadmap, and parent task status according to the QA result without marking Phase 2 verified unless the evidence supports it.
5. Add or update a tracking regression test so the closeout state cannot drift.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Closed Phase 2 Home maturity-gate QA with focused running-app Textual replay evidence, tracker status updates, and a closeout contract test. Phase 2 is verified for explicit Home adapter behavior, local W+C and notification flows, and recoverable unavailable states; schedule and agent-service adapters remain future work.
<!-- SECTION:NOTES:END -->
