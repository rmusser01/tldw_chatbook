---
id: TASK-3.11
title: Phase 3.11 Replay Console live-work maturity gate
status: Done
assignee: []
created_date: '2026-05-05 00:19'
updated_date: '2026-05-05 01:31'
labels:
  - unified-shell
  - phase-3
  - console
  - qa
  - closeout
dependencies: []
parent_task_id: TASK-3
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replay Phase 3 Console live-work workflows in the running app and decide whether the Console live-work hub can be verified or must remain blocked by source-specific runtime gaps.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 running-app QA walkthrough verifies Console launch follow status and recovery flows from implemented sources
- [x] #2 QA evidence proves workflows schedules RAG artifacts and active-work source launches are not render-only or click-only behavior
- [x] #3 ACP MCP and deeper event-stream gaps are documented as verified recovery states or follow-up blockers
- [x] #4 Phase 3 roadmap README and parent task status are updated from the walkthrough result
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-read Phase 3 child evidence and Console live-work tests to define the maturity-gate checklist.
2. Run the focused Console live-work source/status/routing tests that exercise workflows, schedules, RAG, artifacts, W+C active work, ACP/MCP recovery states, and Console follow-through.
3. Record a Phase 3 closeout QA artifact with exact verification output, workflow matrix, defects, and residual risks.
4. Update Phase 3 README, roadmap, and parent task status according to the QA result without marking Phase 3 verified unless the evidence supports it.
5. Add or update a tracking regression test so the closeout state cannot drift.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Closed Phase 3 Console live-work maturity-gate QA with focused mounted Textual replay evidence, tracker status updates, and a closeout contract test. Phase 3 is verified for implemented Home active-work, W+C, Schedules, Search/RAG, Artifacts, and Workflows Console launch/follow/status/recovery flows; ACP, MCP, and durable event streams remain explicit recoverable future-work gaps.
<!-- SECTION:NOTES:END -->
