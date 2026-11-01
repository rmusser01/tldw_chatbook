---
id: TASK-38493
title: Design native phased agent workflows in a shared worktree
status: Done
created_date: 2026-11-01 16:07
assignee:
- '@codex'
labels:
- console
- agents
- design
documentation:
- Docs/superpowers/specs/2026-11-01-native-phased-agent-workflows-design.md
- backlog/decisions/285-native-phased-agent-workflows.md
updated_date: 2026-11-01 17:29
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define native Console workflows that divide coding goals into approved phases using the selected shared worktree, preserve progress for explicit resume, and reuse existing execution, permission, change-review and accounting owners. Coding is the first delivery, followed by research and general tasks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The design covers same-worktree execution, initial-plan and per-phase approval, exact attempt ownership, and explicit restart resume.
- [x] #2 All six source-backed review gaps and the evidence-freshness and bounded-plan improvements have explicit contracts and acceptance scenarios.
- [x] #3 The spec maps changes to existing code and documents a canonical ADR without changing product behavior or existing Goal and fleet contracts.
- [x] #4 The remaining Console controls, failure states, verification and rollout boundaries are documented and presented for user review.
- [x] #5 The written design passes scoped metadata, link, consistency and whitespace checks and receives final user approval before implementation planning.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/285-native-phased-agent-workflows.md
Reason: Durable phase/task authority, worktree coordination, workflow timing and payload ownership change Console/AgentRunsDB contracts. Existing Goal, fleet, permissions and portable Workflows policies remain unchanged.

1. Trace current Console dispatch, fleet launch, tool restrictions, physical ownership, change review, budgets and recovery.
2. Write the revised draft spec and Proposed ADR covering all six review gaps and both improvements, with source ownership and acceptance scenarios.
3. Complete proposed Console controls, error/recovery behavior, targeted verification and delivery boundaries; present remaining sections for user review.
4. Self-review and validate local links, source anchors, task metadata and scoped whitespace; leave the task In Progress pending final approval.
5. After final design approval, finalize and commit only these design artifacts, then request written-spec approval before invoking writing-plans. No product implementation or full test sweep.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Finalized the user-approved shared-worktree specification and accepted ADR-285 on 2026-11-01. The design reuses ConsoleRuntime/controller/bridge, AgentService/FleetCoordinator, existing permission and physical-operation owners, AutomaticWorkLedger, Goal evidence/storage patterns and Change Review. It covers all six review corrections, bounded plans, input/evidence freshness, UI controls, explicit restart recovery and twelve acceptance scenarios. Existing Goal and fleet policy remains unchanged; no product code was modified.

Validation: 42 local links resolved; task frontmatter/owned sections and scoped whitespace checks passed; design/ADR approval statuses agree. The user explicitly approved finalization and implementation planning. Runtime tests and the full suite were not run for this documentation-only task; prior bootstrap failures remain an execution prerequisite.

ADR required: yes; backlog/decisions/285-native-phased-agent-workflows.md records the accepted contract. Product implementation and runtime qualification have not started.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
