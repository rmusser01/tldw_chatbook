---
id: TASK-32116
title: Persist native goal runs and recoverable launch intent
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-09 04:16'
updated_date: '2026-10-11 22:36'
labels:
  - agents
  - console
dependencies: []
references:
  - backlog/decisions/227-native-console-goal-runs.md
documentation:
  - Docs/superpowers/plans/2026-09-08-gnhf-inspired-goal-runs.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Users need a durable objective and a single recoverable conversation when starting or reopening autonomous work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Identical repeated Start delivery resolves to the same goal, conversation identity and allowance; conflicting payloads are refused.
- [x] #2 Interrupted conversation and workspace provisioning can be reconciled without duplicate history or premature model/tool execution.
- [x] #3 Goal requests reject invalid types, oversized UTF-8 payloads and unbounded limits; stored goal policy and scope remain immutable.
- [x] #4 Reopening real SQLite preserves goal state and the existing automatic-work history; migration retains legacy fleet semantics.
- [x] #5 Goal and chain creation roll back together on failure; concurrent writers respect revision and launch identity constraints.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/227-native-console-goal-runs.md
Reason: implements accepted goal design with current schema, runtime custody, shared allowance roots, filesystem and exact backup ownership.
1. Follow Docs/superpowers/plans/2026-10-11-pr2563-current-dev-integration.md under TASK-34797; preserve original notes as historical source evidence.
2. Integrate only goal-specific changes into current owners; add schema23-to24 and exact recovery qualification.
3. Verify targeted native/runtime/storage/UI boundaries plus real CLI/restart and mounted unstarted Pause/Resume.
4. Record fresh receipts and limitations; retain incomplete checks until actual qualification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented strict immutable goal requests, AgentRunsDB v16 launch/chain persistence, and recoverable exact-identity conversation/workspace provisioning. Duplicate Start is idempotent; conflicting launches and retargeted bindings are refused; cross-store retries retain committed history. ADR: backlog/decisions/227-native-console-goal-runs.md. Implementation: 1b8ff9288a. Targeted verification: 165 + 39 passed (204 total); scoped lint/format and diff checks passed, unchanged legacy lint and existing RequestsDependencyWarning documented. Independent spec and quality review approved with no blocking findings. No dispatch, repetition or UI is claimed in this task; those remain dependent tasks. Detailed evidence: .superpowers/sdd/2026-09-08-gnhf-inspired-goal-runs/task-1-report.md.

2026-10-11: reopened for current-dev integration under TASK-34797 and ADR-227. Prior notes and counts describe historical source only; they do not certify the current integration.
<!-- SECTION:NOTES:END -->
