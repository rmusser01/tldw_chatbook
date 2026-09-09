---
id: TASK-32117
title: Execute one bounded native goal iteration with CLI evidence
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-09 04:17'
updated_date: '2026-09-09 15:04'
labels:
  - agents
  - console
dependencies:
  - TASK-32116
references:
  - backlog/decisions/141-native-console-goal-runs.md
documentation:
  - Docs/superpowers/plans/2026-09-08-gnhf-inspired-goal-runs.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Users need goal iterations to use the existing Console execution path with reliable scope, accounting and observed command results.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Only one durably accepted goal attempt may dispatch; invalid or failed authorization causes zero provider/helper/tool calls.
- [x] #2 Goal enablement and finite iteration limits remain independent of fleet settings while sharing runtime ownership and accounting.
- [x] #3 Selected tool scope restricts catalog, runtime, discovered and restored calls at actual dispatch, preserving authorized existing CLI execution.
- [x] #4 Actual script exit, timeout, identity and output evidence survives display formatting and cannot be forged by tool text.
- [x] #5 Fresh goal requests omit prior settled iteration history and preserve provider continuation within the current iteration.
- [x] #6 One runtime startup audit governs goal and fleet coordinators without revoking live owners when a view or service is attached.
- [ ] #7 Selected additional read-only source bindings provide a bounded usable native read path; unselected siblings and all writes outside the primary writable binding remain refused under existing permissions.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/141-native-console-goal-runs.md (Accepted)
Reason: implements the reviewed automatic runtime, authority, policy and native CLI boundary.
1. Read the task 2 brief and completed task 1 contracts; add failing behavioral tests for kind-specific admission, origin-aware budgets and native controller dispatch.
2. Implement one bounded iteration through the existing Console with shared recovery ownership, typed CLI observation, fresh initial request history and scope enforcement at actual dispatch.
3. Run the named targeted tests and relevant manual/fleet/capacity/skill regressions; inspect final provider requests and real subprocess results.
4. Self-review, commit only this task's files, obtain independent spec/quality review and record exact evidence. Repetition and UI remain later tasks.

Final whole-branch review fix wave (before new code): existing ADR-141 applies; no duplicate ADR. Reproduce the affected setup/selected-resource failures through isolated mounted/native requests, fix all findings in the shared final-review list while preserving owner boundaries, run focused amended regressions and scoped static checks, update user/qualification docs and commit. One independent scoped re-review follows. Root owns final AC/status/notes after approval.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented one durably admitted native goal iteration through the existing Console/controller/agent and CLI owners. Goal policy and scope remain separate from fleet enablement while sharing accounting/capacity/recovery. Typed script/MCP outcomes survive formatting; exact selected roots and provider/MCP references are revalidated. Initial goal history is fresh; accepted work remains pending the next task's atomic checkpoint.

Implementation commits: 0ad46264124ffd5bbb0e4621aeb03615d00f6e44 and review fix 8597af3df0bc1e4f33ec33cf069e7bd7fb5f3a9d. Independent spec and quality review approved after fixing script/install refusal metadata, MCP/worker exception classification and cleanup ownership, and live disablement at durable acceptance.

Validation: final fix scope 146 passed; 5 manual regressions passed; 45 overlapping final script checks passed. Earlier broad run had 378 passes and one independently reproduced baseline missing-run_id review-hook failure. Existing requests warning remains. Scoped test lint/format and whitespace checks pass; no new differential production lint. Counts overlap and are not a unique aggregate. MCP tests prove local awaited-worker cleanup, not remote command termination. Live model qualification, checkpointing and UI remain subsequent tasks.

ADR required: yes. Existing accepted backlog/decisions/141-native-console-goal-runs.md governs this runtime/authority boundary; no duplicate ADR. Detailed evidence and review artifacts are in the plan's local SDD workspace.
<!-- SECTION:NOTES:END -->
