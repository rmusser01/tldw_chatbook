---
id: TASK-31213
title: Define vLLM Lab-to-Console readiness and ownership contract
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-03 22:31'
updated_date: '2026-09-03 22:35'
labels:
  - vllm
  - lab
  - console
  - ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define the authoritative process, connection, model, persistence, privacy, and recovery boundaries for launching or attaching to vLLM in Lab and using the verified target in Console.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An accepted ADR distinguishes process liveness, API readiness, served-model identity, Console session adoption, and durable defaults.
- [ ] #2 The contract defines generation fencing, endpoint normalization, network-exposure behavior, privacy boundaries, and rollback.
- [ ] #3 The design specification covers first-time and experienced-user workflows at normal and compact terminal widths.
- [ ] #4 No production code changes are included in the contract task.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Review the latest-dev vLLM launcher, Console provider adoption, profile persistence, and compact Lab patterns.
2. Record ADR-115 for vLLM process/readiness/adoption/profile ownership.
3. Write the approved end-to-end first-time and power-user design specification and responsive wireframe.
4. Verify task/ADR links, dependency order, placeholders, scope, and documentation-only diff.
5. Mark TASK-31213 Done with implementation notes after the contract package passes focused documentation checks.

ADR required: yes

ADR path: `backlog/decisions/115-vllm-lab-console-readiness-and-profiles.md`

Reason: This work defines provider/runtime ownership, a cross-screen service contract, durable profile storage, privacy boundaries, and long-lived UX structure.
<!-- SECTION:PLAN:END -->
