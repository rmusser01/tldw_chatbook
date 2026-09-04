---
id: TASK-31282
title: Define vLLM Lab-to-Console readiness and ownership contract
status: Done
assignee:
  - '@codex'
created_date: '2026-09-03 22:31'
updated_date: '2026-09-04 00:09'
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
- [x] #1 An accepted ADR distinguishes process liveness, API readiness, served-model identity, Console session adoption, and durable defaults.
- [x] #2 The contract defines generation fencing, endpoint normalization, network-exposure behavior, privacy boundaries, and rollback.
- [x] #3 The design specification covers first-time and experienced-user workflows at normal and compact terminal widths.
- [x] #4 No production code changes are included in the contract task.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Review the latest-dev vLLM launcher, Console provider adoption, profile persistence, and compact Lab patterns.
2. Record ADR-117 for vLLM process/readiness/adoption/profile ownership.
3. Write the approved end-to-end first-time and power-user design specification and responsive wireframe.
4. Verify task/ADR links, dependency order, placeholders, scope, and documentation-only diff.
5. Mark TASK-31282 Done with implementation notes after the contract package passes focused documentation checks.

ADR required: yes

ADR path: `backlog/decisions/117-vllm-lab-console-readiness-and-profiles.md`

Reason: This work defines provider/runtime ownership, a cross-screen service contract, durable profile storage, privacy boundaries, and long-lived UX structure.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Accepted ADR-117 and the approved end-to-end vLLM Lab-to-Console specification. Defined process, readiness, model identity, session/default ownership, profile privacy, responsive behavior, rollback, and six-task delivery boundaries. No production code changed.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

This task previously held id TASK-31213. During the post-rebase merge-candidate
guard, current `origin/dev` already shipped `task-31213 -
Restrict-production-PyPI-publishing-to-main.md` at add commit
`2a6f760fbdf0ffc9a25c7f9cdef2be469da34a63`. The unmerged vLLM contract moved
to collision-free TASK-31282 and the complete dependent sequence shifted to
TASK-31283 through TASK-31287. The vLLM record was originally added by
`ffc4f9d8f8343169097dcac40d3ba4ed0a2177c0` (rebased as `3e835a6045`).

A second merge-time sweep found that `origin/dev` had advanced to
`1a1b5c19e0bb3243effb1ae9671158b6670ad6da` and now canonically claimed the
intermediate TASK-31263 and TASK-31264 IDs for unrelated theme follow-up work.
The complete vLLM sequence therefore moved together from TASK-31263..31268 to
the next contiguous block proven free across every fetched non-vLLM ref,
TASK-31282..31287. This contract maps TASK-31263 -> TASK-31282; ADR-117 remained
collision-free.
