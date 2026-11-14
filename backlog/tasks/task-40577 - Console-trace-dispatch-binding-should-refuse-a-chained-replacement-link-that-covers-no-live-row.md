---
id: TASK-40577
title: >-
  Console: trace dispatch binding should refuse a chained-replacement link that
  covers no live row
status: To Do
assignee: []
created_date: '2026-11-12 23:16'
labels:
  - console
  - trace
dependencies:
  - TASK-38298
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The chained trace replacement (TASK-38298, ADR-337 amendment 2026-11-06) is revalidated at dispatch binding: each link's start and end nodes, that its range is contiguous, and that no live row lies between links. Binding does not check that each link covers at least one live row of the surface it replaces. The planner never builds such a link, but if a malformed plan reached binding it would commit, and every later read of that trace segment would fail (surface_replacement_target_inactive) instead of one send being refused. A per-link live-row check would keep the failure closed and local to the send.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Dispatch binding refuses, before writing anything, a chain in which any link's range holds no live row of the predecessor surface
- [ ] #2 A test pins the refusal and is red without the check
- [ ] #3 Chains the planner builds and single replacements still bind and read back unchanged
<!-- AC:END -->

## Identity

Filed 2026-11-12 from the PR #3515 review (finding F3). The backlog CLI offered TASK-40334, which another branch already holds; a sweep of every ref's task paths and every worktree found a maximum of 40546, so this task takes 40577.
