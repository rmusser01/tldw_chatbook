---
id: TASK-26067
title: Keep tool approvals scoped to the active profile
status: Done
assignee:
  - '@codex'
created_date: '2026-08-31 18:05'
updated_date: '2026-08-31 18:42'
labels:
  - tool-packs
  - permissions
  - agents
  - console
  - security
dependencies:
  - TASK-26066
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent named Tool policy sessions from reading, persisting, or clearing approvals in the default or another profile by capturing and propagating one profile id across the control plane, builtin/MCP providers, and Console local, Virtual CLI, and raw-shell paths.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Session approvals are keyed by exact profile/server/tool triples, with profile-scoped clearing and backward-compatible clear-all behavior.
- [x] #2 Control-plane by-key gates and persistent writes carry the selected profile id while global kill-switch behavior remains profile-neutral.
- [x] #3 Builtin and MCP providers use one captured profile id for resolution, session approvals, and persistent approvals.
- [x] #4 Console local, Virtual CLI, and raw-shell closures capture and propagate the turn profile without mutating the default or another profile.
- [x] #5 The focused provider/controller matrix passes with signature-recording and cross-profile isolation coverage.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add red tests for exact session-approval triples, profile-scoped clearing, and no-argument clear-all compatibility.
2. Add provider/controller regressions proving every resolve, session, and persistent call receives the captured profile id.
3. Thread profile ids through control-plane by-key gates while keeping the global kill switch profile-neutral.
4. Update builtin and MCP providers plus Console local, Virtual CLI, and raw-shell closures to use one captured profile id.
5. Run the focused provider/controller matrix, scoped static checks, self-review, and independent review.

ADR required: no new ADR
ADR path: backlog/decisions/107-portable-tool-use-packs.md
Reason: Accepted ADR-107 already defines named-profile runtime propagation and approval isolation; this task implements that existing boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented exact profile/server/tool session approvals and propagated one captured profile id through control-plane, builtin/MCP providers, and Console local, Virtual CLI, and raw-shell paths while preserving profile-neutral kill-switch and raw-shell authority rules. Independent review fix round 1 added mutation-proven builtin session-read signature coverage; one nonblocking scoped-clear concurrency Minor is deferred. Fresh verification: all 10 new cross-runtime regressions passed; 325 provider/control-plane tests passed; scoped Ruff and diff/status passed. The broader requested matrix remains 11 failures/558 passes because 11 unrelated Console streaming/persistence failures reproduce at the Task 3 base; this plan deviation is documented and no Task 3 regression was found. ADR: existing ADR-107; no new ADR required.
<!-- SECTION:NOTES:END -->
