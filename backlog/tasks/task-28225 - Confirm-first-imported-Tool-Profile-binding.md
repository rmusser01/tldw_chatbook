---
id: TASK-28225
title: Confirm first imported Tool Profile binding
status: Done
assignee:
  - '@codex'
created_date: '2026-09-02 05:36'
updated_date: '2026-09-02 06:29'
labels:
  - tool-packs
  - binding
  - workspaces
  - security
dependencies:
  - TASK-28124
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Require an immutable current-state review and one-use confirmation whenever a workspace first adopts an imported Tool Pack profile, enforced at the central workspace mutation authority rather than only in presentation code.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every workspace assistant-default mutation path, including inline create, set, replace, clear, provisioning, and backfill, traverses one dependency-inverted Tool Profile guard; imported create/set/replace cannot bypass confirmation through a direct service call.
- [x] #2 Binding review recomputes current imported lifecycle, policy digest/revision, full intended-default digest, fallback and Allow posture from strict authority plus current inventory, and issues a process-local one-use 10-minute token bound to workspace, action, profile, policy, revision, and the complete intended defaults.
- [x] #3 Confirmation fails closed after policy/default/workspace/action changes, replay, expiry, removal, or concurrent mutation; lifecycle coordinator then store fence are held through the workspace SQLite commit, and the first-bind marker clears only after a known-successful exact binding.
- [x] #4 An uncertain workspace commit is reconciled from exact persisted defaults and returns binding_uncertain without clearing the marker unless the intended binding is proven; bind and removal cannot race past the same lifecycle serialization.
- [x] #5 The existing read-write memory acknowledgement remains independent, existing local and ws- profiles keep their behavior while traversing the guard, and targeted binding/workspace/provisioning tests plus scoped static checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing direct-service bypass, inline-create/set/replace/clear, provisioning/backfill, replay/expiry/mutation, and bind/removal race tests.
2. Define the dependency-inverted registry guard protocol and route every assistant-default transaction through one mutation scope.
3. Add immutable review/summary/token contracts that recompute current imported policy and bind tokens to the full intended defaults/action/workspace/profile identity.
4. Implement one-use final validation under lifecycle coordinator then store fence, hold both through the SQLite commit, and reconcile uncertain outcomes before clearing the first-bind marker.
5. Verify the independent read-write memory acknowledgement matrix and unchanged local/ws-profile behavior.
6. Run focused binding/workspace/provisioning and related authority tests, scoped Ruff/format, diff hygiene, self-review, and independent review.

ADR required: no new ADR
ADR path: backlog/decisions/107-portable-tool-use-packs.md
Reason: ADR-107 already fixes the first-bind review/token contract, workspace-registry authority boundary, lifecycle/store/SQLite lock order, independent memory acknowledgement, and uncertain-commit reconciliation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the dependency-inverted workspace guard and concrete imported-profile
binding authority described by ADR-107. All create/set/replace/clear, provisioning,
and backfill writes now enter lifecycle then permission-store serialization through
the registry, while local/profile-free writes retain their prior no-token behavior.

Binding review now derives current lifecycle, policy/revision, complete defaults,
fallbacks, exact/effective/unavailable/downgraded Allows, and known high-risk Allows
from strict authority plus live inventory. Confirmation uses bounded, pruned,
process-local one-use records and a full ten-minute post-confirmation token lifetime.
Commit holds the lifecycle/store fences through SQLite completion, reconciles exact
persisted defaults after uncertain outcomes, and clears the marker only for a proven
binding. Memory acknowledgement remains a separate gate.

Targeted TDD and related regression verification finished at 142 passing tests with
one pre-existing `RequestsDependencyWarning`; six-file Ruff lint/format and diff
checks passed. Independent review found and then verified fixes for two Important
and two Minor issues; re-review passed with no remaining findings.

ADR required: no new ADR. Existing ADR:
`backlog/decisions/107-portable-tool-use-packs.md`.

Modified files: `tldw_chatbook/Tool_Packs/binding.py`,
`tldw_chatbook/Workspaces/registry_service.py`,
`tldw_chatbook/Workspaces/agent_provisioning.py`, and their focused Tool Pack and
Workspace tests.
<!-- SECTION:NOTES:END -->
