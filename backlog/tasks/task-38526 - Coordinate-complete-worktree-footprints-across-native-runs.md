---
id: TASK-38526
title: Coordinate complete worktree footprints across native runs
status: Done
created_date: 2026-11-01 17:30
dependencies:
- TASK-38524
labels:
- console
- agents
- workflows
references:
- Docs/superpowers/plans/2026-11-01-native-phased-agent-workflows-implementation.md#task-3
documentation:
- Docs/superpowers/specs/2026-11-01-native-phased-agent-workflows-design.md
- backlog/decisions/285-native-phased-agent-workflows.md
assignee:
- '@codex'
updated_date: 2026-11-01 19:35
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver implementation slice 3 of the accepted native shared-worktree workflow design so coordinate complete worktree footprints across native runs. Keep existing execution, permission and persistence ownership; this task does not authorize unrelated changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Canonical aliases, nested roots and read/write dependencies share correct nonblocking activity claims.
- [x] #2 Opaque-capable work claims runtime-wide exclusivity before root-specific reservations; refusal creates no partial-lock deadlock.
- [x] #3 Claims remain held until positive physical drain and conclusive durable settlement, including ordinary runs and survivors.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/285-native-phased-agent-workflows.md
Reason: Implements the accepted runtime-owned activity and physical-drain contract; no new ownership, execution, accounting or storage boundary.

1. Resolve footprint-before-child-capacity ordering with controller; retain actual ExecutionOwner lifetimes without changing capacity accounting.
2. Add real-path, complete-admission, opaque, refusal-unwind and held-worker/child settlement tests; capture RED before implementation.
3. Implement pure all-or-none activity reservations with existing directory identities, positive drain plus durable settlement, and lazy ConsoleRuntime ownership.
4. Run focused new tests, the three named existing native regressions, and targeted static checks; self-review and commit only owned paths.
5. Leave status/AC pending independent controller review and record full interfaces, evidence and limitations in task-3-report.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented complete, runtime-owned read/write/opaque activity reservations under ADR-285. Immutable resolver-admitted scopes reuse captured DirectoryChain metadata and target stat identities; aliases, nested roots, hardlinks and conservative casefold overlap serialize correctly. prelaunch reserves the full footprint before native capacity and unwinds only never-bound reservations; bound claims require positive native drain and durable per-owner settlement, including separate child settlement handles. ConsoleRuntime constructs its activity owner lazily under the existing capacity lifetime lock.

Controller-approved interface refinements: prelaunch(...)/bind_owner(...) resolves child-capacity ordering without a phantom owner; attach_child(...) returns a child-specific settlement handle. A metadata-only ExecutionOwner.is_admitted query rejects fabricated/retired owners outside the activity lock; no capacity accounting or dispatch algorithm changed. Case-only names may serialize on case-sensitive filesystems. Actual ordinary-turn and phase wiring remains Task 8.

Evidence: initial RED on missing module; targeted reparse RED (1) and fabricated/retired membership RED (5), then GREEN. Final exact activity/capacity/fleet/finalization target set: 99 passed in 4.05s using supported bootstrap_profile only on the three source-bound fleet nodes. Original unmodified invocation separately retained 84 pass/3 raw_source_selection_changed pre-body fixture failures; no guard reset or unrelated fixture mutation. New modules/tests Ruff clean, four non-runtime files formatter-clean, py_compile/diff-check pass. Existing runtime/capacity lint debt unchanged against BASE 78e297e8892061e587c7f9788d03197d61f0da98 (24/1 findings; zero introduced). Full report/evidence: .superpowers/sdd/2026-11-01-native-phased-agent-workflows-implementation/task-3-report.md. Status/AC intentionally remain pending independent review.
Independent spec/quality gate passed after Unicode collision correction2145552ea6 (base implementation9a491602ff). Complete prelaunch footprints bind genuine native owners after capacity; child settlement is per owner; release requires positive drain and conclusive durability. NFC/casefold comparison keys preserve admitted paths while conservatively colliding aliases. Final correction48 activity tests pass; earlier final qualified99 native/activity checks pass with supported bootstrap fixture only on exact3 source-bound fleet nodes. Original84 pass/3 raw_source_selection_changed remains explicit, not called green. Scoped lint/format/compile/diff checks pass with no introduced baseline findings. Unrelated index preserved. ADR-285 applies; actual ordinary/phase launch, shutdown and journal wiring remains Task8.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Runtime-owned activity reservation interfaces implemented and independently reviewed, with exact/native-owner retention and conservative alias collision. No live workflow scheduling is enabled by this slice.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
