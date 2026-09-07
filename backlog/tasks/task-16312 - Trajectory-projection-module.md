---
id: TASK-16312
title: Trajectory projection module
status: Done
assignee: []
created_date: '2026-08-15 00:16'
updated_date: '2026-08-15 05:48'
labels: []
dependencies:
  - TASK-16311.1
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pure Chat/trajectory.py folding messages+usage+sidecar+variants+compaction into TrajectorySnapshot (turns/records, nesting, NULL timing, active-path variants). Plan task 3 in Docs/superpowers/plans/2026-08-14-console-trajectory-view.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 derive_trajectory unit tests pass,No Textual dependency,Legacy fallback grouping works
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented in commit 74bb35389: pure Chat/trajectory.py projection, no Textual imports, legacy fallback grouping; 22 tests green; see Implementation Notes and ADR-066

- **Approach**: pure projection implementing ADR-066
  (`backlog/decisions/066-console-trajectory-view-and-trace-metadata.md`) — the module
  never queries the DB and never imports Textual (stdlib + `ProviderUsage` only; enforced
  by a source-scan import-ban test). `derive_trajectory(messages, usage_by_id, traj_rows,
  variant_sets, compaction_records, active_leaf_message_id=None) -> TrajectorySnapshot`
  folds everything into frozen `TrajectoryTurn`/`TrajectoryRecord` dataclasses.
- **Key files**: `tldw_chatbook/Chat/trajectory.py` (new);
  `Tests/Chat/test_trajectory_projection.py` (22 tests, green — grouping, tool nesting,
  NULL timing, seq tie-breaks, variants, compaction placement, legacy fallback, purity ban).
- **Decisions**: active path walks `parent_message_id` from `active_leaf_message_id` to
  the root over the full map (soft-deleted nodes traversed, not emitted) — added as a
  trailing optional 6th parameter per the spec's binding note, so the brief's 5-arg shape
  still works. Variants merge two sources: tree siblings off the active chain, and
  `variant_sets` superseded contents attached at turn granularity (a variant set
  identifies a turn, not a message). Turn-grouping precedence: sidecar `turn_id` →
  in-memory `turn_id` → legacy adjacency (user opens a turn; assistant-first opens its
  own; same-second ties break user-first only for messages with no turn identity).
  Record `seq` is the 1-based ledger render position (sidecar seq still drives ordering);
  compaction markers render from `list_auxiliary_attempts`-shaped rows as `message_id=None`
  depth-0 entries. Never fabricate: NULL timing stays `None`, no duration is computed.
- **Deviations**: inputs are duck-typed via a single `_field` seam (Mapping /
  `sqlite3.Row` / attribute objects) so tests run on local stand-ins; a manual end-to-end
  sanity pass against the real `TrajectoryRowRead`/`ConsoleVariantSet` shapes was done.
  Tests live at repo-root `Tests/Chat/` per repo convention (brief's
  `tldw_chatbook/Tests/...` path does not exist).
<!-- SECTION:NOTES:END -->
