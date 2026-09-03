---
id: TASK-28238
title: Worktree isolation and stale-write guard for parallel sub-agents
status: In Progress
assignee: []
created_date: '2026-09-02 06:39'
labels:
  - agents
  - safety
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred rows C3+C7 combined, promoted by TASK-26041: their shared precondition shipped — fleet_coordinator runs concurrent sub-agent children against ONE shared working tree, so concurrent-edit races are now a live risk, not a future one. Two complementary halves: (a) optional per-child git-worktree isolation like hermes; (b) a stale-write guard on the fs tools (read-version tokens; a write whose base changed since read refuses with a diff) — optimistic locking exists only for notes today (Tools/note_management_tools.py:318).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A fleet child can opt into an isolated worktree; its changes merge back explicitly, never silently
- [x] #2 fs_write/fs_edit refuse when the target changed since the tool last read it, naming the conflict
- [x] #3 Single-agent behavior is unchanged by default
- [x] #4 The refusal path is exercised by a test that races two writers
<!-- AC:END -->

## Renumbering provenance

This task previously held id TASK-28228, colliding with the
"MCP-wire-server-initiated-sampling-elicitation-to-the-live-chat-provider-and-approval-surface" task that arrived on origin/dev first (dev minted 28226-28228
within the hour after this batch's sweep; re-verified at the wave-2 dev merge,
2026-09-02). Per the TASK-19601 owner rule the younger task renumbers with
provenance; it is now TASK-28238.


## Design 2026-09-02 (approved, phased)

Design approved via brainstorming. Spec:
backlog/docs/2026-09-02-task-28238-parallel-subagent-safety-design.md

Scoped as two phases under this task:
- Phase 1 (primary): stale-write guard on fs_write/fs_edit/fs_patch, keyed by
  (run_id, resolved_path) -- NOT per-provider-instance, because fleet children
  share one provider (a per-instance ledger would mask a sibling's race). Auto-
  records each fs_read's whole-file hash (incl. an ABSENT sentinel); refuses at
  execution when a file read this run changed on disk, naming the conflict;
  blind writes/creates proceed (single-agent unchanged). No git dependency.
- Phase 2 (later, sketch): opt-in git-worktree isolation per fleet child with
  explicit (never silent) merge-back; refuses cleanly if the workspace is not a
  git repo.

## Implementation Notes (Phase 1)

Phase 1 (stale-write guard) implemented on branch `feat/task-28238-stale-write-guard`.

**Mechanism**: read-ledger keyed by `(run_id, resolved_path) -> (sha256, size) | ABSENT`; provider-side record on fs_read (via provider-local path resolution + whole-file hash, independent of fs_read success/failure); staleness check at write-execution time (after approval) with refusal naming conflict; fs_write injects `expected_sha256` into the existing atomic CAS (reusing `write_file`'s per-target lock), fs_edit/fs_patch pre-hash each target and refuse the whole operation if any is stale; post-write re-stamp of every target.

**Files touched**:
- NEW `tldw_chatbook/Agents/fs_read_ledger.py` (ledger module, 7 tests)
- MODIFIED `tldw_chatbook/Agents/local_tool_provider.py` (record on read, check on write/edit/patch, update on success, +13 guard tests)

**AC coverage**:
- AC#2 (refusal naming conflict): `test_two_writer_race_refuses_second_writer`, `test_edit_race_refuses_second_writer`, `test_patch_with_one_stale_target_refuses_whole_patch`, `test_absent_then_created_by_peer_refuses`
- AC#3 (unchanged by default): `test_blind_write_proceeds_unchanged`, `test_edit_without_prior_read_proceeds`, `test_model_supplied_precondition_wins_over_ledger`, plus two own-chain tests
- AC#4 (racing writers): `test_two_writer_race_refuses_second_writer` (sequential `use_run_id("A")`/`use_run_id("B")` binding against shared provider)

**Phase 2 remains**: AC#1 (worktree isolation) unticked; dispatch-wiring open question (shared single registry) recorded in spec §Open questions.
