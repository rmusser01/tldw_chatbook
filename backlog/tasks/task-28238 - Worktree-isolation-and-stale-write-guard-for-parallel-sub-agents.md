---
id: TASK-28238
title: Worktree isolation and stale-write guard for parallel sub-agents
status: To Do
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
- [ ] #2 fs_write/fs_edit refuse when the target changed since the tool last read it, naming the conflict
- [ ] #3 Single-agent behavior is unchanged by default
- [ ] #4 The refusal path is exercised by a test that races two writers
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

Not yet planned/implemented. Implementation is unblocked (no dependency on the
mcp-unified 0.3.0 release that gates 25900).
