---
id: TASK-19637
title: Atomically pin local-tool workspace execution
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-20 20:05'
updated_date: '2026-08-28'
labels:
  - security
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent workspace-root rename or replacement races from retargeting local filesystem and Git tool operations after confinement checks. The current Path-based boundary predates project instructions; selected bindings now add defense-in-depth identity checks, but fully atomic confinement needs a cross-platform execution lease.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Filesystem and Git tool operations remain bound to the originally authorized workspace identity across concurrent root rename replacement symlink junction and reparse-point attempts
- [ ] #2 Mutating tools cannot write outside the originally authorized root under deterministic check/use race tests
- [ ] #3 Read-only tools cannot return content from a replacement root under deterministic check/use race tests
- [ ] #4 The solution works or fails closed on macOS Linux and Windows without unsafe preexec_fn use
- [ ] #5 An ADR records the helper-process or alternative runtime boundary and its lifecycle failure and performance trade-offs
- [ ] #6 Existing configured-workspace and selected-binding tool behavior remains compatible when no drift occurs
<!-- AC:END -->

## Renumbering provenance

This task previously held id TASK-16324, colliding with the older
"Atomically-pin-local-tool-workspace-execution" task that arrived on dev first.
Per the owner rule decided 2026-08-21 in TASK-19601 (**older id keeps it;
the younger task renumbers with a provenance note, regardless of Done
status**), it renumbered to TASK-19637. Citations to TASK-16324
in already-merged commit messages, ADRs, or code comments written before
2026-08-21 refer to THIS task; the other TASK-16324 holder is the
older arrival and keeps the id.

## Prior completed foundation

The project-instruction foundation is already complete in
`backlog/tasks/task-16320 - Add-startup-AGENTS.md-project-context-to-Console.md`.
Its governing decision is
`backlog/decisions/069-console-project-instruction-local-state-and-preflight.md`.
The old ambiguous `TASK-16320` dependency was removed during the TASK-19637
renumber; this task must not depend on the unrelated duplicate TASK-16320.

## Implementation Plan

ADR required: yes

ADR path: `backlog/decisions/101-one-shot-pinned-workspace-tool-execution.md`

Reason: This task establishes a cross-platform helper-process security boundary
for local filesystem and Git execution, including authority, lifecycle, failure,
privacy, compatibility, and performance contracts.

Detailed plan:
`Docs/superpowers/plans/2026-08-28-task-19637-atomic-local-tool-workspace-execution.md`

1. Share fail-closed canonical directory identity capture and define a strict,
   bounded one-operation stdin/stdout protocol.
2. Deliver one end-to-end contained worker with platform root pinning and a
   deterministic `stat_path` root-replacement regression.
3. Move read-only filesystem operations behind the retained root identity while
   preserving sensitive-path, symlink, ordering, and truncation behavior.
4. Move file writes, edits, and multi-file patch application behind one retained
   root identity.
5. Run all read-only Git operations from a relative pinned working directory,
   with no `git -C` and no inner process-group escape.
6. Route Local Tool, Virtual CLI, and external MCP production frontends through
   the same one-shot executor without changing schemas or permissions.
7. Publish deterministic Linux/macOS/Windows race/lifecycle evidence and
   direct-versus-worker median/p95 measurements.
8. Rebase, self-review, run approved verification, accept ADR-101, document the
   evidence, and close the task only after every acceptance criterion is proven.
