---
id: TASK-32601
title: Restore the authoring-only Workflows editor on dev
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-15 04:09'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved portable workflow editor on current dev while keeping unfinished execution and SQLite ownership changes isolated.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Users can create, edit, save, import and export workflow definitions in the canonical Workflows screen under ADR-138's approved stable-file exchange contract.
- [ ] #2 The library, step navigator, overview and collapsed continuous form remain usable with keyboard and at supported terminal sizes.
- [ ] #3 Drafts survive navigation and restart, failed persistence stays recoverable, and opaque server fields survive round trips.
- [ ] #4 Only existing SQLite safety utilities are used; new execution and process-ownership infrastructure remain absent.
<!-- AC:END -->

## Implementation Plan

ADR required: yes — amendment to existing ADR-138, not a new ADR number.
ADR path: backlog/decisions/138-portable-workflow-definitions-and-local-execution.md; ADR-125 and ADR-150 also apply.
Reason: record the user-approved stable-file exchange contract (2026-09-15) for authoring on current dev; runtime and the unapproved helper-owned-lock proposal are excluded.

1. Preserve the original and integration branches; use the clean codex/workflows-authoring-dev worktree based on 77eb2601a6.
2. Follow Docs/superpowers/plans/2026-09-14-workflows-authoring-dev.md. Reuse the reviewed editor-only checkpoint b34eda3d64 and the lossless document/draft code.
3. Register workflow document storage through current private_sqlite utilities. Retain existing migrations for file compatibility, but omit runtime locks and execution APIs.
4. Wire the editor into the real app, including durable draft flushing before navigation/quit and explicit local JSON import/export. Preserve current Console follow behavior.
5. Check the current tldw_server dev definition contract; verify targeted storage, authoring, lifecycle, keyboard and production-CSS behavior. Review actual captures at 160x48, 110x36 and 60x20.
6. Record exact verification and independent review. Do not mark Done with unresolved failures or unwaived static debt.

## ID provenance

The CLI offered TASK-32591; the all-ref object-path and 38-worktree scans already contained IDs through 32600. Only this newly created file/header was moved to the checked-free TASK-32601 before implementation. No existing task was renumbered.

## Implementation Notes

**Current review gate:** In Progress. On 2026-09-15 the user approved retaining
file-picker exchange under the stable-file assumption now recorded in ADR-138,
the spec, plan and user guide. Existing metadata-only alias rejection remains;
post-validation substitution or a detached live inode can still disrupt SQLite
locking. This is an accepted operating limitation, not a technical race fix.
No extra helper or file-I/O subsystem is authorized or added. The original
reviewer's contract disposition and final whole-branch review are pending;
baseline static-analysis debt remains unwaived. No Done transition.

Implemented the reviewed b34eda3d64 authoring slice: real library, navigator,
overview and independently collapsed continuous forms; immutable saved revisions,
durable recoverable drafts, explicit private JSON import/export and disabled Run.
The app owns lazy document/draft setup and drains across navigation/quit. Existing
Console follow is a separate stable region. No execution API, runtime model graph,
new helper protocol, schema v5, server/provider writes or model calls were added.

WorkflowsDB uses the existing `connect_private_sqlite` seam, ordinary transactions
and close; v0-v4 migration bytes are preserved. Added only the domain registration
and owner census row. Tests exercise real SQLite foreign writers, persistence,
cancelled setup/close/exchange, failed flush retry, real app navigation/quit and
actual file pickers. The user guide documents privacy and local/server validation
limits. ADR-138 (including its approved stable-file amendment), ADR-125 and
ADR-150 remain the applicable decisions; no new ADR number or shared boundary.

Final targeted verification: 355 passed (authoring/storage/editor/quit/token/
bundle/census) plus 19 passed, 276 deselected (affected Workflows destination and
Console cases). The two visual findings now have the retained reviewer's scoped
ship verdict: precise draft-versus-revision status and Prompt label/value/focus
painted together at 60x20. Final interaction fixes serialize widget reconciliation,
retain raw-editor focus visibility and settle test modal mounting. Full failing
test names, diagnoses, exact RED/GREEN commands and final evidence are in
`.superpowers/sdd/2026-09-14-workflows-authoring-dev/task-1-report.md`.

Task remains **In Progress** pending coordinator code review and full DoD.
Existing whole-file Ruff/formatter debt is reported separately, not waived or
suppressed. New/rewritten authoring files pass Ruff and formatting. All work is
confined to the named authoring-dev worktree; preserved checkouts are untouched.

Code-review fix round 1 adds a feature-local, metadata-only exchange precheck for
database/sidecar aliases. Real DELETE/WAL foreign-writer tests first reproduced
lock loss after a refused import, then verified preserved exclusion. Import and
export reject aliases before generic file I/O; shared helpers, SQLite lifecycle,
migrations and UI are unchanged. The report records exact RED/GREEN evidence and
the concurrent-path-replacement limitation. The covering authoring/storage suite
passes 35 tests, including off-loop metadata checks for import and export; scoped
Ruff/format and diff-check are clean. Task remains In Progress for review.
