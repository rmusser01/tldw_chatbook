---
id: TASK-15706
title: Render and operate the Database Notes folder navigator
status: Done
assignee: []
created_date: '2026-08-13 01:43'
updated_date: '2026-08-14 01:49'
labels:
  - notes
  - folders
  - ui
dependencies:
  - TASK-15705
references:
  - Docs/superpowers/specs/2026-08-12-notes-folder-import-sync-design.md
  - backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md
  - >-
    backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose the local folder foundation in Library Database Notes as a lazy hierarchical tree with Unfiled, multiple note placements, breadcrumbs, stable focus, semantic status text, and visible manual folder actions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Database Notes renders root folders, nested folders, notes, and Unfiled lazily with bounded bulk reads and no per-note service calls.
- [x] #2 A note in multiple folders has distinct placement rows that open one note identity and show the correct breadcrumb.
- [x] #3 Users can create, rename, move, remove, and restore folders; move notes; and add a note to another folder without deleting note content.
- [x] #4 Managed and restored-without-owner placements are visibly distinguished, color-independent, and protected from manual mutation.
- [x] #5 Selection, expansion, focus, and editor identity remain stable across refresh, reorder, resize, and compact 60x20 layouts.
- [x] #6 Database work runs off the Textual event loop for file-backed databases while in-memory test databases keep their thread-local safety.
- [x] #7 Rendered-frame, accessibility, host-routing, performance, and live restart tests pass without regressing the existing Database Note editor or File Notes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR paths: `backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md`, `backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md`
Reason: this task implements the folder UI, placement identity, ownership, and service boundaries already accepted by those ADRs; it does not introduce a new architectural choice.

1. Add a Textual-independent folder-tree projection that consumes bounded `NoteFolderPage` batches, preserves placement-aware identities, produces breadcrumbs and Unfiled rows, distinguishes managed/inactive-managed placements without relying on color, and reconciles expansion/selection/focus by stable identity.
2. Add focused unit tests for hierarchy, duplicate placements, generated-owner ancestor collapsing, manual duplicates, paging metadata, semantic labels, and stable state reconciliation; observe each test fail before implementing its behavior.
3. Extend `LibraryNotesCanvas` with a keyboard-operable lazy navigator plus compact folder action surfaces. Keep note buttons compatible with the existing editor/select handlers while carrying a separate placement identity and breadcrumb.
4. Add screen-owned async loading and mutation orchestration through `NotesScopeService`, including stale-result guards, bounded batch reads, file-backed off-loop behavior, thread-safe in-memory test handling, and actionable capability/error states.
5. Wire create, rename, move, remove, restore, move-note, and add-placement actions. Protect managed/inactive-managed placements from manual detachment or destructive folder actions, and refresh without losing surviving expansion, placement focus, note selection, or editor identity.
6. Add widget, screen, host-routing, accessibility, compact 60x20 rendered-frame, performance, and regression tests for Database Notes and File Notes. Correct the pre-existing multiselect fixture only if the touched handler requires it.
7. Run focused and broader Library/Notes suites, static checks, a real file-backed smoke path, and live TUI restart verification. Review the rendered 60x20 frame without color, document evidence and trade-offs, link ADR-059/073 in Implementation Notes, check every criterion, and only then set TASK-15706 to Done.

### Review Follow-up Plan

1. Rebase onto current `dev`, preserving both upstream and task-owned generated documentation changes.
2. Add failing regressions for authoritative managed-folder protection, filtering collapsed/unloaded placements, and selection reconciliation beyond the legacy notes page.
3. Fix each verified issue at its owning model/service/screen boundary, then rerun its focused tests before continuing.
4. Update task evidence and the PR, request automated and independent review, resolve every actionable thread, and merge only after required checks pass.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the local Database Notes folder navigator defined by
[`ADR-059`](../../backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md)
and
[`ADR-073`](../../backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md).
No new ADR was required because the task preserves those decisions and the
existing normalized `NotesScopeService` boundary.

- Added a pure bounded-page tree projection with lazy expansion, Unfiled,
  breadcrumbs, generated managed-ancestor collapse, exact membership-aware row
  identities, and placement-first focus reconciliation.
- Extended the Database Notes canvas with folder/note rows, semantic status
  text plus theme-token coloring, compact actions, duplicate-placement
  selection behavior, and managed/inactive-owner protections.
- Added screen orchestration for generation-gated loads, independent paging
  cursors, off-loop normalized service calls, typed errors, optimistic folder
  and membership mutations, safe attach-before-detach moves, and one-session
  folder restore receipts.
- Kept File Notes independent. A folder-load completion now updates only a
  still-mounted Database Notes canvas, preventing a late completion from
  rebuilding a source transition.
- Primary modified surfaces are `library_notes_tree_state.py`,
  `library_notes_canvas.py`, `library_screen.py`, the folder dialogs, both
  canonical/aggregate CSS files, and their focused Library/Notes tests.

Evidence:

- Red/green review regressions: five tests first failed for same-folder
  membership identity, duplicate checkbox synchronization, visible-tree Select
  all, and missing/error status repaints; all five passed after the fixes.
- Final combined navigator, folder repository/service, Database Note, File
  Notes service/owner, dialog/canvas, live host, and File Notes routing run:
  `453 passed in 50.47s`.
- The folder repository/service subset independently passed `397` tests; the
  related File Notes source-switch/compact suite passed `8` tests.
- Ruff passed for every new/directly modified module and test; eight task files
  pass `ruff format --check`; targeted `compileall` and `git diff --check`
  passed. `library_screen.py` retains the same 154 pre-existing Ruff findings
  as untouched `dev`, with no new finding from this task.
- The four navigator failure diagnostics were reviewed against the persistent
  logging boundary, changed from traceback capture to bounded operation/error
  type metadata, and added to the generated diagnostic inventory. The focused
  architecture and navigator gate passes `23` tests; the complete post-review
  Notes/Library plus diagnostic-architecture regression run passes `680` tests.
- A full Python 3.12 collection reaches all `42,613` tests. The full run was
  stopped after `1,609 passed` when it reproduced seven unchanged media-ingest
  test failures; the same seven fail on latest untouched `dev`. Latest `dev`
  also has unrelated pre-existing diagnostic-inventory drift across several
  recently merged modules, while this task's own inventory delta is reviewed
  and green on its current base.
- The final 60×20 SVG frame was rendered and visually reviewed: nested rows,
  focus, action labels, `⇄ Sync managed` / `⇄ Synced placement`, and
  `! Needs owner review` remain legible without color.
- One deeper File Notes editor-focus breakpoint test fails identically on this
  branch and untouched `dev` at the first 40×20 resize. Baseline comparison
  establishes that it is pre-existing; all source-transition cases affected by
  this work pass, so the branch introduces no File Notes regression.
- The final `dev` rebase review added authoritative managed-folder metadata and
  repository mutation guards, exact collapsed-branch search pages, tree-aware
  selection reconciliation, and a legacy-search fallback that cannot retain a
  prior query's records. Independent review then found and drove fixes for
  path-only folder searches and repeated descendant traversal in managed-state
  lookup: path matches now union bounded placements with content matches, and
  managed state walks from managed memberships toward loaded ancestors. The
  returned path matches are also merged into list state by note ID so result
  counts, Select all, and checkbox state agree with the rendered tree.
- Qodo's final review raised six findings. Google-style API documentation was
  completed for the three named callables, search bounds now use named policy
  constants, folder-only continuations skip exhausted note queries at both root
  and expanded scopes, and managed-placement collapse now memoizes folder
  ancestry instead of rescanning every membership pair. Regression tests cover
  the query skip, membership continuation, repository/service flag propagation,
  and bounded ancestry-walk cost.
- The final post-Qodo focused gate passes `282` tests, including repository,
  normalized service, pure projection, screen orchestration, multiselect,
  dialogs/canvas, and the reviewed-diagnostic metadata gate. Focused fatal/error
  Ruff checks, `compileall`, and `git diff --check` pass. Five broader diagnostic
  ledger failures reproduce identically on untouched latest `dev`; a broader
  File Notes run likewise has one focus-confirmation failure reproduced unchanged
  on latest `dev`.

Trade-offs: this task intentionally exposes the already-landed local folder
foundation. Server-backed folder capability and the import/sync setup flow stay
behind their later work items; the UI degrades to the existing flat Database
Notes list when the folder capability is unavailable.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Every acceptance criterion is checked with automated or recorded evidence.
- [x] #2 Focused tests, broader Library/Notes regressions, static analysis, and live TUI verification pass.
- [x] #3 Implementation Notes summarize the approach, files, trade-offs, and evidence.
- [x] #4 ADR-059 and ADR-073 are linked from the implementation plan and notes.
- [x] #5 A rendered-frame self-review confirms hierarchy and status remain legible at 60×20 without color.
- [x] #6 The task is set to Done only after all requirements above are complete.
<!-- DOD:END -->
