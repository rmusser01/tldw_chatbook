---
id: TASK-399.4.1
title: Bound File Notes rendering and large-file previews
status: Done
assignee:
  - '@codex'
created_date: '2026-08-12 00:02'
updated_date: '2026-08-12 00:45'
labels:
  - notes library ux performance
dependencies:
  - TASK-969
parent_task_id: TASK-399.4
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep very large File Notes roots and supported files responsive without hiding content or risking lossy copies.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A deep root and a 5,000-sibling folder initially mount only a fixed bounded row batch; keyboard-activating a literal Load more row adds only the next bounded batch while preserving expansion and selection.
- [x] #2 File and search navigator publication callbacks complete below 100 ms on the fixed benchmark trace, with disk scanning and exact export remaining off the UI thread.
- [x] #3 Files above the 200,000-character interactive ceiling never mount their full body editor; users see exact byte and character sizes plus a clearly labeled first-100,000-character read-only excerpt.
- [x] #4 Large-file exact export streams current disk bytes to an absent destination beneath the root, revalidates source identity, and never substitutes the excerpt or overwrites an existing file.
- [x] #5 Focused mounted tests, service tests, mutation evidence, Ruff, and diff checks pass; the legacy formatter baseline and unrelated branch failures are documented, and TASK-399.4 remains open for its other acceptance criteria.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This is bounded rendering and export safety within the existing disk-authority boundary recorded by ADR-029; it does not change storage, sync, ownership, or a cross-module contract.

1. Add deterministic batching and lazy folder materialization to the File Notes navigator.
2. Add a fixed read-only excerpt contract for oversized supported text files.
3. Add an exact, streamed, no-clobber export path that revalidates the source.
4. Cover large sibling sets, deep trees, narrow layouts, exact export, and stale-source safety.
5. Run focused regression, mutation, static, formatting-baseline, and diff checks; document evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented bounded File Notes rendering and exact large-file export.

- The tree and fallback-search navigator now publish at most 100 rows per batch, expose a keyboard-operable Load more row, materialize folders only as expanded, and restore expansion incrementally.
- Supported bodies above 200,000 characters now mount a labeled, read-only first-100,000-character excerpt with exact character and byte sizes instead of the full editor body.
- Export exact copy streams current disk bytes in 64 KiB chunks to an absent destination, preserves mode, verifies the opened hash and source identity before publication, and never substitutes the excerpt or overwrites an existing file.
- Updated the File Notes guide and added service/UI coverage for 5,000 siblings, 500 nested folders, wide/narrow excerpt layouts, no-clobber export, and stale-source rejection.
- Evidence: focused cross-layer matrix 8 passed; selected regression matrix 9 passed; tail UI/theme/compact/focus matrix 13 passed; service module 38 passed, 2 skipped, 1 deselected; Ruff check, compileall, and `git diff --check` passed. Mutation checks proved the 100-row performance guard and exact-export route fail when intentionally broken, then were restored.
- Baselines outside this child scope: `ruff format --check` remains red on the four touched legacy files due pre-existing formatting; the full service run has one Windows-only symlink privilege failure (`WinError 1314`); the full UI module exposes three existing regressions to address as separate atomic tasks (cache-reload warning retention, shell focus under an unavailable local service, and a reload-confirmation expectation mismatch).
- TASK-399.4 remains open because this child closes only its large-root/large-file performance acceptance criterion.

Documentation: `backlog/decisions/029-file-notes-disk-authority.md`; `Docs/User_Guide/library/file-notes.md`.
<!-- SECTION:NOTES:END -->
