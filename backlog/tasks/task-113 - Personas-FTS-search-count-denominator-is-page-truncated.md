---
id: TASK-113
title: Personas FTS search count denominator is page-truncated
status: Done
assignee: []
created_date: '2026-06-11 03:23'
updated_date: '2026-06-27 20:04'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When the FTS path is active the 'n of m' count uses the truncated loaded-page length as m although FTS searched the full corpus. Show an accurate or unambiguous count.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Count line is accurate or explicitly unbounded when FTS is active
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the FTS-count bug with a mounted Personas search test that forces the FTS path and asserts the count line does not claim the loaded-page denominator.
2. Implement the smallest count-copy extension in PersonasLibraryPane so filtered counts can be rendered with an unknown/full-library denominator.
3. Pass the unknown-denominator mode from PersonasScreen only for character FTS search results; leave in-memory character filtering and persona profile filtering unchanged.
4. Run the focused failing test, the Personas search group, the full Personas workbench test file, and git diff hygiene.

ADR required: no
ADR path: N/A
Reason: bounded Personas UI copy/count correctness fix; no storage/schema, service boundary, sync policy, or long-lived architecture decision.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added explicit full-library count copy for filtered results with an unknown denominator: `Showing 1 character match from full library`.
- Wired the unknown-denominator mode only for character FTS searches, where the loaded page may be truncated but the query searches the full local character corpus.
- Left local in-memory character filtering and persona profile filtering on the existing `n of m` copy because those denominators are accurate.
- Added a mounted regression that forced the FTS path and first failed against the old `1 of 2 characters` copy.
- Verification: `.venv/bin/python -m pytest -q Tests/UI/test_personas_workbench.py::TestSearch::test_fts_search_count_uses_unbounded_full_library_copy --tb=short`, `.venv/bin/python -m pytest -q Tests/UI/test_personas_workbench.py::TestSearch --tb=short`, `.venv/bin/python -m pytest -q Tests/UI/test_personas_workbench.py --tb=short`, and `git diff --check` all passed.
- ADR check completed: no ADR required for this bounded UI copy/count correctness fix.
<!-- SECTION:NOTES:END -->
