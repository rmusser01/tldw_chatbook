---
id: TASK-24610
title: Sources names two different things in the same Inspect rail
status: Done
assignee:
  - '@claude'
created_date: '2026-08-30 00:54'
updated_date: '2026-08-30 02:46'
labels:
  - console
  - ux
  - inspector
  - critique-2026-08-29
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The staged-context tray heading Sources means staged context references. The Sources row under Source Readiness means retrieval status and reads 'Sources: not staged'. The pinned authority row means the first. The Run recipe line's sources summary means the second. All four are visible at once in a 33-column column, which makes this the largest comprehension cost in the rail. It is a rename, not a rebuild.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One noun in the Inspect rail refers to exactly one concept
- [x] #2 The retrieval status row is named for retrieval rather than for sources
- [x] #3 Sources means staged context consistently across the tray, authority row, status chip and rail handle badge
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Renamed the run inspector's retrieval-status row from 'Sources' to 'Retrieval', so 'Sources' means staged context everywhere in the rail -- tray heading, pinned authority row, status chip.

The rename was NOT a rename. Two things the task premise missed, both found by running it:

1. The label is also the OWNERSHIP KEY. classify_inspector_content is STRICT and RAISES UnownedInspectorContentError on a label it does not own, so simply renaming turned 36 tests red with 'Unowned Inspector content: row:Sources' -- and, more importantly, would turn any persisted or replayed pre-rename snapshot into a CRASH rather than a mislabelled row. 'Sources' is therefore retained in ROW_GROUPS/ROW_IDS as a classification alias that no producer emits.

2. The alias needs its OWN widget id. Sharing 'console-inspector-sources' between the two labels made a state containing both mount two widgets under one DOM id, which the exhaustive inventory test (one row per EXPECTED_ROW_OWNERS entry) hits immediately. 'Retrieval' now owns 'console-inspector-retrieval'; the alias keeps the historical id.

The AC said 'one noun refers to exactly one concept'. That guarantee is enforced against what PRODUCTION EMITS (ConsoleInspectorState.from_values), not against the ownership allowlist, and the test says so -- asserting the allowlist were alias-free would have forced deleting the crash guard.

project_console_send_authority looked the row up by the literal 'Sources'; it now tries 'Retrieval' first and keeps 'Sources'/'RAG/source' as fallbacks. Missing that would have left Run reading 'Ready' while retrieval was blocked, silently, in the one line pinned above the fold.

Modified: console_inspector_ownership.py, console_display_state.py, console_send_authority_summary.py, Tests/UI/test_console_run_inspector.py, Tests/UI/test_console_internals_decomposition.py.
<!-- SECTION:NOTES:END -->
