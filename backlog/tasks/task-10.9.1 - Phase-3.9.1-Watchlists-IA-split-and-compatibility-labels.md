---
id: TASK-10.9.1
title: 'Phase 3.9.1: Watchlists IA split and compatibility labels'
status: Done
assignee: []
created_date: '2026-05-08 03:36'
updated_date: '2026-05-08 04:18'
labels:
  - product-maturity
  - phase-3-9-library-collections
  - watchlists
dependencies:
  - TASK-10.8
references:
  - Docs/superpowers/specs/2026-05-08-library-collections-ia-split-design.md
parent_task_id: TASK-10.9
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rename the user-facing combined W+C destination to Watchlists while preserving compatibility route IDs and monitored-source workflows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Shell destination metadata command palette navigation and help copy show Watchlists instead of W+C or Watchlists+Collections
- [x] #2 The Watchlists destination body no longer renders Collections sections summaries or collection management copy
- [x] #3 Home and Console visible active-work labels say Watchlists while preserving existing route and payload compatibility where needed
- [x] #4 Focused regressions cover navigation command palette Watchlists body copy and active-run follow-through
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Run the current Watchlists/W+C baseline. 2. Add red tests for Watchlists labels and compatibility. 3. Update shell metadata, command palette, Watchlists screen, Home, and Console visible copy. 4. Run focused verification and diff hygiene. 5. Check ACs, add implementation notes, and mark Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the Phase 3.9.1 Watchlists IA split while preserving compatibility route IDs and historical wc/watchlists_collections selectors. Updated shell destination metadata, tab display labels, command palette help, Watchlists destination copy, local snapshot handoff payloads, Home active-work copy, and Console live-work readiness to use user-facing Watchlists language. Removed visible Collections/read-it-later loading from the Watchlists destination so Collections can move under Library in the next slice. Added a direct app-level navigation fallback for programmatic presses on hidden Textual chrome buttons to keep mounted replay tests deterministic without changing normal visible click handling. Verification: red tests were observed first; final focused suite passed with 94 passed, extra navigation smoke passed with 2 passed, and git diff --check passed.
<!-- SECTION:NOTES:END -->
