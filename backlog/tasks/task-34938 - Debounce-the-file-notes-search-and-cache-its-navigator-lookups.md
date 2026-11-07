---
id: TASK-34938
title: Debounce-the-file-notes-search-and-cache-its-navigator-lookups
status: Done
assignee: []
created_date: 2026-10-13 14:57
labels:
- performance
- ui-perf
- wave-4-blocking-rebuilds
dependencies: []
priority: high
updated_date: 2026-10-13 18:47
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The file-notes search box has NO debounce (library_file_notes_workspace.py:6960): every keystroke runs 5 query_one DOM walks in _sync_navigator_mode, spawns an exclusive worker, issues an FTS replica.search (or, replica-less, re-casefolds EVERY entry per keystroke), and rebuilds a 100-row result tree. Every comparable search box in the repo is debounced (media 0.3s, artifacts 0.2s, emoji 0.3s) — this one was missed, and it is the highest frequency-x-cost rebuild path remaining.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Input.Changed debounced (repo-standard 0.2-0.3s) with in-flight generation guard
- [x] Navigator query_one lookups cached as widget refs (no DOM walks per keystroke)
- [x] Replica-less fallback no longer re-casefolds all entries per keystroke (pre-folded index)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the file-notes search handler + _sync_navigator_mode + replica-less fallback
2. Debounce Input.Changed (repo-standard) + cache the five navigator query_one refs
3. Pre-fold paths for the replica-less fallback
4. Tests + targeted runs + notes
<!-- SECTION:PLAN:END -->

## Implementation Notes

- `Input.Changed` on the search box now debounces execution through `FILE_NOTES_SEARCH_DEBOUNCE_SECONDS = 0.2` (repo set_timer/partial idiom, matching artifacts/emoji): the mode flip, worker spawn, and query run once typing pauses; rapid changes cancel earlier timers so only the latest query runs (the existing `_search_generation` guard still fences stragglers). Clearing the box still restores the files navigator IMMEDIATELY, and `Input.Submitted` bypasses the debounce.
- `_navigator_surface_refs()` caches the five navigator surfaces (search row, files tree, results tree, session-changes button, git panel); display toggles keep them attached, and an `is_attached` staleness check re-queries after any rebuild. `_sync_navigator_mode` and the empty-query path now do five attribute reads instead of five `query_one` DOM walks per keystroke.
- Replica-less fallback: `_entries_folded` is rebuilt at both `_entries` assignment sites and the fallback zips entries with their pre-folded paths — no per-keystroke casefold over every entry.
- Tests: new `Tests/UI/test_library_file_notes_search_debounce.py` (5 tests: debounce fires once with the latest query after rapid changes, immediate clear, submit bypass, folded fallback case-insensitive matching). 5/5 pass. Regression: `Tests/UI/test_library_file_notes_workspace.py` is at its exact pre-existing baseline (10 failed / 152 passed with the change stashed).
- ADR required: no — input debounce + ref caching inside one widget.
