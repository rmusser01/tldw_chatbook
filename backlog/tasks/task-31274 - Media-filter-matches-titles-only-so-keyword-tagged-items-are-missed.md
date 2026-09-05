---
id: TASK-31274
title: Media filter matches titles only so keyword-tagged items are missed
status: Done
assignee: []
created_date: '2026-09-04 13:54'
updated_date: '2026-09-04 20:48'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 P2: filtering the list by `day2` — a keyword on four seeded rows — produced `Media (0)` and `No media matched ‘day2’.` (B cap_15). The filter appears to match titles only (cause suspected, not traced). The user's stated sequential-review scenario is a tag/keyword-filtered browse, so a keyword miss undercuts `Review these` over a tag scope, and the empty state does not say what was searched.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The fields the filter searches are documented in the task notes after a code trace
- [x] #2 Keyword matches are included in the filter, or an explicit `keyword:` syntax exists and the input placeholder says so
- [x] #3 The empty state names what was searched (e.g. `No media matched “day2” in titles or keywords`)
- [x] #4 Tests pin keyword matching and the copy
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace the filter to the DB search fields; 2. RED: real-DB keyword-only match at the scope seam, UI keyword filter shows the row, miss copy names the fields; 3. GREEN: keywords as an opt-in DB search field unioned with title/content, defaulted once at the scope-service seam; 4. Live 235x52
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Trace (AC#1): filter → LibraryMediaBrowseController._search (and _collect_review_pairs_from_scope for 'Review these') → MediaReadingScopeService.search_media → LocalMediaReadingService (search_fields=None) → search_media_db default ['title','content'] — keywords was not a valid field at all. Now the Library browse searches title, content and keywords (LIBRARY_BROWSE_SEARCH_FIELDS), defaulted once at the scope-service seam so the list and 'Review these' cannot diverge; the DB leg is opt-in (other search_media callers unchanged), parameterized, LIKE-escaped, EXISTS-based (no fan-out), count and page share the same WHERE. Miss copy: 'No media matched “x” in titles, content or keywords.'; placeholder 'Title/keyword…' (the longer copy painted as just 'Filter by' at the pane width). Relevance sort is unavailable while keywords are requested (FTS5 forbids a joined MATCH inside an OR) — the browse offers none today; documented. The shell static fake now matches keywords like production. Live: 'day2' → Media (3) with the three tagged rows; 'Review these' → 'Search: "day2" — 1 of 3'; 'zz' → the new miss copy with Clear filter. Deferred: pin the opt-in default, LIKE escaping and multi-keyword single-row properties; a timed count for the new filtered-browse count shape.
<!-- SECTION:NOTES:END -->
