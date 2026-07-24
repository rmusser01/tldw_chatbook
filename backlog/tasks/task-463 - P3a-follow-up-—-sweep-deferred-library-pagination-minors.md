---
id: TASK-463
title: P3a follow-up — sweep deferred library-pagination minors
status: Done
assignee: []
created_date: '2026-07-22 03:01'
labels:
  - tech-debt
  - personas
  - p3a
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Cleanup of Minor findings deferred from the P3a whole-branch review (PR #755). None are user-visible bugs; this is hygiene.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Dead _character_record (0 callers) removed from personas_screen.py,Stale LIBRARY_FTS_THRESHOLD constant removed and fetch_character_names docstring reference dropped (search_characters_fts stays — still used by the handler/tests),Count-cache writes (_character_total/_count_cache_key) moved below the freshness guard in _reload_character_page so a superseded reload cannot leave a stale (search,tag) key,personas sort cycle excludes 'relevance' when active_mode != characters (currently harmlessly remapped)
<!-- AC:END -->

## Implementation Plan

1. Verify each deferred minor against current dev (post-#755 drift expected)
2. Fix whatever remains + pin test
3. Document already-satisfied items with evidence

## Implementation Notes

Swept on branch claude/roleplay-followups-sweep. Three of the four minors were
already satisfied on dev by the time of the sweep (concurrent-session drift):
- Dead `_character_record`: already absent (only `_full_character_record`
  remains, 4 call sites).
- `LIBRARY_FTS_THRESHOLD` + docstring ref: zero grep hits repo-wide.
- Count-cache-below-guard: `_reload_character_page` already commits
  `_character_total`/`_count_cache_key` after the post-await freshness guard
  ("Commit shared count state only after the guard passes" comment in place —
  fixed during the P3a pre-merge review).

Fixed here: `_character_sort_cycle` now prepends "Relevance" only when a
search is active AND `active_mode == "characters"` (personas page in-memory
with no FTS; previously the option appeared and was silently remapped to
name_asc). Pin test `test_sort_cycle_excludes_relevance_outside_characters_mode`
in Tests/UI/test_personas_library_scale.py (8 passed).
