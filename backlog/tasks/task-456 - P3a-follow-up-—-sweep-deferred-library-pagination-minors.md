---
id: task-456
title: P3a follow-up — sweep deferred library-pagination minors
status: To Do
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
- [ ] #1 Dead _character_record (0 callers) removed from personas_screen.py,Stale LIBRARY_FTS_THRESHOLD constant removed and fetch_character_names docstring reference dropped (search_characters_fts stays — still used by the handler/tests),Count-cache writes (_character_total/_count_cache_key) moved below the freshness guard in _reload_character_page so a superseded reload cannot leave a stale (search,tag) key,personas sort cycle excludes 'relevance' when active_mode != characters (currently harmlessly remapped)
<!-- AC:END -->
