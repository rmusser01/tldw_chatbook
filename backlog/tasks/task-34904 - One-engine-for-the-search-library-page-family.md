---
id: TASK-34904
title: One engine for the search_library_*_page family
status: To Do
created_date: 2026-10-12 17:53
labels:
- simplification
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
search_library_media_page search_library_prompts_page search_library_notes_page and search_library_conversations_page are one engine copied four times (575 LOC of methods plus ~250 LOC of per-entity helper quartets). Only the column list and the item projector differ.

Evidence (verified on dev df4985e): Locations: Client_Media_DB_v2.py:8974-9083 Prompts_DB.py:3558-3686 ChaChaNotes_DB.py:18767+ and 19170+. Each carries its own _escape_library_*_like _library_*_fts_query _library_keywords_for_* _library_*_item quartet. Shared docstring structure confirms common origin.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One search engine takes column specs an item projector and optional branches,All four surfaces route through it,Prompts per-column FTS attribution and the conversation branch-eligibility projector preserved as hooks,Dedicated per-surface tests pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
