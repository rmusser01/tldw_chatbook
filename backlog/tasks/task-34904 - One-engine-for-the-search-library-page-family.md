---
id: TASK-34904
title: One engine for the search_library_*_page family
status: Done
created_date: 2026-10-12 17:53
labels:
- simplification
assignee:
- '@zcode'
updated_date: 2026-10-15 01:56
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
search_library_media_page search_library_prompts_page search_library_notes_page and search_library_conversations_page are one engine copied four times (575 LOC of methods plus ~250 LOC of per-entity helper quartets). Only the column list and the item projector differ.

Evidence (verified on dev df4985e): Locations: Client_Media_DB_v2.py:8974-9083 Prompts_DB.py:3558-3686 ChaChaNotes_DB.py:18767+ and 19170+. Each carries its own _escape_library_*_like _library_*_fts_query _library_keywords_for_* _library_*_item quartet. Shared docstring structure confirms common origin.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 DB/library_search_engine.py: library_search_page + LibrarySearchSpec + build_lexical_group/matched_fields_from_hits own the match-branch machinery (exact + escaped-LIKE + conditional FTS + keyword-last) hit_N evidence projection COUNT+page in one transaction and matched attribution,Media Prompts and Conversations delegate their full bodies to the engine (specs carry columns FTS branches incl conversations' dual title/message FTS archive clause validators error constructors),Notes routes its LEXICAL GROUP and hit attribution through the engine (its multi-selector AND-shape stays in the host - documented deviation),Prompts per-column FTS probes preserved via the spec's fts_attribution hook (FTS branch field=None so the hit attributes nothing by itself exactly as the original),DEVIATION documented: the branch-eligibility projector never touched this seam (the review claim was wrong - it serves the separate CharacterConversationSearchRepository) so no hook was needed,All four surfaces' wire pins green (539-test surface suite at baseline + conversations equivalence classes + fts quoting seams)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Baseline the four surfaces' test suites 2. Map the four search_library_*_page bodies + helper quartets exactly 3. Design the engine (column specs + item projector + FTS-attribution and branch-eligibility hooks) 4. Migrate all four surfaces 5. Tests vs baseline 6. Review round + fixes + notes
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
One engine, four surfaces. DB/library_search_engine.py (~300 lines): LibrarySearchSpec (table/filters/exact column/substring+FTS branches with matched_fields vocabulary entries incl None for probe-attributed FTS/keyword branch/page columns/tiebreaker/injected fetch+project+error callables) + library_search_page (COUNT+page+attribution in one transaction; conn.execute-only so both raw connections and ChaCha's cursor-wrapper transaction yield work) + build_lexical_group/matched_fields_from_hits shared with Notes. Media Prompts Conversations delegate fully; Notes (multi-selector AND-shape) routes its lexical group + attribution through the engine while its exact-keyword (BINARY + pending-receipts leg) and folder selectors and organization projection stay in-host. Engine bug caught by the conversations equivalence tests during bring-up: the OR group lost its parentheses in the filter join (Media/Prompts tests under-caught it - AND bound only the first branch) - fixed with a regression note in the engine. Also fixed en route: hit-branch params bind twice (projection + WHERE); ESCATE typo in Notes' spec. Review-claim correction: SelectedBranchEligibilityProjector never touched this seam (serves the separate character-conversation corpus) - AC hook item dropped as moot. Inventory re-frozen via sanctioned --write (engine +2 diagnostics; Media/ChaCha -1 each from the consolidated error logs). VERIFICATION: full 539-test surface suite + Media_DB + Prompts_DB dirs vs baseline - only the known env failure (stash-verified pre-existing at HEAD); checker green; app import ok.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
