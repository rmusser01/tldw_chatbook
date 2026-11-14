---
id: TASK-34904
title: One engine for the search_library_*_page family
status: Done
created_date: 2026-10-12 17:53
labels:
- simplification
assignee:
- '@zcode'
updated_date: 2026-10-15 02:32
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
One engine, four surfaces. DB/library_search_engine.py (~300 lines): LibrarySearchSpec (table/filters/exact column/substring+FTS branches with matched_fields vocabulary entries incl None for probe-attributed FTS/keyword branch/page columns/tiebreaker/injected fetch+project+error callables) + library_search_page (COUNT+page+attribution in one transaction; conn.execute-only so both raw connections and ChaCha's cursor-wrapper transaction yield work) + build_lexical_group/matched_fields_from_hits shared with Notes. Media Prompts Conversations delegate fully; Notes (multi-selector AND-shape) routes its lexical group + attribution through the engine. REVIEW ROUND 1 (subagent in Qodo's place): verdict With-fixes -> ALL APPLIED: the ten dead helpers/constants deleted (the half-harvested quartet tails: _library_{media,prompt,note,conversation}_fts_query + their token-limit constants + the note/conversation escape twins; Media/Prompts escape helpers kept - other seams use them), unused dataclasses.field import dropped, _fts_indexes hoisted above the item loop, Notes' unused fts_query unpack underscored, and the TASK-34784 uncorrelated-subquery rationale restored on the conversations spec so the shape cannot be innocently regressed. REVIEWER'S OWN EVIDENCE: sqlite3 trace callbacks captured every executed statement on base-vs-head - byte-identical after whitespace normalization across all four surfaces (incl Prompts' FTS probe statements and Notes' AND-selector combo); ~30 behavioral probes identical (attribution/exclusions/hostile inputs/error types+messages+cause chains); matched_keywords NOCASE order preserved. Known bring-up bugs (fixed + documented in the engine): OR-group parenthesization, hit-param double-bind, ESCATE typo. Split-brain cap constants (engine vs host class constants) noted as a follow-up polish, not a blocker. VERIFICATION: 917-test surface suite green except the 3 stash-verified env failures; checker green; app import ok.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
