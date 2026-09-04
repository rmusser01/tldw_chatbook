---
id: TASK-31203
title: 'Library decomposition wave 3: combined search+RAG series'
status: Done
assignee: []
created_date: '2026-09-03 05:55'
updated_date: '2026-09-04 02:19'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave-2 Task 8's entanglement gate fired: 8/14 search methods (57.1%) call or are called by RAG methods; search submit is structurally the RAG entry point (_start_library_rag_query). Per the wave-2 plan's contingency, search and RAG extract as ONE combined series. Census in the wave-2 SDD task-8 report, copied to the recipe per-subsystem table.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Combined cluster enumerated with the recipe ownership script (search 14 + rag ~39 at 2026-09-02 snapshot)
- [x] #2 Series follows the recipe (state, RED-commit wiring, controller(s), cleanup) with both guards green throughout
- [x] #3 Recipe per-subsystem table updated with actual numbers
- [x] #4 Wave-2 final review's size-governance note considered: Library_Modules controller files (e.g. library_collections_controller.py, 1,689 lines; library_conversations_controller.py, 1,738 lines) have no size-ratchet governance today, unlike the screens they were extracted from -- wave 3 records a decision (add _BUDGETS-style rows for Library_Modules controllers, an equivalent mechanism, or an explicit defer-with-reason) rather than leaving the question unaddressed
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AC#4 only (Task 1 of the wave-3 plan): decided option (a), exact per-file _BUDGETS rows in a new sibling guard, Tests/Architecture/test_library_modules_size_ratchet.py, discovered by glob (UI/Library_Modules/*_controller.py) so a new controller is born governed instead of needing a hand-edit. 12 current controller files pinned at their exact measured line counts (699-2023 lines). Mutation-tested both directions plus the self-defending unlisted-file property (all 4 fire correctly, reverted cleanly). Full decision, reasoning, re-pin flow, and measured rows recorded in backlog/docs/library-decomposition-recipe.md new section 17. This ticks AC#4 only -- AC#1-3 (the search+RAG extraction series itself) remain To Do; task stays not-Done.

AC#1-3 (Tasks 2-4 of the wave-3 plan, the extraction series itself): complete. Task 2 (state PR) re-derived the combined cluster fresh (60 raw "search"/"rag" name matches, 50 after excluding 3 Prompts-owned + 7 Media-owned) and moved 20 fields to LibraryRagSearchState (one combined object -- field-level census found all 20 consumed inside one lock-serialized call graph, so no search/rag split). Task 3 (controller PR) re-verified the single-controller decision independently at the method level and moved 42 of the 50 candidates to LibraryRagSearchController (8 excluded: 3 @work framework-decorator hazard, 1 module-globals-coupling exclusion found by running the battery, 4 instance-attribute-monkeypatch test bypasses); one fix round corrected two false-caller-count claims and a shipped-red path-census test. Task 4 (cleanup PR) deleted the screen's generated state shim (66 literal field references across 11 screen-resident methods retargeted to self._rag_search_state.<field> -- corrected in fix round 1 from an initial undercount of "35 across 9"); a wider census also flagged canvas_sync.py's _sync_library_canvas (a cross-module write of the flat name), but its only two callers forward the CONTROLLER as "screen" (which has no _rag_search_state attribute by design), so retargeting it broke a real test caught by this task's own sweep -- reverted, canvas_sync.py needed no change (see recipe §18 for the full trace). Pruned 12 of the 42 screen delegators with zero references anywhere outside their own body, removed 14 dead imports total (5 in the original PR, 9 more of the same kind found in the same import block during fix round 1), and fixed the one moved-body docstring Task 3's review had ruled out of scope for the controller PR itself.

Fix round 1 (post-review, commit a150fc766) also corrected this task's own record: the 9 additional dead imports above, the 35/9 -> 66/11 retarget-count correction, and a canvas_sync.py guard comment. Final pins (after fix round 1): library_screen.py 43977/1316 (task 2 start) -> 42940/1304 (task 4 final; 42949/1304 pre-fix-round); library_rag_search_controller.py born-governed at 1857 -> 1895. Full per-task detail in backlog/docs/library-decomposition-recipe.md §18 and .superpowers/sdd/2026-09-03-library-decomposition-wave3-search-rag/task-{2,3,4}-report.md. All four ACs now met; task moves to Done.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

Filed as `TASK-27021` on the wave-2 branch (2026-09-03 05:55, last
edited 07:30). Merging `origin/dev` (2026-09-03) surfaced a collision
with dev's own `TASK-27021` ("Console references wire expansion into
the send-path composer completion"), an unrelated, already-filed task
-- per the 2026-08-21 owner rule (TASK-19601) and the precedent
recorded in `backlog/docs/lessons-backlog-hygiene.md`, the OLDER
arrival keeps the id and this, the younger claimant, renumbers.

Renumbered `TASK-27021` -> `TASK-31203`, derived from the same fresh
sweep as `TASK-27020` -> `TASK-31202` above (true max at merge time:
31201, then 31202 for that task, so 31203 here). Every reference to
`task-27021`/`TASK-27021` in the wave-2 SDD ledger
(`.superpowers/sdd/2026-09-02-library-decomposition-wave2-cold-trio/progress.md`,
`task-10-report.md`, `final-fix-wave-report.md`) and in
`backlog/docs/library-decomposition-recipe.md` was updated to
`task-31203`/`TASK-31203` in the same merge commit. The archival
`review-*.diff` snapshots in that SDD directory are frozen `git diff`
captures of specific historical commit ranges and were deliberately
left unedited, for the same reason recorded on `TASK-31202`'s
provenance note.
