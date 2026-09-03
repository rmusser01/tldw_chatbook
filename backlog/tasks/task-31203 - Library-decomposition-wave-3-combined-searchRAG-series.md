---
id: TASK-31203
title: 'Library decomposition wave 3: combined search+RAG series'
status: To Do
assignee: []
created_date: '2026-09-03 05:55'
updated_date: '2026-09-03 20:49'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave-2 Task 8's entanglement gate fired: 8/14 search methods (57.1%) call or are called by RAG methods; search submit is structurally the RAG entry point (_start_library_rag_query). Per the wave-2 plan's contingency, search and RAG extract as ONE combined series. Census in the wave-2 SDD task-8 report, copied to the recipe per-subsystem table.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Combined cluster enumerated with the recipe ownership script (search 14 + rag ~39 at 2026-09-02 snapshot)
- [ ] #2 Series follows the recipe (state, RED-commit wiring, controller(s), cleanup) with both guards green throughout
- [ ] #3 Recipe per-subsystem table updated with actual numbers
- [x] #4 Wave-2 final review's size-governance note considered: Library_Modules controller files (e.g. library_collections_controller.py, 1,689 lines; library_conversations_controller.py, 1,738 lines) have no size-ratchet governance today, unlike the screens they were extracted from -- wave 3 records a decision (add _BUDGETS-style rows for Library_Modules controllers, an equivalent mechanism, or an explicit defer-with-reason) rather than leaving the question unaddressed
<!-- AC:END -->

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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AC#4 only (Task 1 of the wave-3 plan): decided option (a), exact per-file _BUDGETS rows in a new sibling guard, Tests/Architecture/test_library_modules_size_ratchet.py, discovered by glob (UI/Library_Modules/*_controller.py) so a new controller is born governed instead of needing a hand-edit. 12 current controller files pinned at their exact measured line counts (699-2023 lines). Mutation-tested both directions plus the self-defending unlisted-file property (all 4 fire correctly, reverted cleanly). Full decision, reasoning, re-pin flow, and measured rows recorded in backlog/docs/library-decomposition-recipe.md new section 17. This ticks AC#4 only -- AC#1-3 (the search+RAG extraction series itself) remain To Do; task stays not-Done.
<!-- SECTION:NOTES:END -->
