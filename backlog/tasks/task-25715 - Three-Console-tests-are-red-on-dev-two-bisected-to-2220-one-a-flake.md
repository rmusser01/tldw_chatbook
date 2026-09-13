---
id: TASK-25715
title: 'Three Console tests are red on dev: two bisected to #2220, one a flake'
status: Done
assignee: []
created_date: '2026-08-31 14:27'
updated_date: '2026-09-01 19:05'
labels:
  - console
  - testing
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two Console rail tests pass at 4da99a884 and fail on origin/dev at 46c2b0e5f0fb. Both were found while baselining the Context rail UX work (PRs #2233, #2242, #2260) against dev, and neither is caused by it -- each is bisected below to the commit that introduced it. Filed so they are not silently re-attributed to whatever change happens to run next to them, and so the two owning changes get the decision they each imply.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 test_context_section_headers_match_inspector_title_band passes, or the contract it pins is deliberately retired with the Inspector/Context divergence recorded
- [x] #2 test_active_reveal_queue_retains_only_identity_across_target_and_rail_removal is made deterministic -- it fails intermittently (~1 in 12) at every commit measured, so it is a flake, not a regression
- [x] #3 test_console_workbench_standard_width_inspector_snapshot passes, or its "Blocked impact" assertion is updated to the Inspector's current copy
- [x] #4 No test in this set is left red on dev without an owner
<!-- AC:END -->



## Notes

Filed in the same spirit as TASK-15512. `origin/dev` had by this point absorbed
the Context rail PRs, so "red on dev" alone no longer told me the failures were
not mine -- both had to be re-run at the pre-branch commit and then bisected
before either could honestly be called someone else's.

## Renumbering provenance

This task previously held id TASK-25713, colliding with the older
"Census-warm-boot-flakes-on-sys.modules-mutation-during-iteration" task that
arrived on dev first (created 14:12; this one 14:27 the same day).
Per the owner rule decided 2026-08-21 in TASK-19601 (**older id keeps it;
the younger task renumbers with a provenance note, regardless of Done
status**), it renumbered to TASK-25715. Citations to TASK-25713 in
this branch's own commit messages and in PR #2260's body refer to THIS
task; the other TASK-25713 holder is the older arrival and keeps the id.
