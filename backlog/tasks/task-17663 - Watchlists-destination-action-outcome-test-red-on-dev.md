---
id: TASK-17663
title: 'Watchlists: destination action-outcome test red on dev (collections param)'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-18'
labels:
  - watchlists
  - test-health
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_destination_shells.py::test_destination_action_buttons_explain_their_outcome[watchlists_collections]` fails on clean origin/dev — verified 2026-08-18 in a detached baseline worktree at `dd1a82146` (solo run, same failure on a task-17656/17660 branch that touches neither watchlists nor this test's fixtures). Found during task-17660's collateral run. Needs the usual decide-by-reproducing pass: either a watchlists collections action button stopped explaining its outcome (real copy/tooltip regression) or the pin is stale against an intended change from a recent watchlists/destination merge.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 The failing parameterization is green on dev, with the regression fixed or the pin updated to the intended contract (decided by reproducing the surface live first)
- [x] #2 The task records which merge introduced the red
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce with detail; identify the tooltip-less buttons and the introducing commit.
2. Decide regression vs stale pin; fix at the right layer; watchlists collateral sweep.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
This one WAS a real defect (unlike its siblings 17656/17660): the destination contract — every action button carries an outcome tooltip, pinned by `test_destination_action_buttons_explain_their_outcome` — was broken by `1a57986ee` ("feat(watchlists): add explicit read pager controls"), which shipped the `items-page-previous`/`items-page-next` pagination buttons tooltip-less. Two-line production fix in `UI/Watchlists_Modules/article_list.py`: outcome tooltips ("Load the previous/next page of items.") matching the module's existing copy voice, with a comment naming the contract and the introducing commit. The pin itself was the RED (born red on dev); green after the fix; the parameterized audit's one skip is the pre-existing, deliberately tracked Personas exclusion.

Files: `tldw_chatbook/UI/Watchlists_Modules/article_list.py`.
<!-- SECTION:NOTES:END -->
