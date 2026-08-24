---
id: TASK-21969
title: Test workflow cancels every pull-request run when the base moves
status: In Progress
assignee: []
labels: [ci, infrastructure]
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The test workflow supersedes an in-flight run for any reference other than the default
branch. For a pull request the reference is the merge reference, which the platform recreates
every time the base branch moves — so on a repository taking this many merges a day, every
open pull request's run is cancelled repeatedly without anyone touching the branch.

The smaller required gate had the same rule and it was changed for exactly this reason. The
test workflow kept the original.

This cost nothing while the core job could not finish inside its budget anyway: cancelling a
run that was going to be killed by its own timeout loses nothing. Splitting that job changed
the arithmetic. The slices now finish comfortably inside the budget, so cancellation is the
only thing stopping them reporting, and each cancelled run discards most of an hour of runner
time on the pool that is already the scarcest resource here.

The trade is not free and is being taken deliberately. Not superseding means a rapidly
updated pull request queues its runs instead of replacing them, so the pool sees them one
after another rather than not at all. That is more total time than cancelling — but
cancelling spends the same compute and produces no verdict, which is worse on both counts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A pull-request run is not superseded when the base branch moves
- [ ] #2 Pushes to the development branch still supersede, and the default branch keeps its exemption, so neither of the behaviours the original rule existed for is lost
- [ ] #3 The two workflows cannot drift apart on this again without something failing
- [ ] #4 Each half of that guard is shown to fail when the rule it pins is reverted
<!-- AC:END -->
