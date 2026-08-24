---
id: TASK-21967
title: The required gate still fails to report on about half of pull requests
status: To Do
assignee: []
labels: [ci, infrastructure]
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-21250 identified why the one required status check kept failing to produce a verdict,
and stopped it cancelling its own pull-request runs. That helped, and the problem is not
solved: measured over the most recent 100 runs of that workflow, **50 are cancelled, 34
succeed, 7 fail** and 9 are still in flight. Restricted to pull-request runs it is **39
cancelled against 29 successes**.

The consequence is the one the earlier task described, at roughly half the previous rate: a
pull request sits blocked with its only required check showing no result, and stays there
until somebody notices and re-runs it by hand. It is not specific to any author or branch —
it is whichever pull requests happen to be open while the base branch moves.

The residual cause is named in the earlier task's own comment. For a pull-request event the
run is tied to a merge reference that the platform recreates every time the base branch
moves, and on a repository absorbing this many merges a day the reference is frequently
recreated before a queued run gets a chance to start. Not cancelling the run ourselves does
not help when the reference it was created for no longer exists.

What makes this worth another pass rather than acceptance: the workflow only builds runs from
pull-request events for a branch under review. Pushes are watched for the two long-lived
branches alone, so a topic branch never gets a run tied to a stable reference — the volatile
one is the only one it has.

Two things need establishing before changing anything, because this is the check every merge
depends on and a mistake here blocks the whole repository rather than one branch. First,
whether a run tied to the branch reference is accepted for the same commit, given a
same-named result from the volatile reference may already exist against it. Second, whether
the extra runs are affordable, given queueing is already the binding constraint and the
earlier task accepted redundant runs as the cheaper trade.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The proportion of required-gate runs that reach a verdict is measured before and after any change, over a comparable number of runs
- [ ] #2 Whether a result recorded against the branch reference satisfies the requirement for a commit that also has a result from the merge reference is established by observation, not by reasoning about the platform's documented behaviour
- [ ] #3 The change does not increase the queueing that is already the binding constraint, or the increase is measured and accepted explicitly
- [ ] #4 If no safe change exists, that conclusion is recorded with its evidence so the next person does not repeat the investigation
<!-- AC:END -->

## Notes

Found while merging a series of test-health pull requests: three of them were blocked in turn
by this, each needing a manual re-run of the required check before it could merge. The
measurement above was taken at that point.

Deliberately filed rather than fixed. This is the one check the repository's merges depend
on, TASK-21250 is recent work on the same mechanism, and the safe form of the change rests on
a platform behaviour that should be observed rather than assumed.
