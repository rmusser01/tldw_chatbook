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

**Correction, from the experiment this task called for: the residual cause is not what is
described below.** Ninety-five per cent of cancellations happen in coordinated bursts that
kill every queued run in the repository at once, across every workflow — forty-six runs
inside twenty-four seconds in the largest observed case, thirty-six inside three minutes in
the next. Those bursts are indiscriminate: the required gate, the test suite, the guards and
the one-off evidence workflows all die together, whatever triggered them and whatever
reference they are tied to.

The experiment settled the proposed fix as well, and settled it negatively. A run triggered
by a push, tied to a branch reference that nothing recreates, was cancelled in the same burst
as its pull-request sibling on the same commit. So watching pushes on every branch would not
help: the runs it creates are swept along with the rest.

What remains unknown is what performs the sweeps, and that cannot be established from the
runs themselves. The candidates are an account limit being reached, or somebody clearing a
backlog by hand — the queue regularly holds twenty to thirty runs against a pool that starts
one or two at a time, so clearing it is a reasonable thing for a person to be doing. Whoever
knows which will know immediately; nobody else can tell from the outside.

The mechanism described in the rest of this section is real and is what the earlier task
fixed. It is simply not what is causing the current rate.

The originally suspected cause, retained because it is still worth understanding: For a pull-request event the
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

## The same rule, unfixed, in the expensive workflow

Found after filing, and it is the more wasteful half.

TASK-21250 changed the small gate's rule so pull-request runs are not cancelled. The main
test workflow still carries the original form, which cancels for any reference that is not
the default branch — pull requests included. Since the platform recreates a pull request's
merge reference every time the base moves, its test runs are cancelled repeatedly without
anyone touching the branch, exactly as the earlier task described for the small gate.

The consequence is now measurable, because the core job was recently split into six slices
and one of them finishes in about half an hour. On a run observed after that change, one
slice completed and **five were cancelled after between 55 and 86 minutes each** — well
inside their time budget, and none of them produced a verdict. That is several hours of
runner time per run, spent and discarded, on the pool that is already the binding constraint.

The fix has a precedent in this repository: the same expression the small gate now uses.
What differs is cost. The small gate is a minute and a half, so redundant runs were accepted
without much thought; a full test run is far larger, and not cancelling means a rapidly
updated pull request queues its runs rather than replacing them. That trade needs deciding
rather than assuming — but the present state spends the compute and gets nothing, which is
the worst of both.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The proportion of required-gate runs that reach a verdict is measured before and after any change, over a comparable number of runs
- [ ] #2 Whether a result recorded against the branch reference satisfies the requirement for a commit that also has a result from the merge reference is established by observation, not by reasoning about the platform's documented behaviour
- [ ] #3 The change does not increase the queueing that is already the binding constraint, or the increase is measured and accepted explicitly
- [ ] #4 If no safe change exists, that conclusion is recorded with its evidence so the next person does not repeat the investigation
- [ ] #5 The main test workflow's cancellation rule is decided on the same evidence, including whether queueing rather than replacing runs is affordable on the current pool
<!-- AC:END -->

## Notes

Found while merging a series of test-health pull requests: three of them were blocked in turn
by this, each needing a manual re-run of the required check before it could merge. The
measurement above was taken at that point.

Deliberately filed rather than fixed. This is the one check the repository's merges depend
on, TASK-21250 is recent work on the same mechanism, and the safe form of the change rests on
a platform behaviour that should be observed rather than assumed.
