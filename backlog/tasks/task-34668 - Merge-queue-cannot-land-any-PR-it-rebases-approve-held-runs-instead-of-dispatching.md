---
id: TASK-34668
title: >-
  Merge queue cannot land any PR it rebases: approve held runs instead of
  dispatching
status: In Progress
assignee:
  - '@claude'
created_date: '2026-10-07 02:30'
updated_date: '2026-10-07 03:04'
labels:
  - ci
  - merge-queue
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The in-repo merge queue dispatches the required check after rebasing a PR, but a workflow_dispatch run's checks are absent from the PR's status rollup, so every queue-rebased PR stays BLOCKED with all checks green until the stuck eviction (#2874, #3026 on 2026-10-06). The queue was turned off. GitHub holds the PR's own pull_request runs for approval after a token rebase; a live probe (#3033) showed the queue's GITHUB_TOKEN can approve them and that the approved run's required check counts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After rebasing the front PR, the queue approves the held pull_request runs its own token caused, and never dispatches the required check
- [x] #2 Only held pull_request runs triggered by the queue's actor, on the front PR, are approved
- [x] #3 A first failure re-runs the failed run in its own check suite; when the deciding tick is the failed run, a woken queue run waits for it to complete and re-runs it; a run already live again is left alone
- [x] #4 A GitHub outage fails the queue run without disarming any PR; only a re-run refusal or having no CI run to start evicts
- [x] #5 CLAUDE.md, AGENTS.md, the spec and ADR-218 describe the approve design with the 2026-10-06 evidence
- [ ] #6 Before the queue is re-enabled, one throwaway PR is merged by the queue end to end, and the owner approves re-enabling
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
scripts/merge_queue.py no longer dispatches the required check, which never counted: a workflow_dispatch run's checks are absent from the PR's statusCheckRollup (spec V4; #2874 and #3026 stranded, #3025 the pull_request control). Instead:
- Every on-mode tick approves the front PR's held pull_request runs triggered by github-actions[bot] (approve_held_runs, best-effort), before deciding.
- After a rebase the queue waits, bounded at 10 x 3 s, for the new head's held required run and approves it (_approve_until_required), then cancels old-head runs. The re-dispatch of other workflows and the deletion of held runs (cleanup_approval_runs) are gone.
- 'dispatch' becomes 'start' (_start): approve a held run strictly, so a GitHub error re-raises and an outage never falls through to an eviction; else re-run a cancelled run in full; else evict (evict-no-run).
- 'retry' (_retry): re-run the failed run in its own check suite (rerun-failed-jobs; a broken run's stand-in is re-run in full), once per head (retry marker). If the run is the tick's own (still in progress), send a queue kick (derived-artifacts.yml dispatched on dev, input wait_run); a live run is left alone and a held one approved. Re-run errors: refused (409/422, month-old 403) evicts (evict-rerun); transient (5xx, 429, rate limit, network) re-raises; anything else re-raises with a rerun-error warning and evicts once that warning is 10 minutes old (rounds 2 and 7).
- derived-artifacts.yml gains a wait_run workflow_dispatch input, passed to queue-tick as WAIT_RUN; run()/main() wait for that run, bounded at 50 x 6 s. merge-queue.yml is unchanged (not on main, so never dispatchable).

Evidence: live probe #3033 on 2026-10-06. The queue's GITHUB_TOKEN approved a held run (APPROVE_OK), and the approved run's required check appeared in the PR rollup as SUCCESS/pull_request, merge state UNSTABLE (mergeable).

Tests: Tests/CI/test_merge_queue_actions.py rewritten from dispatch to approve, plus new tests for own-run wake, live stand-down, the woken run's bounded wait, a re-run held then approved, cancelled and broken runs re-run in full, an outage failing without a disarm, a refused re-run evicting, no-run eviction, and dry mode staying read-only. Rules and workflow tests updated. Tests/CI 465 passed. Six guards mutation-checked: actor filter, strict start approval, own-run wake, live stand-down, post-rebase approval, held re-run approval.

Docs: CLAUDE.md and AGENTS.md (the 'never click Approve and run' rule reversed), spec (revision note, F2/F4 corrected, V3/V4 added, decision table and sections 7-8), ADR-218 amendment.

AC 6, the end-to-end live test, waits for this PR to merge: merge-queue.yml runs dev's script, so the new code can only be exercised from dev. Then ask the owner before re-enabling.

Independent review round 1 (Qodo out of credits). Fixed:
1. A 403 rate limit or permission error was read as a refused re-run and evicted. Refusals are now 409/422, or a 403 that says the run is over a month old; everything else re-raises.
2. The self-wake dispatched merge-queue.yml, which is not on main, so GitHub can never dispatch it (0 dispatch runs ever). The wake is now a queue kick: derived-artifacts.yml (on main) dispatched on dev without pr, with a new wait_run input passed to queue-tick as WAIT_RUN. merge-queue.yml is back to dev's version.
3. A broken run (no check reported) keeps its run id when re-run, so it could be retried forever. A retry marker on the head now caps it at one retry (evict-failed-twice).
4. Dispatched required checks counted as green, and a live dispatched run counted as in flight. counted() drops checks from non-pull_request suites, and stand-ins consider only pull_request runs.
5. When the post-rebase approval wait ended with nothing approved, nothing woke the queue. It now sends a kick.
Nits fixed: Args sections; stale comments in merge_queue.py and derived-artifacts.yml; the spec's V1 row, sections 9-11 and its 'Approve and run' rule; the pre-decide approval pass skips BEHIND/DIRTY fronts; approval also requires head_repository == this repo.
New tests: 403 rate limit (primary, secondary) and permission errors raise; a month-old 403 refuses; a broken run retried twice evicts; a refused wake fails without a disarm; dispatched checks never count; a live dispatched run is not waited on; a bot PR's and a foreign repo's held runs are never approved; a BEHIND front is not pre-approved; the post-rebase timeout wakes a tick. Tests/CI 475 passed; seven new guards mutation-checked; no new lint versus dev.

Independent review round 2 (Qodo still out of credits). Fixed:
1. A held re-run whose approval failed was evicted as failed twice. _retry now approves a queue-held run strictly (an outage fails the run).
2. A re-run error the queue could not classify (GitHub answers a broken workflow file with a 403) re-raised forever and stalled the line. It now fails once with a rerun-error comment, then evicts (evict-rerun).
3. Two queue runs racing on one re-run: the loser's refusal evicted. A refusal, and the retry-cap check, now re-read the run and stand down if it is live or held again.
4. A cancelled (not failed) retry attempt hit the retry cap. It is re-run in full without counting.
5. A head whose commit predates its push could reach start before its run was listed, and was evicted (evict-no-run). _start now looks again for up to 10 x 3 s first.
6. Wake-path stalls: in on mode a young-head wait now sleeps out the window once and decides again; the wait_run bound is 5 minutes (inside queue-tick's 10-minute timeout); a wait_run that cannot be read is decided on anyway.
7. Mutants that survived: _start's own copy of the held-run filter (now one shared predicate, _queue_held), _required_runs' pull_request filter, main() reading WAIT_RUN, the rate-limit classification. Each now has a test. (Round 3 found two round-2 guards still unpinned; see below.)
8. Doc drift: spec sections 2, 5, 6, 7, 8 and 11; the fork note (UNQUEUED_NOTES and spec: approving a fork's held runs would bypass GitHub's outside-contributor gate); stale 'merge queue re-runs this' comments in perf-guard.yml, task-19642 and task-32011 workflows; test_pr_workflows_dispatch_safe docstring; required_run_stand_ins docstring; this task's bullets above.

Independent review round 3. Fixed:
1. A re-run GitHub kept holding after the 30 s approval polls woke nothing (a held run never runs its own queue-tick), so the line stalled. _rerun now wakes a tick. A rebase whose branch never moves wakes one tick per head (rebase-unmoved marker) for the same reason.
2. The young-head sleep was unbounded (a commit dated hours ahead slept past the job timeout). It is capped at 3 minutes.
3. The young re-decide acted on a front that was re-armed (moved to the back) during the sleep. It now stops when the arming changed, not only on a disarm.
4. A racing re-run refused with an unclassified code (e.g. a 403) used up the head's rerun-error allowance with a spurious comment. The live-or-held re-read now comes before the error classes.
5. wait_for_run ended at the first read error. A failed read is retried within the bound.
6. Unpinned guards: the 'held again' half of the re-read, and the disarm break. Both have tests now; 24 mutants (round 2 and 3 guards) are all killed.
7. The month-old refusal matched any 403 mentioning 'month'; it now needs 'month ago'.
8. Doc drift: ADR-218 eviction causes (a refused re-run of a cancelled run evicts as evict-no-run), spec sections 6, 7, 8 and 11.
Test-harness trap: a parametrize id 'live' made Tests/conftest.py skip the test as --run-live (it matches item.keywords); ids renamed to 'running'.
Independent review round 4: no critical or major findings. Fixed all seven minor ones:
1. The re-run wake fired before the retry marker, so a failing comment POST could chain kicks. _rerun now returns 'pending' and the caller wakes after recording the attempt.
2. A rebase accepted but never landing re-rebased on every event forever. The second unmoved rebase on a head evicts (evict-rebase).
3. One run could exceed the 10-minute job timeout across several fronts. A 4-minute run budget: no further front and no young-head wait that would end past it; a fresh run is woken instead. _evict now comments before it disarms, so a killed job never disarms silently.
4. A head dated in the future stalled the line until the author's clock time. Future dates are not young; start looks for the run (this replaces round 3's sleep cap, now dead code).
5-6. wait_for_run: the sleep after a failed read is pinned, and a later good read clears the remembered error from the log line.
7. Doc drift: ADR-218 eviction causes per path; spec section 8 on what wakes after an approval failure.
Mutation check: 32 mutants of the round-2 to round-4 guards, all killed.
Independent review round 5: one major finding (predating this PR) and five minor ones, all fixed:
1. Major, already on dev: two 5xx answers to the rebase mutation, even days apart, evicted the PR, against the outage rule and this PR's own ADR line. Transient rebase errors now re-raise and never count.
2. A second rebase strike (refused or unmoved) could land a minute after the first, so one GitHub slowdown disarmed. A strike evicts only if the warning comment is at least 10 minutes old (STRIKE_GAP, read from the comment's createdAt). The unmoved eviction gets its own slug (evict-rebase-unmoved).
3. The first front is decided whatever the budget, after a wait_run of up to 5 minutes, which could hit the job timeout. The wait_run bound is now 3 minutes (30 x 6 s).
4. A head dated 30 s ahead got 27 s of start polls, then evict-no-run. Heads dated within 3 minutes either side are young again (round 3's capped sleep is back), and a head still young after the window is started rather than waited on.
5. _evict's disarm was best-effort, so a failure left an armed PR under a 'removed, auto-merge is off' comment and, with the budget wake, could repeat forever. A failed disarm re-reads the PR and raises if it is still armed. A comment GitHub refuses for good (4xx, e.g. a locked conversation) no longer blocks the disarm.
6. Spec section 8 now says which approval failures wake a tick.
Mutation check: 42 mutants of the round-2 to round-5 guards, all killed.
Independent review round 6: one major finding and five minor ones, all fixed:
1. Major: _transient matched only REST-shaped errors. gh 2.90.0, captured against a fake server, prints a GraphQL timeout as 'gh: Something went wrong while executing your query', a non-JSON 502 as 'gh: HTTP 502', and a dropped connection with no 'gh: ' prefix. So an outage on the rebase mutation still counted as strikes and could evict. _transient now matches those shapes and abuse detection; only answers GitHub gave about the PR count.
2. The still-young-to-start step fired on a head that changed during the wait (a push or a racing rebase), evicting a genuinely new head. It now applies only to the head the queue waited on.
3. A strike inside its 10-minute gap holds the line on a quiet repo until the next event. Kept deliberately and documented as a known gap in spec section 8 (evicting sooner lets one slowdown disarm the PR).
4. Only _evict tolerated a locked conversation; a locked PR whose rebase failed raised forever. comment_once now raises CommentRefused for a locked PR, and apply() takes the PR out of the line without a comment (the 'rebased' comment is just skipped). Any other comment refusal fails the run. _evict's own catch was dead code (every _evict call runs under apply) and was deleted, found by a surviving mutant.
5. A failed rebase whose head moved anyway (the response lost) woke nothing, so the new head's held runs had no approver. It wakes a tick.
6. Tests: the strike boundary is pinned (exactly 10 minutes evicts), the second-strike test uses a real createdAt, and a failed re-read after a failed disarm fails the run.
Mutation check: 43 mutants of the round-2 to round-6 guards, all killed (after the dead catch was removed).
Independent review round 7: no critical or major findings; four minor, two plausible-minor and nits, all fixed:
1. _rerun_error still read any error without '(HTTP ' as transient, so gh's bodiless 'gh: HTTP 404' re-raised forever. The clause is gone (_transient covers network errors); refusals match 'HTTP 409/422' with or without parentheses.
2. Round 6's head-moved wake sat on the wrong branch: a lost rebase response arrives as a transient error. The transient branch now re-reads and wakes only if the head moved; the non-transient head-moved branch (a racing run losing) no longer kicks, saving a run per merge.
3. 'locked' matched GitHub's 'temporarily blocked from content creation' rate limit. CommentRefused now needs 'is locked', with that rate-limit message pinned as a test.
4. A gh warning line printed before the 'gh: ' message would have read as a network error; any line starting 'gh: ' now counts as a server answer.
5. The rerun-error cap had no time gap, so one odd answer seen by two runs a minute apart evicted. It now uses STRIKE_GAP like rebase strikes.
6. An Actions incident delaying run creation past the young window evicted on first sight (evict-no-run). No-run is now a strike too: warn first, evict 10 minutes later. Documented: an incident longer than that, seen twice, still evicts.
7. Nits: ADR-218 now names the locked-PR rule and the strike kinds; bare 'HTTP 429' tested; a stale comment removed; ISC004 string concatenations parenthesised.
Mutation check: 41 mutants of the round-2 to round-7 guards, all killed.
Independent review round 8: no critical or major findings; all fixed:
1. On the start path a final re-run refusal (409/422, month-old 403) now struck instead of evicting, and an unclassified error served two gaps (20 minutes) and was evicted as no-run. A refused re-run of a cancelled run evicts at once again (evict-no-run, naming the refusal); the unclassified error's gap is served inside _rerun.
2. The transient-rebase re-read happened before an accepted rebase's ref could land; it now waits REBASE_POLL_S first, like the non-transient branch.
3. CommentRefused again requires a non-transient error, so an outage can never take a PR out of the line by construction.
4. A strict approval re-raised every error, so a lasting refusal failed every run without evicting. A non-transient refusal is re-read (a racing run may have approved it) and is otherwise an approve-error strike (evict-approve after the gap).
5. Tests: bare and parenthesised 409/422, the no-run warning not waking, the re-read sleep; stale comments in merge_queue.py and the spec fixed. The parenthesised-403 mutant is equivalent (a body-less 403 cannot carry the 'month ago' text) and has no test.
Mutation check: 49 mutants of the round-2 to round-8 guards, all killed.
<!-- SECTION:NOTES:END -->
