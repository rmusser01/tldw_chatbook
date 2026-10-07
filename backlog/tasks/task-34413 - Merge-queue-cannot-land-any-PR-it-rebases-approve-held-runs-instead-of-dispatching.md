---
id: TASK-34413
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
- 'retry' (_retry): re-run the failed run in its own check suite (rerun-failed-jobs; a broken run's stand-in is re-run in full). If the run is the tick's own (still in progress), wake merge-queue.yml with wait_run; a live run is left alone; a refused re-run (403/409/422) evicts (evict-rerun); 5xx re-raises. A re-run held for approval is approved.
- merge-queue.yml gains a workflow_dispatch trigger with a wait_run input, and run()/main() wait (bounded at 40 x 6 s) for it.

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
<!-- SECTION:NOTES:END -->
