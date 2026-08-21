---
id: TASK-19300
title: Make research runs durable, resumable jobs
status: In Progress
assignee: []
created_date: '2026-08-18 05:10'
updated_date: '2026-08-18 05:40'
labels:
  - research
  - scheduling
  - llm-calls
dependencies:
  - task-17386
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A research run that takes longer than a provider's timeout cannot finish, and a run interrupted by an app exit is stranded: nothing resumes it, and the resume path that does exist restarts the phase machine from the beginning, discarding every search already paid for. Large pools are now the normal case rather than the exception, so both of these moved from rare to routine.

Treating the run as a durable job rather than a synchronous call resolves the timeout problem without finely tuned clocks: a run whose result is an artifact and a message in the conversation that requested it can afford to take twenty minutes. What it cannot afford is to lose its work, to be executed twice, or to finish silently. Those become the substance of this task, and the timeout work reduces to generous bounds.

This supersedes the earlier framing of the same criterion, which set out to derive a size-aware synthesis budget. The measurements from that work still stand and justify the bounds here; the derivation machinery it proposed does not survive the change of framing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Exactly one executor can run a given research run at a time, and a run whose executor died can be taken over rather than stranded
- [x] #1a A worker that wakes after its lease was taken over cannot complete or renew the run it lost
- [x] #1b A run whose executor keeps dying is failed on a retry budget rather than retried forever
- [x] #1c A phase that reports no progress for longer than the lease is not taken over
- [x] #1d An executor that lost its lease cannot write artifacts or settle budget, not merely cannot finish the run
- [x] #2 A resumed run restores the budget it already spent instead of being granted it again
- [ ] #3 A resumed run continues from its last completed phase without repeating searches it already paid for
- [x] #4 Persisted evidence is bounded by a stated cap, and an artifact that exceeded it records that it did
- [ ] #5 A run interrupted by an application exit continues on the next launch without user intervention
- [ ] #6 A run parked for checkpoint review is not re-entered by the scheduler on every tick
- [ ] #7 A completed run announces itself EXACTLY ONCE in the conversation that requested it, however it was launched and however many times it was taken over
- [ ] #8 A completed run's report is reachable from the artifacts screen
- [ ] #9 A synthesis that takes longer than any single provider timeout completes, and a user-set runtime limit that cannot cover the work refuses before spending
- [ ] #10 The documentation states that runs resume at next launch and do not progress while the application is closed
- [ ] #11 A terminal or cancelled run leaves no scheduler task behind
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/070-research-run-lease-and-durability.md
Reason: the lease adds DB schema columns (lease_owner/lease_id/leased_until/
lease_attempts) and a cross-module execution-ownership contract (claim/renew/
release/fence) -- storage and service-contract decisions per the ADR policy.
Recorded retroactively with PR #1822's post-review fixes; the branch itself
landed without one, which the review flagged.

The design, the measurements behind it, and the alternatives rejected are in
`Docs/superpowers/specs/2026-08-18-durable-research-jobs-design.md`.

1. Lease a run to exactly one executor, following the server's job manager
   (`tldw_Server_API/app/core/Jobs/`, dev): a lease id alongside the worker id
   as a fencing token, stale-lease reclaim folded into acquisition rather than
   a separate reaper, a retry budget deciding requeue versus terminal failure,
   and the existing phase-progress emission doubling as the heartbeat.
   Everything else depends on this, so it lands first.
2. Restore the budget ledger from its artifact on resume.
3. Persist each round's evidence pool under a stated cap, and resume from it.
4. Drive execution through a scheduler task whose handler claims the run and
   skips runs awaiting review.
5. Surface completion: a handoff target for window-launched runs, and the
   metadata the artifacts screen filters on, with its user-guide page updated.
6. Set the generous bounds and document what "resumes at next launch" means.
<!-- SECTION:PLAN:END -->

## Implementation Notes (durability core landed; the rest is named, not done)

<!-- SECTION:NOTES:BEGIN -->
The durability core is implemented: a lease with a fencing token and a reclaim
budget, a keep-alive that survives blocking pipeline calls, a fence before every
persisting write, budget restored across all four axes on resume, and each
round's evidence persisted under a stated cap.

**Met:** #1, #1a, #1b, #1c, #1d, #2, #4.
**Not met and deliberately so:** #3 (reading the evidence pool back to skip a
completed round -- the artifact that makes it possible is written, the read-back
is not), #5-#11 (scheduler auto-resume and completion surfacing, which the plan
scoped to follow-on work).

What the review gates caught, recorded because the pattern generalises:

- A timestamp format divergence made a LIVE lease compare as expired -- the
  double-claim the lease exists to prevent.
- A retry budget that counted healthy claim/release cycles stranded runs that
  had never crashed.
- The keep-alive was an asyncio task behind a blocking call, so it never ran
  during collection; the design was right and the execution model defeated it.
- Six persisting writes were unfenced, including the one that marks a run
  completed.
- An exhausted budget left a run permanently unclaimable rather than failed --
  a regression against pre-branch behaviour -- and the fix for THAT then failed
  runs whose executors were healthy.
- Three times a test passed for the wrong reason, twice by asserting the absence
  of a bad outcome rather than the presence of the good one.

**Named follow-ups, not fixed here:** fence coverage is 2 of 11 when each fence
is removed individually; the `except ResearchLimitExceeded` half of the
lease-lost guard is untested; `evidence_pool.json` is written before the
doc-budget trim, so it holds entries the run excluded (harmless until AC #3
reads it back); `_lease_id`/`_run_id` are engine-instance state, which is safe
only because every caller builds a fresh engine per run -- a scheduler reusing
one engine would silently disable the fence; and a run killed by SIGKILL is
declined for up to `lease_seconds` before takeover, with a message that blames
another executor.

**PR #1822 external review (post-branch, fixed in the review round):** four
defects found by an independent review, each fixed test-first:

- A `renew_lease` that RAISES (e.g. transient SQLITE_BUSY) surfaced at
  `await keepalive` inside `execute_run`'s `finally`, escaping the engine AND
  skipping `release_lease` (stranding the lease; the run left running with no
  fail_run). Renewal errors are now contained in the keep-alive (treated as
  lost) and the finally's await suppresses both CancelledError and any stored
  exception, so release is unconditional.
- `release_lease` matched `lease_id` only, resetting the crash budget for an
  already-EXPIRED lease: a stalling-but-alive executor could loop
  claim -> expire -> release forever without spending the budget (defeating
  AC #1b). Only a LIVE release is a clean hand-off; an expired release leaves
  the record so the next claim counts the abandonment.
- The DECLINED executor wrote run state (`update_run_progress`) on a run it
  did not own, stomping the live executor's progress message and version
  mid-flight. Decline is now an append-only `lease_declined` event via the
  new `record_run_event` (observers append events; only the lease holder
  writes run state).
- A non-JSON-native value in an evidence entry raised TypeError inside
  `_bounded_evidence`'s measurement and failed the whole run; it is now
  dropped and counted like an oversized entry (the artifact degrades, the
  run does not fail).
- ADR-070 records the lease/durability contract (linked above).

**Qodo PR-1822 finding 7 (unversioned lease migration): fixed, reversing the
earlier decline.** The branch had landed the lease columns as bare startup
ALTERs and the first adjudication declined rehoming them ("no migration
infrastructure in this service"). On merge review the repo DOES carry a
versioned-migration convention (``TTS/migrations/`` Python steps stamped via
``PRAGMA user_version``; numbered SQL under ``DB/migrations/``), so the
service now has ``Research_Interop/migrations/``: v0->v1 adds the lease
columns and stamps ``user_version = 1``, fresh and pre-existing databases
pass through the same path, interim unversioned databases (columns present,
version 0) upgrade without re-ALTERing, and a database stamped by a newer
build is refused rather than silently downgraded. Four tests cover the
upgrade, the fresh stamp, idempotent reopen, and the future-version refusal.
ADR-070's "no migration framework" consequence entry is superseded by this.
<!-- SECTION:NOTES:END -->

## Renumbering note (2026-08-21)

Renumbered from **TASK-18060** -- that ID was independently claimed by two
sessions (the repo's 8th collision) and the other claimant
("Inspector-rail multi-file review and review comments") is already Done,
so per the collision playbook the Done task keeps the ID and this
In Progress task moves. Existing `task-18060` citations in
`tldw_chatbook/Research_Interop/*`, `Tests/Research/*`, and the
2026-08-18 durable-research-jobs plan refer to THIS task; they are left
in place to avoid conflicting with the active research branch and can be
swept to 19300 when that branch lands. `task-18060` citations in
Console/UI/AgentRuns_DB code and the review-rail docs refer to the
inspector task.
