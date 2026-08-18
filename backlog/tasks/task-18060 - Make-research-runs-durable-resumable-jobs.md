---
id: TASK-18060
title: Make research runs durable, resumable jobs
status: To Do
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
- [ ] #1 Exactly one executor can run a given research run at a time, and a run whose executor died can be taken over rather than stranded
- [ ] #1a A worker that wakes after its lease was taken over cannot complete or renew the run it lost
- [ ] #1b A run whose executor keeps dying is failed on a retry budget rather than retried forever
- [ ] #1c A phase that reports no progress for longer than the lease is not taken over
- [ ] #1d An executor that lost its lease cannot write artifacts or settle budget, not merely cannot finish the run
- [ ] #2 A resumed run restores the budget it already spent instead of being granted it again
- [ ] #3 A resumed run continues from its last completed phase without repeating searches it already paid for
- [ ] #4 Persisted evidence is bounded by a stated cap, and an artifact that exceeded it records that it did
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
