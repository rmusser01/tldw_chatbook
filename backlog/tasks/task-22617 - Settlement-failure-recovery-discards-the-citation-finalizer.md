---
id: TASK-22617
title: Settlement-failure recovery discards the citation finalizer
status: Done
assignee:
  - '@claude'
created_date: '2026-08-26 19:20'
updated_date: '2026-08-27 20:55'
labels:
  - console
  - citations
  - durable-turn
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-22302 fixed four sites where a durable turn's citation trace was never persisted. A fifth call site was found during that work and deliberately left out of scope, because it sits on a different path and may be intentional.

`ConsoleChatController.resume_durable_postcommit` has an `except BaseException:` arm that publishes a recovery owner via `store.publish_durable_recovery_owner(...)` with `terminal_citation_finalizer=None` hard-coded (console_chat_controller.py:6126). The enclosing `continuation` object IS in scope there -- the very next argument reads `continuation.citation_repair_session` -- so the finalizer is available and discarded rather than unavailable.

Two readings, and the task is to decide which is right:

1. It is the same data-loss defect as TASK-22302. A turn whose settlement failed still has a committed assistant row, and its citation trace is dropped with no way to recover it.
2. It is correct as written. This path reports "Delivery status is unknown", so arming a finalizer might persist a trace describing content that never reached the user -- provenance worse than absent.

Reading 2 is plausible enough that this must not be 'fixed' by pattern-matching TASK-22302. Establish which it is with a test before changing anything.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The BaseException path in resume_durable_postcommit is reproduced in a test that observes whether a citation trace survives a settlement failure
- [x] #2 A decision is recorded, with evidence, on whether dropping the finalizer there is data loss or a deliberate guard against attributing unsent content
- [x] #3 If it is data loss: N/A -- established NOT data loss (see notes); the deliberate branch (#4) applied instead
- [x] #4 If it is deliberate: the hard-coded None carries a comment naming the reason, so the next reader does not file this again
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Locate the BaseException arm's publish_durable_recovery_owner call on current dev and map the FULL lifecycle after it: what happens to a recovery owner on retry/resume -- is the finalizer ever re-armed, or is the trace unpersistable from then on?
2. Write the reproduction: durable turn with a real citation finalizer armed, settlement fails after the terminal effect, recovery owner published. Observe rag_citation_traces.
3. Decide from evidence: if the retry path rebuilds/re-arms the finalizer, dropping it here is safe (document per AC4). If nothing downstream can ever persist the trace, it is data loss (fix per AC3).
4. Either way: mutation-prove whatever test lands.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
VERDICT: not data loss -- and not quite either filed reading. Established by
instrumentation on the real citation stack, then pinned by tests.

MECHANISM (measured, not read): `durable_owner_publication` -- the effect that
ARMS the terminal citation finalizer -- also registers the session's dispatch
recovery via `publish_durable_turn_owners` -> `publish_durable_dispatch_
checkpoint`. The failure arm gates its recovery publish on "no recovery
exists", so the None publish is UNREACHABLE once a finalizer is armed. A spy
confirmed it fires exactly once across both failure orderings: in the
pre-arming one. There, nothing was ever armed, the continuation survives, and
the resume's owner publication forwards `continuation.terminal_citation_
finalizer` (TASK-22302), so the trace persists on retry.

THREE TESTS on the real stack (test_console_terminal_citation_persistence.py),
THREE MUTATIONS all red: (M1) a None publish that CLEARS armed state; (M2) the
resume publish dropping the finalizer; (M3) owner publication no longer
registering the recovery -- M3 is what makes "unreachable while armed"
falsifiable rather than folklore.

TWO OF MY OWN ASSERTIONS DIED EN ROUTE: the fail-after-arming test first
claimed to guard a clobber its scenario cannot reach (M1 stayed green); its
replacement post-hoc recovery check was mis-timed (the arm restores a recovery
AFTER its publish gate, so the check is true in every ordering). Both replaced
by a call spy on the publish, with the pre-arming test as positive control.

AC3 (data-loss branch) N/A; AC4 done -- the None site carries the full
evidence-backed reason in-code.

Verification: A/B of Tests/Chat vs merge-base -- 0 newly broken, +3 collected,
94 failures both sides; file itself 96 passed; preflight green.

Files: tldw_chatbook/Chat/console_chat_controller.py (comment only),
Tests/Chat/test_console_terminal_citation_persistence.py.
<!-- SECTION:NOTES:END -->
