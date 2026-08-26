---
id: TASK-22301
title: Citation tests observe a persistence seam the durable path bypasses
status: In Progress
assignee:
  - '@claude'
created_date: ''
updated_date: '2026-08-26 23:21'
labels:
  - tests
  - console
  - citations
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
29 tests in `Tests/Chat/test_console_local_citation_boundary.py` assert on
persistence CALL COUNTS — `assert len(persistence.create_calls) == 1` and
similar. They cannot pass as written, for two compounding reasons.

**1. The session must be ephemeral for the send to happen at all.** The doubles
(`_ReadyCitationPersistence`, `_RecordingCitationStore`) predate
`commit_durable_turn` and carry `db = None`, so a non-ephemeral MANUAL send is
refused twice: by the durable-turn gate (`56db75386`) and by the DB-open gate
(`#2088` / TASK-22030). Both are guarded by `not session.ephemeral`. Making the
sessions ephemeral is what took this file from 69 failures to 40 — but an
ephemeral session deliberately does not persist, so call-count assertions see 0.

**2. The seam itself may be obsolete.** `a26cdafd8`'s durable dispatch
checkpoint writes its USER and assistant rows with raw SQL in
`console_dispatch_repository.insert_with_messages`, bypassing `create_message`
entirely. So even a durable-capable double might never see the calls these tests
count.

The tests' subject — that a sealed citation write reaches persistence exactly
once, and that a failed repair writes nothing — is still worth testing. What is
stale is *where* they watch for it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The tests observe a seam the current durable path actually uses
- [ ] #2 Each still fails when its real invariant is broken (mutation-proven)
- [ ] #3 No test asserts a call count that an ephemeral session cannot produce
- [ ] #4 `test_console_local_citation_boundary` reports 0 failures
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
PARTIAL PROGRESS 2026-08-26 (TASK-22587 merged, so this is unblocked).

Measured, in order -- each step's effect verified before the next:

1. Swapping the db=None fake for a real ChatPersistenceService changed NOTHING
   (40 failures before and after). This REPRODUCES the earlier inert-spy
   finding: real persistence is inert while sessions are ephemeral.
2. Making sessions durable alone made it WORSE (40 -> 65): _persisted_store()
   defaulted to persistence=None, harmless when ephemeral, a refused send when
   durable ('Not sent: your conversation database could not be opened'). Fixed
   by defaulting to real persistence and using production's own rule --
   ephemeral IFF nothing can persist.
3. With durable sessions + real persistence: user AND assistant rows ARE
   committed, and create_calls is 0. That CONFIRMS the task's core claim -- the
   durable path writes rows via insert_with_messages and bypasses
   create_message entirely.
4. Citation traces still wrote 0 rows. Cause: the old fake hard-coded
   canonical_citation_writes_ready = True; the real service COMPUTES it and
   reports False without a citation_repository, silently skipping every
   citation write. Wiring the real CitationTraceRepository stack (same as
   test_console_terminal_citation_persistence's rig) made create_calls carry a
   real SealedCitationWrite.
5. Traces STILL wrote 0 rows. Instrumented prepare_write/write_prepared:
   prepare_write raises CitationPersistenceUnavailable('run_authority_mismatch')
   and write_prepared is never reached.

THAT IS THE REMAINING BLOCKER. _citation_builder() builds a test-local
_WeakCitationBuilder; a builder must instead come from the repository via
repository.create_local_trace_builder(...) for its run authority to match. See
_real_captured_builder() in test_console_terminal_citation_persistence.py for
the worked example.

Also learned: the DB must be FILE-backed, not ':memory:'. CharactersRAGDB opens
a thread-local connection and TASK-22205 offloads durable DB calls to a worker
thread, which with ':memory:' gets its own empty database.

State: 40 -> 33 failures, 55 -> 62 passing. Remaining: 25 x 'assert 0 == 1'
(blocked on the builder authority above), 2 x 'assert 2 == 1' (the assistant row
is now written twice -- once with the citation write and once without, which is
TASK-22302's terminal-persistence pattern and may be correct), 2 x 'Durable
continuation owner changed', 2 hangs, and a few singles.

NOT YET DONE: the assertion conversion itself. Row helpers (message_rows,
citation_trace_rows, cited_message_rows) are in place on the persistence double
and ready for the 23 call-count sites once traces actually persist.
<!-- SECTION:NOTES:END -->
