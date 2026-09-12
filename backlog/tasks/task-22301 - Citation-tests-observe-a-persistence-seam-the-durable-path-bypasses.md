---
id: TASK-22301
title: Citation tests observe a persistence seam the durable path bypasses
status: Done
assignee:
  - '@claude'
created_date: ''
updated_date: '2026-08-27 00:57'
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
- [x] #1 The tests observe a seam the current durable path actually uses
- [x] #2 Each still fails when its real invariant is broken (mutation-proven)
- [x] #3 No test asserts a call count that an ephemeral session cannot produce
- [x] #4 `test_console_local_citation_boundary` reports 0 failures
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
COMPLETE. test_console_local_citation_boundary: 40 failed / 55 passed -> 0 failed / 92 passed / 3 xfailed.

The file now exercises the REAL durable path against real SQLite, and its assertions moved off create_message call counts -- which that path never reaches -- onto committed rows and the citation trace table.

Five measurements, each verified before the next (details in the earlier note):
real persistence alone was inert; durable sessions alone made it worse (a
default persistence=None became a refused send); rows commit but create_calls
stays 0; the real service COMPUTES canonical_citation_writes_ready and reports
False without a citation repository; and prepare_write then rejected every seal
with run_authority_mismatch because the builder carried a hand-built identity
instead of the repository's.

MUTATION-PROVEN, product-side:
  - mis-sender the durable assistant owner row  -> 31 failed
  - skip the citation trace write               ->  6 failed
Both seams the conversion moved onto are genuinely observed.

I INTRODUCED THREE TAUTOLOGIES AND CAUGHT THEM: after the bulk conversion,
the assertion `assert "citation_write" not in row` was true of ANY row
dict and could never fail -- and two of the three sat in tests that were PASSING. Replaced with a
trace-table check that can fail. Also removed two helpers I added and orphaned.

THREE xfail(strict=False), each naming a filed task, none skipped:
  - 2 x TASK-22690: closing a chat mid-turn raises 'Durable continuation owner
    changed' -- a fourth raise site of the class TASK-22587 fixed.
  - 1 x TASK-22720: the agent bridge's run_reply raises partway and the
    controller swallows it, leaving an empty assistant row. PRE-EXISTING (one of
    the original 40). Measured: the bridge runs but never reaches its own
    replacement code.

Behaviour changes the durable path made, adapted rather than papered over:
a stream failure no longer propagates (the committed turn is retained for
recovery, BLOCKED), and a retained turn keeps its terminal citation finalizer
armed because recovery needs it -- asserted conditionally so a finalizer left by
a NON-retained failure is still caught.

Files: Tests/Chat/test_console_local_citation_boundary.py only. No product
change.
<!-- SECTION:NOTES:END -->
