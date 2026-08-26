---
id: task-22301
title: Citation tests observe a persistence seam the durable path bypasses
status: To Do
labels:
  - tests
  - console
  - citations
priority: medium
---

## Description

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

## Acceptance Criteria

- [ ] The tests observe a seam the current durable path actually uses
- [ ] Each still fails when its real invariant is broken (mutation-proven)
- [ ] No test asserts a call count that an ephemeral session cannot produce
- [ ] `test_console_local_citation_boundary` reports 0 failures

## Implementation Notes

Decide first WHICH seam is authoritative for a citation write now: the raw-SQL
checkpoint, `create_message`, or the citation repository. Then move the
assertions there — asserting on durable ROWS rather than on call counts would
survive the next refactor of the write path, which counting calls did not.

Do not "fix" these by adding `commit_durable_turn` to the doubles. The real
method owns a transaction, creates the conversation, writes the Library policy
and returns a checkpoint; hand-copying that into a fake is the
fake-validates-the-mistake trap, and it is what these doubles already did with
`create_message`.
