---
id: TASK-22030
title: >-
  A failed ChaChaNotes open makes Console Send silently do nothing
status: To Do
assignee: []
created_date: '2026-08-24'
labels:
  - bug
  - console
  - error-handling
priority: high
---

## Description

If a user's ChaChaNotes database fails to open, pressing Send in the Console now does **nothing
at all** — no message, no system row in the transcript, no toast, no user-visible log. The draft
stays in the composer and the app looks like it ignored the keypress.

The refusal itself is arguably correct — a durable turn that cannot be committed should not be
started. What is wrong is that it is **invisible**. The user has no way to learn that their
database is broken, and the most likely reading of the symptom is "the app is broken".

## Acceptance Criteria

- [ ] With an unopenable ChaChaNotes database, pressing Send produces a visible, legible explanation of why the message was not sent and what to do about it
- [ ] The explanation names the real cause (the database could not be opened) rather than a generic failure
- [ ] The draft is preserved so the user does not lose what they typed
- [ ] An ephemeral/temporary conversation still sends in this state, since it needs no durable commit
- [ ] A test drives the degraded path end to end and asserts the user-visible surface, not just the return value — the current gate returns a correct-looking result object while showing the user nothing
- [ ] The test fails if the refusal is made silent again

## Evidence (verified on dev, 2026-08-24)

Introduced by `56db75386` ("fix(console): harden durable turn ownership", 2026-08-23), a
review-fix on TASK-19900.3. The hunk replaced this:

```python
if (
    self.store.persistence is not None
    and getattr(self.store.persistence, "db", None) is not None
    and not session.ephemeral
    and origin in {ConsoleSubmissionOrigin.MANUAL, ConsoleSubmissionOrigin.QUEUED}
    and not callable(durable_commit)
):
    return self._block(session.id, ...)
```

with this:

```python
if durable_turn and not callable(durable_commit):
    return ConsoleSubmitResult(False, False, session_id=session.id, ...)
```

Two changes compound. The `persistence is not None and persistence.db is not None` precondition
is gone, so a session with no working persistence now qualifies as "a durable turn that cannot
commit" instead of being exempt. And `_block(...)` — which writes a system row and raises a toast
— became a bare `ConsoleSubmitResult`, so the refusal has no user-visible surface at all.

**Users with a working database are unaffected**: `durable_commit` is callable, the gate does not
fire, and send works. This was confirmed by a live headless Pilot run of the real app with an
isolated profile: one provider call, draft cleared, transcript complete, run state COMPLETED.

Found while repairing the stale Console send test harness (TASK-21590). The harness had been
masking this: every factory-built test app has `persistence=None`, so the doubles hid the
behaviour change behind 26 unrelated-looking test failures.
