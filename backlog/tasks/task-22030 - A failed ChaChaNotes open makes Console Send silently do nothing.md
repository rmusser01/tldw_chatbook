---
id: TASK-22030
title: >-
  A failed ChaChaNotes open makes Console Send silently do nothing
status: Done
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

- [x] With an unopenable ChaChaNotes database, pressing Send produces a visible, legible explanation of why the message was not sent and what to do about it
- [x] The explanation names the real cause (the database could not be opened) rather than a generic failure
- [x] The draft is preserved so the user does not lose what they typed
- [x] An ephemeral/temporary conversation still sends in this state, since it needs no durable commit
- [x] A test drives the degraded path end to end and asserts the user-visible surface, not just the return value — the current gate returns a correct-looking result object while showing the user nothing
- [x] The test fails if the refusal is made silent again

## Implementation Plan

1. Confirm the mechanism on the current dev: `chachanotes_db = None` →
   `persistence=None` → `durable_commit` is not callable → bare result.
2. Restore a user-visible refusal (run state + transcript row + toast) with copy
   that names the database, keeping the fail-closed behaviour.
3. Keep the ephemeral exemption exactly as-is.
4. Drive the degraded path end to end in a mounted test and live, asserting the
   user-visible surfaces.
5. Mutation-check each surface independently.

## Implementation Notes

The refusal is kept (a turn that cannot be committed must not reach the
provider) and made loud. The bare `ConsoleSubmitResult` is replaced by
`ConsoleChatController._block_undurable_turn`, which routes through the existing
`_block` (blocked run state + SYSTEM transcript row) and adds an error toast via
a new defensive `_notify_app` helper (mirrors `_notify_detached_approval`: an
app double whose `notify` takes the message alone, or no app at all, must not
turn a refusal into an exception on the send path).

The copy distinguishes the two causes rather than emitting one generic string.
With no usable adapter (`persistence is None` or `persistence.db is None` — the
precondition `56db75386` dropped, now used to *choose the wording* instead of to
skip the gate):

> Not sent: your conversation database could not be opened, so this message
> could not be saved. Restart Chatbook, and check the app log for the database
> error if it keeps happening. Your draft was kept; a temporary chat still
> sends.

A persistence adapter that exists but cannot commit durable turns gets the
neutral variant. `should_clear_draft` stays `False`, so the draft survives.
`durable_turn` still exempts ephemeral sessions, so a temporary chat sends
normally in this state.

### Live verification (real `TldwCli`, genuinely unopenable database)

Driven through real config: `[database] chachanotes_db_path` points at a
**directory**, so `sqlite3.connect` fails and `app.__init__` records
`chachanotes_db = None` exactly as it does after a real failed open. Confirmed
in-run: `app.chachanotes_db: None`, `store.persistence: None`.

| observation | pristine dev `a71e62e4b` | this branch |
|---|---|---|
| transcript rows after Send | `[]` | one SYSTEM row naming the database |
| run state | `IDLE` | `BLOCKED` (same copy) |
| toasts (`app._notifications`) | `[]` | 1, severity `error` |
| composer draft | preserved | preserved |
| provider calls | 0 | 0 |
| temporary conversation | sends | sends |

Pressing Send on dev was *completely* silent — no row, no state change, no
toast — which is the defect verbatim. The real profile's `config.toml` and
ChaChaNotes DB were byte-identical before and after every run.

### Mutation results

| mutation | tests that caught it |
|---|---|
| revert to the bare `ConsoleSubmitResult` | `test_unopenable_database_refuses_send_visibly_and_keeps_the_draft`, `test_real_durable_adapter_without_atomic_method_fails_closed` |
| keep the transcript row, drop only the toast | `test_unopenable_database_refuses_send_visibly_and_keeps_the_draft` |
| collapse both causes into the generic copy | `test_unopenable_database_refuses_send_visibly_and_keeps_the_draft` |
| `should_clear_draft=True` on the refusal | `test_unopenable_database_refuses_send_visibly_and_keeps_the_draft`, `test_real_durable_adapter_without_atomic_method_fails_closed` |
| `durable_turn` no longer exempts ephemeral | `test_unopenable_database_still_sends_a_temporary_conversation` |

Every mutation was applied and reverted in place (no `git checkout`), and the
full set is green again afterwards.

### Shutdown / error walk

Degraded-database quit: `has_loss_risk` is False (nothing was sent), the quit
confirmation is not raised, and `ConsoleRuntime.dispose()` completes cleanly
with `persistence=None`.

### Modified or added files

* `tldw_chatbook/Chat/console_chat_controller.py` — `_block_undurable_turn`,
  `_notify_app`, and the gate now calling them.
* `Tests/UI/test_console_degraded_database_send.py` — new (3 tests: degraded
  refusal surface, temporary-conversation exemption, working-DB control arm).
* `Tests/Chat/test_console_first_send_atomicity.py` — the existing fail-closed
  test now asserts the user-visible surface rather than only `visible_copy`.
* `Docs/security/production-diagnostic-inventory.json` — regenerated for the two
  new content-free `logger.debug` calls in `_notify_app` (both reviewed with
  `--statements`; neither interpolates user content, secrets, paths, or URLs).

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
