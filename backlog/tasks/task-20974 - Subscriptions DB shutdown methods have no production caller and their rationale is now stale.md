---
id: TASK-20974
title: >-
  Subscriptions DB shutdown methods have no production caller and their
  rationale is now stale
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - bug
  - database
  - shutdown
  - durability
priority: medium
dependencies:
  - TASK-19562
  - TASK-19561
---

## Description

Source: noted but deliberately not fixed by **TASK-19562**, which added the two
methods. Re-verified at `684c6aba4`, after TASK-19561 merged (`5622f6987`).

`SubscriptionsDB.checkpoint_wal()` (`DB/Subscriptions_DB.py:1608`) and
`SubscriptionsDB.close_all_connections()` (`:1659`) exist and are tested, but
nothing in the running application calls them. Their only caller is
`_checkpoint_open_databases_at_exit` (`:212`), registered via `atexit`
(`:1605`) — and that hook's own docstring says plainly that it is *not* what
saves the `-wal` on an ordinary exit, because CPython finalizes the connection
objects and SQLite checkpoints when the last connection closes. That was tested
when it was written: a child process that wrote a 4.1 MB `-wal` and exited
normally left only `subs.db`, identically with the hook enabled and suppressed.

So the two methods are, in practice, machinery with no moment at which they
run for the reason they were written.

**TASK-19562 said wiring them in belonged with TASK-19561's signal path.
TASK-19561 has now merged, and it built exactly that seam** — `Utils/
app_shutdown.py`, with one process-level `SIGTERM`/`SIGINT` handler that
answers the first signal with an ordinary `App.exit()` and bounds the exit with
a watchdog. `os._exit(0)` no longer appears anywhere in `app.py`.

That makes the docstring at `DB/Subscriptions_DB.py:193-196` **false as
written**. It still says:

> "The path where the `-wal` genuinely does survive is `app.py`'s SIGINT/SIGTERM
> handler, which calls `os._exit(0)` — that skips `atexit` too, so no hook here
> can reach it."

The premise is gone. A hook *can* now reach that path, because that path is now
an ordinary shutdown. A comment that describes a mechanism the code no longer
has is worse than no comment: the next reader takes it as a reason not to look.
This is the same "asserts something it did not produce" theme the holistic
review found throughout.

## Acceptance Criteria

- [ ] `checkpoint_wal()` and `close_all_connections()` are reachable from the
      application's real shutdown path, or are removed
- [ ] If wired in, the seam they attach to is named explicitly, and the
      behaviour is verified against a real termination rather than by unit test
      alone
- [ ] If wired in, shutdown remains bounded — settling the database cannot make
      a slow exit into a hung one
- [ ] `DB/Subscriptions_DB.py`'s docstrings no longer describe an `os._exit(0)`
      signal path that no longer exists, and state accurately what the `atexit`
      hook does and does not achieve
- [ ] The already-tested facts are preserved and not re-litigated: a clean exit
      does not leave a `-wal` behind, and SQLite refuses a cross-thread close, so
      other threads' connections are reported rather than closed
- [ ] A test fails if these methods lose their production caller again

## Notes

The honest framing is "wire them in — the seam now exists", not "these are dead
code". They were written for a real hazard (a standing `-wal`, and connections
abandoned by a hard exit); what was missing was a defined moment to run them,
and TASK-19561 has now created one.

Removing them instead is an acceptable outcome of this task, but only with the
hazard explicitly re-checked against the post-TASK-19561 shutdown — not by
inheriting the stale docstring's conclusion.
