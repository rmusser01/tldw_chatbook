---
id: TASK-1240
title: 'The persistent app log admits nothing, so every profile writes a zero-byte log'
status: In Progress
assignee: []
created_date: '2026-07-28 10:20'
updated_date: '2026-07-29 01:49'
labels:
  - logging
  - observability
  - privacy
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Corrected diagnosis (2026-07-28).** This was filed as "a fresh profile writes a zero-byte app
log", on the evidence that a new profile produced 0 bytes while long-lived `default_user` held
8.4 MB. That framing was wrong in two ways: it is not specific to fresh profiles, and the cause is
not a missing handler.

`PersistentDiagnosticFilter` is attached to the one persistent file sink
(`PrivateRotatingFileHandler`) and admits a record only when it carries the
`_tldw_metadata_only_record` marker, which is set exclusively by
`Utils/persistent_diagnostics.log_persistent_metadata()`:

```python
def filter(self, record):
    if _is_chatbook_record(record):
        return getattr(record, _PERSISTENT_METADATA_MARKER, False) is True
    return False          # third-party records are rejected outright
```

**`log_persistent_metadata` has zero production call sites.** Every operational diagnostic in the
app goes through `logger.info(...)` / loguru and is therefore rejected. The sink is correctly
enforcing a boundary that nothing has been migrated to cross.

`Metrics/logger_config.py` deliberately disables the alternate Loguru file sinks, so there is no
second path. Terminal and in-app UI handlers are unaffected and remain descriptive, which is why the
Logs screen still works and made the file log look like the anomaly.

**It affects every profile, not new ones.** The filter reached the file handler in `1df0c4cb4`
(2026-07-27). `default_user`'s log looks healthy only because its last entry is 2026-07-26 — it is
a historical file that stopped growing when the filter landed. Any profile, old or new, has written
nothing since.

**Where the gap is.** ADR-029 requires that "persistent application logs are metadata-only **with
respect to user and model content**", listing prompts, message bodies, provider payloads, key
fragments and tool values. It does not call for excluding operational diagnostics. The privacy
design's own goals include "keep persistent diagnostics **useful** without retaining private payload
values" and "disable only unsafe persistent file sinks while retaining terminal/UI logs". The
implementation is stricter than the decision: it admits nothing at all, so "useful" is not met.

**Why this is not a unilateral fix.** Changing what reaches this sink means changing a deliberate
security boundary with its own ADR (029), design spec, inventory
(`Docs/security/production-diagnostic-inventory.json`) and task series (489-494). The decision of
which operational diagnostics may be persisted, and in what shape, belongs to that work's owner.
This task records the gap and the evidence; it should not be closed by loosening the filter.

**Why it matters.** Watchlist checks did nothing for the entire life of the feature and it went
unnoticed because a working scheduler and an unwired one were indistinguishable by observation
(TASK-1210, TASK-1212). Diagnosing it needed a runtime import trace and a seeded database probe.
With an operational log it would have needed one line. TASK-1212 added structured scheduler startup
reporting that currently has nowhere to land.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A decision is recorded on which operational diagnostics may be persisted, consistent with ADR-029's scope (user and model content) rather than the current admit-nothing behaviour
- [x] #2 If operational diagnostics are to be persisted, representative ones - scheduler startup and handler registration, background worker failures, unhandled exceptions - reach the file log through the metadata-only API
- [x] #3 The boundary continues to reject prompts, message bodies, provider payloads, key fragments and tool values, with the existing sentinel matrices still passing
- [x] #4 A test asserts the log is non-empty after a boot path, rather than asserting a handler is attached
- [x] #5 If admitting nothing is the intended end state, ADR-029 and the privacy design's "keep persistent diagnostics useful" goal are amended to say so, and the app documents where operational diagnostics can be read instead
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Status stays In Progress, not Done — see Sign-off gate below.**

## What shipped

Six operational events reach the persistent log through a new `persist_event(component, event,
*, level, **fields)` wrapper in `Utils/persistent_diagnostics.py`: `app_started`, `app_stopping`
(`TldwCli.on_mount`/`on_unmount`), `persistent_sink_installed` (`Logging_Config`, immediately
after `addHandler`, so an empty file is unambiguous), `worker_failed` (the single existing
`on_worker_state_changed` hook — no per-call-site changes to 398 `run_worker` sites), `unhandled_exception`
(`App._handle_exception` override, type only, no message), and `scheduler_configured`
(`SchedulerLoop.report_configuration`, discriminating handler count from queue depth and orphaned
types). One schema field, `component`, was added to `_TOKEN_FIELDS` for this.

**The Loguru path stays deliberately closed.** `_forward_loguru_to_standard` rebuilds `extra` from
scratch and drops the `_tldw_metadata_only_record` marker; `persist_event` always goes through the
stdlib logger, never Loguru. This is pinned as a security property, not an oversight, by
`Tests/test_persistent_diagnostic_boundary.py` (mutation-checked: reintroducing the marker through
a Loguru `.bind()` call is asserted to still be rejected).

**The non-empty guard asserts named events, not bare non-emptiness.** `persistent_sink_installed`
alone would satisfy `assert file_is_non_empty` even if every other event were broken — the same
vacuous-guard shape this repo has been burned by before (`Tests/test_persistent_log_is_not_empty.py`).
The assertion is therefore on `event=app_started` *and* at least one event that is not
`persistent_sink_installed`.

**Coverage is two disjoint halves, corrected in the spec's Testing section (this was previously
mis-stated as one guard "that would have caught this").** `Tests/test_persistent_log_is_not_empty.py`
proves a `persist_event` call reaches the file with a synthetic (non-booted) caller.
`Tests/App/test_app_lifecycle_events.py` and `Tests/Scheduling/test_scheduler_observability.py`
monkeypatch `persist_event` to a recorder and boot the real app/scheduler, proving production
calls it, but nothing reaches a file in those tests. Neither half alone would have caught the
original TASK-1240 defect (a real sink with no caller); only the pair does. A residual gap — no
test composes a real emitter with a real installed sink, so an app.py-before-sink-install
ordering regression would pass both halves silently — is recorded in the spec's Risks section and
tracked as TASK-1330 (filed To Do, not started).

Full affected suite (`Tests/Utils/ Tests/Scheduling/ Tests/App/` plus the five persistent-diagnostics
test files) passes: 848 passed, 0 failed.

## Sign-off gate (do not merge without this)

AC #1 and #5 are satisfied by documentation, not by unilateral authority: the clarification of
ADR-029's "metadata-only" scope to admit these six events is written as a **proposed amendment**
in `backlog/decisions/029-local-private-data-boundary.md`, marked pending and explicitly not
authoritative, because this branch adds one admitted field and six admitted events to a privacy
boundary owned by ADR-029/task series 489-494. The branch cannot merge until that ADR's owner
signs off on the amendment. This task is left at **In Progress** rather than **Done** for exactly
that reason — marking it Done would misrepresent an unratified privacy-boundary change as settled.

## Modified/added files

- `tldw_chatbook/Utils/persistent_diagnostics.py` (`persist_event`, `component` field) — Task 1
- `tldw_chatbook/app.py`, `Logging_Config.py`, `Scheduling/scheduler/loop.py` — event emission sites, Tasks 3-7
- `Tests/Utils/test_persist_event.py`, `Tests/App/test_app_lifecycle_events.py`,
  `Tests/App/test_worker_failure_event.py`, `Tests/App/test_unhandled_exception_event.py`,
  `Tests/Scheduling/test_scheduler_observability.py`, `Tests/test_persistent_log_is_not_empty.py`,
  `Tests/test_persistent_diagnostic_boundary.py` (new/extended assertions) — Tasks 1-8
- `Docs/superpowers/specs/2026-07-28-persistent-operational-diagnostics-design.md` — Testing
  section corrected, Risks section extended, Governance "seven" → "six" fixed (Task 9)
- `backlog/decisions/029-local-private-data-boundary.md` — proposed amendment recorded, pending
  sign-off (Task 9)
- `backlog/tasks/task-1330 - ...md` — new follow-up task filed for the residual ordering gap (Task 9)
<!-- SECTION:NOTES:END -->
