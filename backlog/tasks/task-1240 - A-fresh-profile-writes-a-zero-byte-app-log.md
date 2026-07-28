---
id: TASK-1240
title: The persistent app log admits nothing, so every profile writes a zero-byte log
status: To Do
assignee: []
created_date: '2026-07-28 10:20'
labels:
  - logging
  - observability
  - privacy
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
- [ ] #1 A decision is recorded on which operational diagnostics may be persisted, consistent with ADR-029's scope (user and model content) rather than the current admit-nothing behaviour
- [ ] #2 If operational diagnostics are to be persisted, representative ones - scheduler startup and handler registration, background worker failures, unhandled exceptions - reach the file log through the metadata-only API
- [ ] #3 The boundary continues to reject prompts, message bodies, provider payloads, key fragments and tool values, with the existing sentinel matrices still passing
- [ ] #4 A test asserts the log is non-empty after a boot path, rather than asserting a handler is attached
- [ ] #5 If admitting nothing is the intended end state, ADR-029 and the privacy design's "keep persistent diagnostics useful" goal are amended to say so, and the app documents where operational diagnostics can be read instead
<!-- AC:END -->
