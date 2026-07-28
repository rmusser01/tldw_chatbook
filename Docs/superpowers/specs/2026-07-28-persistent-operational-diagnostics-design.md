# Persistent operational diagnostics — design

**Task:** TASK-1240
**Date:** 2026-07-28
**Status:** proposed — requires sign-off from the ADR-029 privacy work's owner (see Governance)

## Problem

`tldw_cli_app.log` is zero bytes on every profile, and has been since `1df0c4cb4`
(2026-07-27). The long-lived `default_user` log looks healthy only because its last entry
predates that commit; it is a historical file that stopped growing.

The cause is not a missing handler. `PersistentDiagnosticFilter` admits a record only if it
carries the `_tldw_metadata_only_record` marker, which is set exclusively by
`Utils/persistent_diagnostics.log_persistent_metadata()`. That function has **zero production
call sites**. Every ordinary `logger.info(...)` is rejected, so the sink correctly enforces a
boundary that nothing was ever migrated to cross. `Metrics/logger_config.py` deliberately
disables the alternate Loguru file sinks, so there is no second path.

ADR-029 requires persistent logs be metadata-only **with respect to user and model content** —
prompts, message bodies, provider payloads, key fragments, tool values. It does not call for
excluding operational diagnostics. The privacy design's own goals include *"keep persistent
diagnostics useful without retaining private payload values"* and *"disable only unsafe
persistent file sinks while retaining terminal/UI logs"*. The implementation is stricter than the
decision: it admits nothing, so "useful" is not met.

The cost is concrete. Watchlist checks did nothing for the entire life of the feature
(TASK-1210) and a working scheduler was indistinguishable from an unwired one by observation.
Diagnosing it required a runtime import trace and a seeded database probe. It should have
required reading one log line.

## Goals

- A crash or restart leaves behind enough to reconstruct what the app was doing.
- The persistent sink demonstrably carries records, and cannot silently return to carrying none.
- ADR-029's exclusion list is preserved exactly: no prompt, message body, provider payload, key
  fragment, tool argument or tool result value becomes persistable.

## Non-goals

- Migrating the ~5,247 existing logging call sites. None are touched.
- A support-bundle command. Purpose 2 (a user sends a log with a bug report) is a real goal but
  is not delivered here; this design must not foreclose it.
- Changing terminal or in-app Logs screen behaviour. Those remain descriptive and unaffected.

## Design

### 1. `persist_event()` — one wrapper, one idiom

**Loguru cannot carry these records, and must not be made to.** Verified empirically: the
sanctioned stdlib path writes `event=scheduler_started item_count=2`; the same call routed
through Loguru writes nothing, because `_forward_loguru_to_standard` rebuilds `extra` from
scratch and drops the marker.

That drop is correct and load-bearing. If the marker survived Loguru, any code could write
`logger.bind(_tldw_metadata_only_record=True).info(secret)` and bypass the schema entirely —
defeating ADR-029. The constraint is therefore designed around, not removed.

Since the codebase uses `from loguru import logger` almost everywhere, reaching for the usual
idiom at a persist site would silently write nothing. A thin wrapper in
`Utils/persistent_diagnostics.py` removes that trap:

```python
def persist_event(component: str, event: str, *, level: int = logging.INFO, **fields: Any) -> None:
    """Record one operational event in the persistent log.

    Uses stdlib logging deliberately: the persistent marker does not survive the
    Loguru forwarder, and must not — see module docstring.
    """
    if "component" in fields:
        raise TypeError("component is passed positionally, not as a field")
    log_persistent_metadata(
        logging.getLogger(f"tldw_chatbook.diagnostics.{component}"),
        level, event, component=component, **fields,
    )
```

Call sites never choose between logging libraries, and the one place that can get it wrong is a
single function with a test on it.

Two details in that signature are deliberate:

- **A `tldw_chatbook.diagnostics.*` namespace, not the caller's module logger.** Naming the
  logger `tldw_chatbook.app` would collide with the real module's records, interleaving persisted
  events with descriptive ones and exposing them to any per-logger level configuration aimed at
  that module. The distinct namespace keeps persisted events greppable and independently
  configurable, and still satisfies `_is_chatbook_record`, which requires the `tldw_chatbook.`
  prefix.
- **`component` is positional and rejected as a field**, so a caller cannot pass it twice and get
  a confusing `TypeError` from the inner call.

**These records also reach the terminal and the in-app Logs screen**, because they go through the
root logger like everything else. That is intended — a persisted event is worth seeing live — but
it means the event set is a UI surface too, which is a further reason to keep it small.

### 2. One new schema field: `component`

Added to `_TOKEN_FIELDS`, validated by the existing token regex like every other token field.
It carries code-side identifiers only (`scheduling`, `app`, `logging`), never user data. This is
the only widening of the admitted surface in this design.

### 3. The event set

Seven events, each tied to a failure this project has actually had. All fields already exist in
the schema apart from `component`.

| Event | Component | Fields | Why |
| --- | --- | --- | --- |
| `app_started` | `app` | — | Anchors a session; its absence dates a crash. |
| `app_stopping` | `app` | — | Distinguishes a clean exit from a kill. |
| `persistent_sink_installed` | `logging` | `status` | Emitted immediately after install, so an empty file is unambiguous. |
| `worker_started` | caller's | `operation` | Which background work began. |
| `worker_failed` | caller's | `operation`, `exception_type` | The TASK-1210 class: a worker that dies leaves a trace. |
| `scheduler_configured` | `scheduling` | `item_count`, `status` | Handlers registered, and whether queued work had none. |
| `unhandled_exception` | caller's | `exception_type`, `error_category` | A crash names its type without its message. |

No message text, no traceback, no paths. `exception_type` is a class name, which is a code-side
identifier.

**Emission points.** The listed events are not scattered across the codebase; each has one
defined home, and the two that could have sprawled do not:

| Event | Emitted from |
| --- | --- |
| `app_started`, `app_stopping` | `TldwCli.on_mount` / `on_unmount` |
| `persistent_sink_installed` | `Logging_Config._configure_private_file_logging`, after `addHandler` |
| `worker_started`, `worker_failed` | `TldwCli.on_worker_state_changed` — **one existing hook** that already sees every worker transition; `WorkerState.ERROR` carries the exception on `event.worker.error` |
| `scheduler_configured` | `SchedulerLoop.report_configuration` (added in TASK-1212) |
| `unhandled_exception` | `App._handle_exception` override |

The worker pair is the load-bearing one: without a central hook it would have meant editing every
`run_worker` call site, which is the kind of sprawl that made this area expensive in the first
place.

Deliberately excluded for now: provider/model call outcomes (they serve the unrealized
support-bundle goal), and an app `version` field (would need an eighth schema field; add it with
the bundle work, when there is a consumer).

### 4. Surfacing install failure

`_configure_private_file_logging` catches `Exception`, logs a warning and returns `False`, so a
permissions or path problem yields an empty log forever — the same silent-failure class as
TASK-1240 itself. `persistent_sink_installed` is emitted immediately after a successful install,
which makes the two states distinguishable: a file with one line means the sink works and
nothing else has happened; an empty file means the sink did not install.

## Testing

- **The guard that would have caught this.** Install the real sink into a `tmp_path` via
  `_configure_private_file_logging`, run the startup emitter, assert the file is **non-empty**.
  Asserting a handler is attached is what passes today against an empty log.
- **…and that guard must not be satisfiable by its own install line.** `persistent_sink_installed`
  is written the moment the sink installs, so "non-empty" alone would pass even if every other
  event were broken — the same vacuous shape this repo keeps paying for. The assertion is
  therefore on **named events**: the file must contain `event=app_started` *and* at least one
  event that is not `persistent_sink_installed`.
- **The boundary still holds.** An ordinary `logger.info` on the same logger is rejected; a
  Loguru-routed record carrying the marker is rejected. Both asserted, because both are
  security properties rather than incidental behaviour.
- **The wrapper is the only idiom.** A test that `persist_event` writes and that its output
  parses as `event=… component=…` key/value pairs.
- **The emitter actually runs.** A boot-path test that the startup emitter fires, rather than a
  source scan for the call — this repo has been burned by name-matching guards that pass
  vacuously.
- The existing sentinel matrices and `Tests/test_persistent_diagnostic_boundary.py` must pass
  unchanged.

## Governance

ADR-029 is **Accepted**, with a design spec, a checked inventory
(`Docs/security/production-diagnostic-inventory.json`, 401 owners) and task series 489–494. This
design adds one admitted field and seven admitted events to a boundary that work owns.

The ADR amendment recording that operational metadata events are in scope requires that owner's
sign-off. It is not a unilateral doc edit, for the same reason TASK-1240 was filed rather than
fixed.

## Risks

- **The allowlist rots.** Seven events could drift out of date as the app changes. The non-empty
  guard catches total regression but not staleness; that is accepted, because the alternative
  (an automatic adapter) trades the guarantee for coverage and was rejected.
- **`component` is a new admitted field.** Small, validated, code-side only — but it is the one
  place this design widens what can be written.
