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
- **`component` is positional**, so passing it twice is already a `TypeError` from Python itself
  ("got multiple values for argument 'component'"). An explicit guard would be unreachable — the
  interpreter raises before the function body runs. A first draft specified one, and the Task 1
  review caught that both it and its test were dead.

**These records also reach the terminal and the in-app Logs screen**, because they go through the
root logger like everything else. That is intended — a persisted event is worth seeing live — but
it means the event set is a UI surface too, which is a further reason to keep it small.

### 2. One new schema field: `component`

Added to `_TOKEN_FIELDS`, validated by the existing token regex like every other token field.
It carries code-side identifiers only (`scheduling`, `app`, `logging`), never user data. This is
the only widening of the admitted surface in this design.

### 3. The event set

Six events, each tied to a failure this project has actually had. All fields already exist in
the schema apart from `component`.

| Event | Component | Fields | Why |
| --- | --- | --- | --- |
| `app_started` | `app` | — | Anchors a session; its absence dates a crash. |
| `app_stopping` | `app` | — | Distinguishes a clean exit from a kill. |
| `persistent_sink_installed` | `logging` | `status` | Emitted immediately after install, so an empty file is unambiguous. |
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
| `worker_failed` | `TldwCli.on_worker_state_changed` — **one existing hook** that already sees every worker transition; `WorkerState.ERROR` carries the exception on `event.worker.error` |
| `scheduler_configured` | `SchedulerLoop.report_configuration` (added in TASK-1212) |
| `unhandled_exception` | `App._handle_exception` override |

`worker_failed` is the load-bearing one: without a central hook it would have meant editing every
`run_worker` call site, which is the kind of sprawl that made this area expensive in the first
place.

**Why there is no `worker_started`.** An earlier draft had one. The app has 398 `run_worker` call
sites and 118 `@work` decorators, and `on_worker_state_changed` fires on every transition — so a
start event would emit a line per worker, including keystroke-triggered searches and periodic
timers. Since these records also reach the terminal and the Logs screen, that is volume in three
places at once, for a signal that says only "something began". Failures are rare and diagnostic;
starts are neither. Dropped.

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

- **Two halves, together closing the original gap; neither proves the other alone.** The original
  defect was that `log_persistent_metadata()` had zero production call sites — the wrapper →
  filter → handler → file machinery worked correctly throughout. Coverage is split to match that
  shape, and the split must be read as two halves, not one guard:
  - `Tests/test_persistent_log_is_not_empty.py` installs the real sink into a `tmp_path` via
    `_configure_private_file_logging` and calls `persist_event` directly, asserting the file ends
    up **non-empty** with **named events** (`event=app_started` and at least one event that is not
    `persistent_sink_installed`; a bare non-empty check is satisfiable by the install line alone).
    This proves a `persist_event` call reaches the file. Its caller is **synthetic** — no app is
    booted — so this half says nothing about whether production ever makes that call.
  - `Tests/App/test_app_lifecycle_events.py` and `Tests/Scheduling/test_scheduler_observability.py`
    monkeypatch `persist_event` to a recorder and boot the real app / scheduler, proving production
    **does** call it at the intended sites. But the monkeypatch means nothing in these tests ever
    reaches the file.
  Only together do the two halves cover the original defect: something calls `persist_event`, and a
  call from that something reaches the file. **The machinery half alone — a real sink exercised by
  a synthetic caller — would not have caught the original defect**, because the original defect was
  entirely a missing caller, not broken machinery. Neither half should be described as "the guard
  that would have caught this"; only the pair does. Composing a real production emitter with a real
  sink in one test remains an open gap — see Risks.
- **The boundary still holds.** An ordinary `logger.info` on the same logger is rejected; a
  Loguru-routed record carrying the marker is rejected. Both asserted, because both are
  security properties rather than incidental behaviour.
- **The wrapper is the only idiom.** A test that `persist_event` writes and that its output
  parses as `event=… component=…` key/value pairs.
- The existing sentinel matrices and `Tests/test_persistent_diagnostic_boundary.py` must pass
  unchanged.

## Relationship to metrics and OpenTelemetry

The admitted schema reads like OpenTelemetry span attributes — `provider`, `model`, `operation`,
`status`, `duration_ms`, `status_code`, `cache_hit`, `streaming`, `retry_count` — and this repo
does have a real telemetry layer: `Metrics/Otel_Metrics.py`, wired at `app.py:8331`, whose own
docstring warns about attribute cardinality in the same terms `_TOKEN_RE` enforces. It is
reasonable to wonder whether the metadata-only boundary was really a telemetry design.

It was not. `persistent_diagnostics.py` describes itself as a *"strict admission boundary for
metadata written to persistent diagnostics"*, and it arrived with ADR-029 and task series
489-494 — privacy hardening, not observability. The vocabularies converge because both problems
have the same answer: structured, low-cardinality, no payloads.

**They cannot substitute for each other**, which is why this design does not simply route the six
events to OTel:

| | `Metrics/` + OTel | Persistent diagnostics |
| --- | --- | --- |
| Consumer | Prometheus/OTLP collector | a file on the user's disk |
| Shape | aggregates over many runs | individual events in one run |
| Available to a user filing a bug | no — needs a collector running | yes |
| Survives a crash on a laptop | no | yes |

Purposes 1-3 in this design (post-mortem, support bundle, developer debugging) all require bytes
on local disk after the process is gone. Aggregate monitoring is a different job that the metrics
layer already does.

**Two facts worth carrying forward.**

`metrics_logger._log_metric` emits through `logger.bind(event=..., type=..., value=..., ...)` —
Loguru. So metrics hit exactly the wall described in §1: the persistent filter rejects Loguru
records, and metric lines never reach the file either. The two systems are separated by transport
rather than by any deliberate decision about what belongs where.

Both use an `event=` vocabulary and neither shares a schema with the other. New fields added here
should reuse `metrics_logger`'s label names where the concept already exists, so the two do not
drift into separate dialects for the same idea. `component` is consistent with that: it names a
subsystem, the way a metric label would.

## Governance

ADR-029 is **Accepted**, with a design spec, a checked inventory
(`Docs/security/production-diagnostic-inventory.json`, 401 owners) and task series 489–494. This
design adds one admitted field and six admitted events to a boundary that work owns.

The ADR amendment recording that operational metadata events are in scope requires that owner's
sign-off. It is not a unilateral doc edit, for the same reason TASK-1240 was filed rather than
fixed.

## Risks

- **Volume.** Six events per session is negligible. That holds only while the set stays failure-
  shaped; any future event that fires per-operation needs the same volume check `worker_started`
  failed.
- **The allowlist rots.** Six events could drift out of date as the app changes. The non-empty
  guard catches total regression but not staleness; that is accepted, because the alternative
  (an automatic adapter) trades the guarantee for coverage and was rejected.
- **`component` is a new admitted field.** Small, validated, code-side only — but it is the one
  place this design widens what can be written.
- **No test composes a real production emitter with a real installed sink.** Coverage is split
  across two halves (see Testing): a synthetic caller against a real sink, and monkeypatched
  production callers against no sink at all. An ordering regression — `app.py` emitting
  `app_started` *before* `Logging_Config` installs the persistent sink — would pass both halves
  while the event reaches nowhere, reproducing the original TASK-1240 failure mode (machinery
  works, caller calls, file stays empty) in a new form. Today the correct order holds only by
  entry-point accident (`app.py` sets `_early_logging_initialized` before `run()`), and that
  ordering is itself untested. Tracked as [TASK-1330](../../../backlog/tasks/task-1330%20-%20Prove-app_started-is-never-emitted-before-the-persistent-sink-installs.md).
