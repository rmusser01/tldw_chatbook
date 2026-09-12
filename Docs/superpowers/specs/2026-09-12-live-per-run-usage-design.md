# Live Per-Run Usage Design

**Task:** TASK-18923  
**Date:** 2026-09-12  
**Status:** Accepted design  
**Decision:** ADR-156 (`backlog/decisions/156-live-per-run-stream-usage-attribution.md`)

## Purpose

Show one honest, bounded live output-token scalar for every active Console agent run. The primary activity line and each fleet row update through the timers that already repaint those surfaces. Final run accounting, automatic-work budgets, pricing, persistence, and historical rendering remain unchanged.

## User-visible contract

- During an active model call, the row may append `N provider output tok` when the provider has reported an output count.
- Until such a report is observable, actual received textual stream data may produce `~N local output tok`, computed from cumulative UTF-8 bytes divided by four and rounded up.
- Provider data is authoritative once observed. An explicit provider zero is valid and suppresses the local estimate; the formatter omits a zero-valued segment until the provider advances it.
- No count is shown before text or valid provider usage is observed. No dollar amount, projected price, or fabricated precision is shown.
- The scalar describes the current model call. It resets between calls and disappears when that call finishes. A fleet row's existing terminal `budget tok` remains the final budget/accounting value and is never relabelled as live usage.
- Existing setup, approval, tool, fleet, thinking, generating, elapsed, ordering, drill-in, and error behavior remains intact. Usage augments only generating/thinking activity and active child rows; it does not make a stale prior call appear during tools or approval waits.

## Data model

Add immutable bridge-facing values in `console_agent_bridge.py`:

```python
LiveUsageSource = Literal["provider", "local"]

@dataclass(frozen=True, slots=True)
class AgentLiveTurnUsage:
    output_tokens: int
    source: LiveUsageSource
    started_at: float
    sequence: int

@dataclass(frozen=True)
class AgentLiveSnapshot:
    # existing fields unchanged
    turn_usage: AgentLiveTurnUsage | None = None
```

The mutable implementation is private and bounded:

```python
@dataclass(slots=True)
class _LiveTurnUsageAccumulator:
    sequence: int
    started_at: float
    received_utf8_bytes: int = 0
    provider_output_tokens: int | None = None
    last_published_at: float | None = None
    published: AgentLiveTurnUsage | None = None
```

`ConsoleAgentBridge` owns `_live_turn_usage: dict[str, _LiveTurnUsageAccumulator]`, keyed only by active `run_id`, behind a `threading.Lock`. It never retains chunks, messages, per-chunk records, tokenizer objects, or finished-call history. The map is bounded by simultaneously active runs and entries are removed at model-call finish, run terminal cleanup, survivor pruning, and session shutdown.

## Exact run attribution

`AgentService` gains one optional internal service contract:

```python
run_model_scope: (
    Callable[[str, str], contextlib.AbstractContextManager[None]] | None
) = None
```

`_run_one` enters `run_model_scope(run_id, agent_kind)` immediately around `run_agent_loop`. A missing callback uses `contextlib.nullcontext()`, preserving every non-Console caller. The scope always exits on success, cancellation, and exception.

`_StreamingModelAdapter.run_scope(run_id, agent_kind)` stores the attribution in thread-local state because primary and fleet children share one adapter while invoking it from different threads. `chat_call` reads the scope before submitting the async coroutine and passes the captured immutable `(run_id, agent_kind)` into that coroutine; it never attempts to read the thread local on the event-loop thread.

The adapter emits a single typed callback rather than exposing its mutable state:

```python
@dataclass(frozen=True, slots=True)
class AgentLiveUsageEvent:
    kind: Literal["started", "text", "provider_usage", "finished"]
    run_id: str
    agent_kind: str
    sequence: int
    observed_at: float
    text: str = ""
    provider_output_tokens: int | None = None

live_usage_sink: Callable[[AgentLiveUsageEvent], None] | None = None
```

The adapter uses one monotonic sequence allocator for its lifetime (or an equivalently bounded active-scope allocator), never an unpruned dictionary of finished run IDs. It assigns an increasing sequence per attributed run and emits `started` before consuming a model stream, `text` for each actual provider textual delta, `provider_usage` after normalizing observable usage snapshots, and `finished` from `finally`. Structured tool-call objects and locally synthesized fallback copy do not contribute bytes. Reasoning/thinking text received from the provider is output text and contributes when it is represented as a textual delta.

### Gateway emission provenance

The adapter may opt into an optional keyword-only `ConsoleProviderGateway.stream_chat` emission observer receiving the existing per-emission synthetic boolean immediately before each yield. Default callers receive exactly the same item types and behavior. The adapter retains one call-local scalar flag, consumes/resets it for every item before any early continue, and excludes only explicitly synthetic text from live byte counting. It does not infer origin from the string or add provider-specific branches. An observer failure is contained and cannot alter stream, transcript or accounting behavior; any diagnostic is fixed, content-free and bounded to one per stream call. This carries existing producer provenance through the boundary required by ADR-156; it creates no permission or configuration policy.

## Validation and arbitration

Provider output counts are accepted only from recognized output/completion fields after the existing partial usage normalization, with `type(value) is int and value >= 0`. This deliberately rejects booleans, floats, numeric strings, negatives, missing fields, and aggregate-only totals. Zero is a valid provider observation; malformed later payloads cannot erase a valid provider value.

Before provider observation, local output tokens are:

```python
ceil(received_utf8_bytes / 4)
```

and exist only when `received_utf8_bytes > 0`. Computing from the cumulative byte count avoids chunk-boundary inflation. Provider observation permanently wins for that sequence, including zero.

Each event must match the active map entry's sequence. A new `started` replaces the prior call for that run. Late text, usage, or finish events from an older sequence are ignored, so one call cannot overwrite its successor.

## Publication and UI cadence

The bridge updates the accumulator on every event but replaces the immutable `published` scalar immediately only for the first observable value and thereafter at most once per second. A source change from local to provider also respects this ceiling. `finished` clears the ephemeral scalar rather than forcing an extra live repaint. This bounds snapshot churn without delaying the first useful count.

The existing 0.2-second primary transcript poll and 1-second survivor tick remain the only UI clocks. Snapshot equality continues to prevent unchanged DOM writes. No per-chunk Textual message, timer, worker, or repaint is added.

`console_turn_activity_text` appends the live label to generating/thinking output and uses `turn_usage.started_at` to supply the pre-first-step elapsed base. Tool and approval branches keep their existing text and omit prior-call usage. `_fleet_row_from_handle` accepts `live_snapshot: AgentLiveSnapshot | None = None` and appends that child's live usage label without changing primary text, ordering, status, cancellation, steering, or terminal budget display.

## Lifecycle and failures

- Start: allocate/replace exactly one accumulator for `(run_id, sequence)`.
- Stream: update cumulative bytes or validated provider scalar under the bridge lock.
- Finish: remove only when sequence matches. The next tool/model call starts cleanly.
- Run terminal, cancellation, adapter exception, survivor pruning, and bridge shutdown: defensively remove that run's entry even if `finished` was not delivered.
- Missing attribution or sink: streaming proceeds normally and no live count is shown.
- Callback failure: contain it and emit a fixed, content-free diagnostic with at most a bounded exception type; never log text deltas, payloads, arbitrary exception repr, or per-chunk error spam. Live telemetry must never fail or cancel the model call.
- The final `RunOutcome.total_tokens`, persisted provider accounting, cache weighting, and fleet `total_tokens` path are untouched.

## Security and privacy

The bridge counts `len(delta.encode("utf-8"))` and discards the delta. It stores no prompt, response text, credentials, provider payload, pricing data, or user configuration. Run IDs originate inside `AgentService`; UI callers only request snapshots for already attached run IDs. The new callback does not grant permissions or broaden filesystem, MCP, provider, or database access.

## Verification

Tests must cover local cumulative estimation across chunk boundaries, provider precedence, explicit zero, invalid values, at-most-one-second publication, sequence reset and stale-event rejection, concurrent sibling isolation, callback containment, all cleanup paths, unchanged final accounting, primary labels, per-child labels, ordering, terminal budget semantics, and reuse/self-stop of the existing timers.

## Scope

Expected implementation files:

- `tldw_chatbook/Agents/agent_service.py`
- `tldw_chatbook/Chat/console_agent_bridge.py`
- `tldw_chatbook/UI/Console_Modules/agent.py`
- `Tests/Agents/test_agent_service.py`
- `Tests/Chat/test_console_agent_bridge.py`
- `Tests/UI/test_console_turn_activity_line.py`
- `Tests/UI/test_console_fleet_panel.py`
- `Tests/UI/test_console_fleet_survivor_tick.py`
- `Tests/UI/test_console_fleet_lifecycle_controller.py`
- `Docs/User_Guide/console/agent-runs-and-tools.md`

No schema, migration, provider-specific adapter, pricing, run-log, CSS, or historical snapshot change is required.

