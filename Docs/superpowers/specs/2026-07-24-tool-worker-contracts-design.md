# Tool Worker Contracts Design

Date: 2026-07-24
Status: Superseded on current dev; retained as a verified historical design
ADR:
[ADR-031](../../../backlog/decisions/031-bounded-evaluation-and-tool-worker-execution.md)
Backlog:
[TASK-545](../../../backlog/tasks/task-545%20-%20Wire-built-in-tool-executor-into-MCP-permission-gate.md)

## Current-Dev Reconciliation

Do not implement the `ToolExecutor` work described below. After this design
was reviewed, TASK-545 verified that System A had zero production execution
callers and removed `ToolExecutor`, `ToolResultCache`, `get_tool_executor()`,
and `reload_tool_executor()`. Reintroducing those symbols to satisfy this
historical plan would restore a dead parallel tool system and conflict with
the live MCP permission-gated provider architecture.

The callback, public-cancellation, and batch-cleanup findings remain valid for
the deleted implementation and are retained as an audit record. They are not
public contracts of the current application. The only live change from this
document is the cross-thread MCP submission ownership rule: close a coroutine
when `run_coroutine_threadsafe()` rejects it before ownership transfers.

See:

- [System A retirement design](2026-07-26-retire-system-a-design.md)
- [System A retirement plan](../plans/2026-07-26-retire-system-a.md)
- [Superseded implementation plan](../plans/2026-07-24-tool-worker-contracts.md)

## Summary

Historical proposal: make `ToolExecutor.max_workers` a real async concurrency
limit and remove the
unused thread pool. Tool timeouts remain contained error results. Cancellation
becomes explicit control flow: every begun call cancelled while queued or
executing receives a terminal history record and re-raises, and batch execution
no longer swallows child cancellation or abandons siblings. Existing async
tool, cache, result-order, and ordinary error-isolation contracts remain
intact.

## Verified Problems

- `ToolExecutor` creates `ThreadPoolExecutor(max_workers=...)`, but no execution
  path submits work to it.
- A five-tool reproduction reached five simultaneous `Tool.execute()` calls
  with `max_workers=2`.
- A cancellation after the `started` history entry bypasses the timeout and
  ordinary exception handlers, leaving a permanently non-terminal record.
- `asyncio.gather(..., return_exceptions=True)` is redundant for ordinary tool
  errors, which leaf execution already converts to result dictionaries, and
  can treat child cancellation as batch data instead of control flow.
- Plain `asyncio.gather()` would propagate a child cancellation but would not,
  by itself, guarantee that every unfinished sibling is cancelled and drained.
  The batch must explicitly own and clean up its child tasks.
- `MCPToolProvider._execute()` constructs the control-plane coroutine inline as
  an argument to `asyncio.run_coroutine_threadsafe()`. When the target loop is
  already closed, submission raises before ownership transfers and the
  coroutine is never closed. The existing closed-loop regression passes while
  emitting `RuntimeWarning: coroutine ... was never awaited`.
- `reload_tool_executor()` shuts down the unused pool, coupling reload to
  infrastructure that never executed tools.

## Goals

- Validate execution limits at construction.
- Bound actual simultaneous `Tool.execute()` calls across the executor.
- Keep queue time separate from execution timeout.
- Preserve per-call timeout and ordinary error results.
- Record cancellation regardless of whether a call is queued or executing.
- Propagate cancellation through single and batch APIs.
- Cancel and drain every unfinished batch sibling on cancellation or unexpected
  child control flow.
- Close a cross-thread MCP execution coroutine when loop submission rejects it
  before ownership transfers.
- Preserve batch result order and cache behavior.
- Remove dead pool construction, shutdown, and destructor code.

## Non-Goals

- Running async tools in worker threads or subprocesses.
- Adding sync tool support; `Tool.execute()` remains async.
- Preemptively terminating arbitrary code that suppresses cancellation.
- Prioritization, per-tool limits, rate limiting, or a persistent work queue.
- Sharing one `ToolExecutor` concurrently across multiple event loops.
- Changing tool schemas, registration policy, cache format, or log privacy.
- Repairing unrelated Textual/Console worker-group races.

## Construction Contract

`max_workers` must be a positive integer and `timeout_seconds` a positive finite
number; booleans are rejected. Invalid values raise before the executor can
register or run tools.

The executor stores one `asyncio.Semaphore(max_workers)` and no
`ThreadPoolExecutor`. This matches the application-owned single event loop and
the existing async `Tool.execute()` abstraction.

## Single-Call Lifecycle

Validation and cache lookup keep their current behavior. Only an uncached,
valid call enters the execution limiter.

1. establish the call's start time and metadata;
2. validate the tool and arguments;
3. return a valid cached result without acquiring execution capacity;
4. wait for the semaphore;
5. once admitted, apply `timeout_seconds` to `tool.execute()`;
6. append exactly one payload-free terminal history record.

The timeout covers tool execution, not time waiting for capacity. A queued call
can therefore wait longer than `timeout_seconds`; this avoids timing out work
that has not begun and preserves the parameter's documented execution meaning.

Terminal history statuses are:

- `success` for a completed execution;
- `cached` for a cache hit;
- `timeout` for `asyncio.TimeoutError`;
- `parse_error` for invalid serialized arguments;
- `error` for validation or ordinary execution failure;
- `cancelled` for `asyncio.CancelledError`.

TASK-492 replaced the old mutable `started` record with one bounded,
metadata-only terminal record per call. This task preserves that privacy and
history contract; it does not reintroduce raw arguments/results or an
indefinitely `started` record.

Cancellation handling surrounds every awaited stage after call metadata is
established, including cache lock waits, limiter waits, tool execution, and
cache writes. A call whose coroutine has begun records `cancelled` and duration,
then re-raises. It does not return a normal error dictionary. The semaphore
context releases capacity during cancellation.

## Batch and Reload Contracts

`execute_tool_calls()` creates an ordered list of explicit
`asyncio.Task` objects and awaits them with normal `asyncio.gather()`.
`execute_tool_call()` already converts ordinary failures into dictionaries, so
successful and failed results remain isolated and returned in request order.

The batch owns every child it creates. In `finally`, it cancels each unfinished
child and drains the complete child set with
`asyncio.gather(..., return_exceptions=True)`. Therefore parent cancellation, a
child `CancelledError`, or another unexpected child control-flow exception
cannot leave siblings running or produce "Task exception was never retrieved"
warnings. After cleanup, the original cancellation or unexpected exception
propagates. The cleanup gather is only for draining; its collected values are
never returned as tool results.

`reload_tool_executor()` discards the global executor reference and constructs
the configured replacement. It has no shutdown call or destructor because
there is no owned thread pool. The caller remains responsible for not reloading
while calls on the retired executor are still active, matching current
behavior.

## Cross-thread Submission Ownership

`MCPToolProvider._execute()` remains synchronous and bounded. It creates the
`execute_hub_tool()` coroutine into a named local before submission. Ownership
transfers to the main event loop only when
`asyncio.run_coroutine_threadsafe()` returns a future. If submission raises
before that point, `_execute()` closes the still-local coroutine before
returning its contained error result.

Once a future exists, the bridge retains its existing best-effort
`future.cancel()` behavior on outer timeout or bridge failure; the event loop
then owns coroutine cancellation and result retrieval. The provider must not
close a successfully submitted coroutine from the worker thread. The
closed-loop regression runs with the unawaited-coroutine warning promoted to an
error.

## Verification

The superseded executor design would have required focused regression coverage
for:

- invalid `max_workers` and `timeout_seconds`;
- peak active tools at or below `max_workers`;
- queued calls beginning after capacity is released;
- request-ordered results despite out-of-order completion;
- ordinary failure isolation;
- unexpected child failure cancels and drains unfinished siblings;
- timeout history and subsequent-capacity release;
- cancellation while queued and while executing, with terminal history and
  propagated `CancelledError`;
- parent batch cancellation records cancellation for begun children, drains the
  complete batch, and leaves no live child tasks or un-retrieved exceptions;
- cache hits not consuming execution capacity;
- terminal history remains bounded, payload-free, and one-record-per-started
  call;
- closed-loop MCP submission closes the unsubmitted coroutine and passes with
  unawaited-coroutine warnings treated as errors;
- global reload without an executor shutdown attribute.

Current verification must instead prove the retired symbols remain absent,
the live tool provider remains functional, and the MCP bridge closes a
pre-transfer coroutine without producing an unawaited-coroutine warning.

## Re-review Record

The corrected contract was checked against `tool_executor.py`, the TASK-492
history/privacy regressions, and the existing tool and agent-call sites. The
second pass resolved:

- the unused thread pool being mistaken for a concurrency boundary;
- the obsolete mutable `started` history shape, which conflicts with
  TASK-492's bounded terminal-only metadata history;
- cancellation gaps around cache, limiter, execution, and cache-write awaits;
- `return_exceptions=True` treating cancellation as result data;
- plain `gather()` propagating without guaranteeing sibling cancellation and
  drain; and
- the MCP worker bridge leaking a coroutine when cross-thread submission fails
  before ownership transfer.

Had System A remained live, the implementation would have needed one semaphore
and explicit task cleanup. TASK-545's verified zero-caller result superseded
that implementation choice with deletion.
