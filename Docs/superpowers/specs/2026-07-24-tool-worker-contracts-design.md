# Tool Worker Contracts Design

Date: 2026-07-24
Status: Design approved; written specification review pending
ADR:
[ADR-024](../../../backlog/decisions/024-bounded-evaluation-and-tool-worker-execution.md)
Backlog:
[TASK-497](../../../backlog/tasks/task-497%20-%20Enforce-ToolExecutor-concurrency-and-cancellation-contracts.md)

## Summary

Make `ToolExecutor.max_workers` a real async concurrency limit and remove the
unused thread pool. Tool timeouts remain contained error results. Cancellation
becomes explicit control flow: every cancelled call receives a terminal history
record and re-raises, and batch execution no longer swallows child
cancellation. Existing async tool, cache, result-order, and ordinary
error-isolation contracts remain intact.

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
- `reload_tool_executor()` shuts down the unused pool, coupling reload to
  infrastructure that never executed tools.

## Goals

- Validate execution limits at construction.
- Bound actual simultaneous `Tool.execute()` calls across the executor.
- Keep queue time separate from execution timeout.
- Preserve per-call timeout and ordinary error results.
- Record cancellation regardless of whether a call is queued or executing.
- Propagate cancellation through single and batch APIs.
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

1. append the existing `started` history record;
2. validate the tool and arguments;
3. return a valid cached result without acquiring execution capacity;
4. wait for the semaphore;
5. once admitted, apply `timeout_seconds` to `tool.execute()`;
6. record exactly one terminal status.

The timeout covers tool execution, not time waiting for capacity. A queued call
can therefore wait longer than `timeout_seconds`; this avoids timing out work
that has not begun and preserves the parameter's documented execution meaning.

Terminal history statuses are:

- `success` for a completed execution;
- `cached` for a cache hit;
- `timeout` for `asyncio.TimeoutError`;
- `error` for validation, parsing, or ordinary execution failure;
- `cancelled` for `asyncio.CancelledError`.

Cancellation handling surrounds every awaited stage after the history record,
including cache lock waits, limiter waits, and tool execution. It records
`cancelled` and `end_time`, then re-raises. It does not return a normal error
dictionary. The semaphore context releases capacity during cancellation.

## Batch and Reload Contracts

`execute_tool_calls()` creates the same ordered list of leaf coroutines and
uses normal `asyncio.gather()` without `return_exceptions=True`.
`execute_tool_call()` already converts ordinary failures into dictionaries, so
successful and failed results remain isolated and returned in request order.
Cancellation or another unexpected control-flow exception propagates from the
batch.

`reload_tool_executor()` discards the global executor reference and constructs
the configured replacement. It has no shutdown call or destructor because
there is no owned thread pool. The caller remains responsible for not reloading
while calls on the retired executor are still active, matching current
behavior.

## Verification

Focused regression coverage will include:

- invalid `max_workers` and `timeout_seconds`;
- peak active tools at or below `max_workers`;
- queued calls beginning after capacity is released;
- request-ordered results despite out-of-order completion;
- ordinary failure isolation;
- timeout history and subsequent-capacity release;
- cancellation while queued and while executing, with terminal history and
  propagated `CancelledError`;
- cache hits not consuming execution capacity;
- global reload without an executor shutdown attribute.

Focused tool tests, relevant chat/worker integration tests, Ruff, Python
compilation, and repository diff checks gate completion.
