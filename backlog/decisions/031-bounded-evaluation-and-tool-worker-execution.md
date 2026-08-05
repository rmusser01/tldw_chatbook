# ADR 031: Bounded Evaluation Execution and Retired Tool Worker Contract

Status: Accepted
Date: 2026-07-24
Amended: 2026-07-24
Reconciled: 2026-07-27
Related Tasks:
[TASK-902](../tasks/task-902%20-%20Enforce-evaluation-execution-and-run-state-contracts.md),
[TASK-545](../tasks/task-545%20-%20Wire-built-in-tool-executor-into-MCP-permission-gate.md)
Supersedes: N/A

## Decision

Keep the synchronous `chat_api_call()` dispatcher as the production provider
boundary and adapt it at the evaluation boundary with a worker thread,
per-attempt timeout, and awaitable-result compatibility. Bound concurrent
evaluation samples with an `asyncio` semaphore in their owning runtime.
Preserve input order in returned batches while reporting progress as work
settles. Treat sample errors as a failed evaluation run while retaining stored
results. Persist evaluation cancellation as `cancelled` and always re-raise
`asyncio.CancelledError`.

The existing evaluation status vocabulary remains unchanged. A run containing
one or more failed samples uses `failed`, not a new `completed_with_errors`
schema value.

The `ToolExecutor` concurrency, callback, cancellation, history, reload, and
batch-cleanup design recorded below is historical and non-operative. Current
`dev` subsequently proved that executor had zero production execution callers
and retired System A under TASK-545. This ADR must not be used to restore
`ToolExecutor`, `ToolResultCache`, `get_tool_executor()`, or
`reload_tool_executor()`. The only still-live adjacent tool correction is the
MCP bridge's ownership of a coroutine until cross-thread submission succeeds.

## Context

The production chat dispatcher is synchronous, but
`BaseEvalRunner._call_llm()` awaited it directly. A synchronous reproduction
failed with `TypeError: object str can't be used in 'await' expression`.
Existing tests patched the dispatcher with async mocks and therefore encoded a
different contract from production.

`EvalRunner` exposed `max_concurrent_requests`, `request_timeout`, and
`retry_attempts`, but evaluated samples sequentially and did not apply the
timeout to provider attempts. A two-sample reproduction with one failed sample
stored both results and marked the run `completed` with no error message.
Cancelling an orchestrated run removed it from the in-process concurrent-run
registry but left its durable database status `running`.

The retired `ToolExecutor` constructed a
`ThreadPoolExecutor(max_workers=...)` that no tool call used. Historical
reproductions established that its advertised callbacks, public cancellation,
and batch cleanup did not satisfy their contracts. Those findings justified
the corrected design recorded below, but TASK-545's later zero-caller analysis
made deletion safer than adding lifecycle machinery to a dead executor.

The adjacent, live synchronous MCP tool bridge also leaked an unsubmitted
coroutine when `run_coroutine_threadsafe()` rejected a closed target loop.

These are runtime and cross-module service contracts. They need one explicit
decision before the larger application-state decomposition changes ownership
or worker topology.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Rewrite all provider handlers and `chat_api_call()` as native async APIs | This is a broad provider migration with substantially larger compatibility risk. The evaluation boundary can adapt the existing synchronous contract. |
| Continue awaiting the dispatcher and make tests use only async mocks | This preserves a test-only contract that production does not satisfy. |
| Keep evaluation sequential and treat concurrency settings as advisory | The public configuration would remain misleading and large evaluations would unnecessarily serialize independent samples. |
| Add `completed_with_errors` to the database status constraint | It requires a schema migration and downstream UI/status changes. Existing `failed` already expresses an unsuccessful run while stored results remain inspectable. |
| Execute async tools through the existing `ThreadPoolExecutor` | Tool implementations already return awaitables and belong on the owning event loop; a thread pool does not bound them unless another event loop is introduced per worker. |
| Let batch gathering contain cancellation like an ordinary tool failure | Cancellation is control flow. Swallowing it leaves callers unable to stop the owning run reliably. |
| Use subprocesses to force-kill timed-out provider calls | Process isolation can enforce a hard stop but adds serialization, credential, platform, and cleanup contracts outside this repair tranche. |

## Consequences

Evaluation construction rejects non-positive concurrency and timeout values,
negative retry values, and non-finite numeric bounds. `retry_attempts` from the
model execution configuration takes precedence over the task metadata
`max_retries`; the task value remains the fallback for compatibility.
`retry_delay` follows the same model-then-task fallback. Basic runners retain
their existing retry wrapper; the specialized-runner `LLMInterface` applies the
same normalized retry policy because those runners currently call it directly.
Each provider attempt, including an awaitable returned by a test double, is
covered by the configured timeout.

The provider call runs through `asyncio.to_thread()`, so it no longer blocks the
application event loop. Cancelling or timing out the awaiting coroutine cannot
kill Python code already running in that thread. Chatbook stops awaiting the
late result and may start a configured retry; the underlying handler can
continue until it returns, so a timeout can produce overlapping or duplicate
billed provider work. Hard cancellation requires the rejected subprocess or
native-async provider redesign and must not be inferred from this contract.

Evaluation sample tasks acquire one semaphore. Results are written into slots
by input index and returned in dataset order. Progress callbacks run centrally
once per durably stored sample, in completion order, so callback or persistence
failures fail the run instead of being converted into duplicate sample errors.
The public progress contract is
`(completed, total, result)` and accepts either a synchronous callback or one
returning an awaitable; an awaitable is always awaited. A separate
`run_started_callback(run_id)` uses the same sync-or-awaitable rule and exposes
the durable run identity after active-task registration but before sample work.
Callbacks run on the owning event loop; application UI adapters use same-loop
updates rather than thread-only marshalling.
When the caller cancels or a central callback fails, outstanding sample tasks
are cancelled and drained.

An orchestrated run is `completed` only when all returned samples are
error-free and every result was stored. Sample errors and storage mismatches
produce `failed` with a count-based summary while retaining results and
aggregate metrics. Caller cancellation attempts to persist `cancelled` without
masking the original cancellation and always unregisters the run. The public
`cancel_evaluation(run_id)` operation targets the registered owning task,
cancels and drains it, and returns only after normal cancellation cleanup. It
does not remove the task or write terminal state independently of the owning
run. Asynchronous orchestrator shutdown uses the same cancel-and-drain path
before closing the database; synchronous close refuses to close while a run is
active.

The historical ToolExecutor proposal would have used one positive
`max_workers` semaphore around actual execution, treated cancellation as
control flow, and explicitly cancelled and drained every batch child. It is
retained as evidence of the reviewed contract, not as an implementation plan.
Because System A is retired, there is no public ToolExecutor cancellation,
callback, cache, history, batch, or reload contract in the live architecture.

Cross-thread MCP submission does remain live. It keeps ownership of the
coroutine until `run_coroutine_threadsafe()` succeeds and closes the coroutine
if submission rejects before returning a future.

## Links

- [Evaluation execution design](../../Docs/superpowers/specs/2026-07-24-evaluation-execution-contracts-design.md)
- [Superseded tool worker design](../../Docs/superpowers/specs/2026-07-24-tool-worker-contracts-design.md)
- [System A retirement design](../../Docs/superpowers/specs/2026-07-26-retire-system-a-design.md)
- [System A retirement plan](../../Docs/superpowers/plans/2026-07-26-retire-system-a.md)
- [TASK-332: platform resource-limit disclosure](../tasks/task-332%20-%20Eval-runner-resource-limits-robust-on-macOS.md)
