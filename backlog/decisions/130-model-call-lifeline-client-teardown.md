# ADR-130: Model-call lifeline client teardown

Status: Accepted
Date: 2026-09-07
Related task: TASK-15665
Related review: [Agent orchestration review](../docs/agent-orchestration-review-2026-09-07.md)

## Decision

The model-call lifeline owns the teardown of its event loop and the gateway's
HTTP client for that loop. The gateway exposes `aclose_current_loop()` to remove
and close only the caller loop's owned client. Injected clients remain owned by
their injector. App-level `aclose()` continues to spare other running loops.

Both the primary turn and each fleet child bind this cleanup when constructing
their lifeline. When the owner stops submission and shuts the lifeline down,
the driver thread stops normal dispatch, cancels and drains any remaining loop
tasks, closes its HTTP pool on that same loop, and closes the loop. A bounded
join keeps a stuck cleanup from freezing the Console; a still-running cleanup
keeps its loop alive and closes it itself when it eventually returns. Cleanup
failure must not replace the agent's result or stop the final loop close.

The loop remains running through asynchronous cleanup and stops only after
the pool-close attempt finishes. Stopping and then restarting the loop for
cleanup creates an idle interval in which app-level teardown could detach and
schedule the same pool for a second close.

No cleanup callback is called for a lifeline whose thread failed to start; no
request or loop client could have been created there. Shutdown stays safe when
repeated. Gateway doubles that own no per-loop resources may omit the hook.

## Alternatives and consequences

- Closing a child's pool from app teardown interrupts an active request and
  violates the existing live-child isolation contract.
- Waiting until a loop is already closed is too late to close loop-bound
  transports reliably. Garbage collection is not deliberate pool cleanup.
- A watcher thread or polling reaper duplicates the lifeline's existing owner
  and introduces another shutdown race. Teardown belongs to the loop driver.

This fixes normal and cooperative teardown, including children that outlive
their supervisor. It cannot force Python cleanup after process termination or
while an event-loop callback is permanently blocked. It adds no dependency,
schema, global cleanup sweep, or broader cancellation authority.
