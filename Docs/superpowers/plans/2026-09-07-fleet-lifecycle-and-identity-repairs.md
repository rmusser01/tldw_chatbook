# Fleet lifecycle and identity repairs implementation plan

**Goal:** Complete TASK-15665 and TASK-18312 from the orchestration review.

**Architecture:** Preserve the existing gateway/client ownership and resolution
ladder. Close each pool from its lifeline, and remember only bounded terminal
identity metadata after active handles are pruned. The two repairs are independent.

**Tech stack:** Python, asyncio/threading, httpx, SQLite, pytest.

**Spec:** ADR-130 (transport teardown) and ADR-129 (bounded pruned identities).

ADR required: yes
ADR path: backlog/decisions/130-model-call-lifeline-client-teardown.md
Reason: Make the gateway/lifeline resource teardown contract explicit.
TASK-18312 directly implements the amendment to ADR-129; no additional ADR.

## Constraints

- Preserve sibling transport isolation, injected-client ownership, bounded
  shutdown waits, and the existing continuation/tool authority boundaries.
- Preserve unrelated dirty files and the first repair pass. No full test suite.
- Write and observe each regression fail before production edits.
- No commit or integration operation is requested.

## TASK-15665 — loop-owned pool cleanup

Files: `Chat/console_provider_gateway.py`, `Chat/console_agent_bridge.py`,
new `Tests/Chat/test_fleet_client_lifecycle.py` under the existing source/test roots.

- [x] Exercise the real bridge and owned HTTP clients with a child held in a
  provider call after its parent returns. App teardown must leave the child's
  client open; releasing the child must close that client and remove its cache
  entry before the lifeline closes. The primary pool must also close at turn end.
- [x] Cover injected clients, repeated shutdown, thread-start failure,
  cleanup exceptions, and delayed cleanup after the bounded join.
- [x] Run the new tests against the current implementation and confirm the
  missing pool close is the failure.
- [x] Add `async aclose_current_loop() -> None`: under `_client_lock`, remove
  only `asyncio.get_running_loop()`'s entry, then await that client's `aclose`.
  Return immediately for injected clients; do not sweep another loop.
- [x] Bind that optional async cleanup in both `_ModelCallLifeline` constructors.
  Move final teardown to the driver thread: cancel/gather pending tasks, await
  cleanup, then `loop.close()` in `finally`. The caller requests stop and joins
  with the existing five-second ceiling; an unfinished cleanup owns its loop.
- [x] Run the new lifecycle, gateway, bridge, and fleet teardown suites.

## TASK-18312 — accurate pruned-ID refusal

Files: `Agents/fleet_coordinator.py`, `Agents/agent_service.py`,
new `Tests/Agents/test_fleet_pruned_identity.py`.

- [x] Reproduce a cancelled child's handle refusal after a later turn prunes
  it; check that both handle and run-ID copy acknowledge the terminal child.
- [x] Test bounded oldest-first identity eviction and conversation isolation;
  live/retained resolution must continue to win.
- [x] Store frozen `(handle_id, run_id, status)` records during terminal prune,
  capped at 256. Expose `get_pruned_identity(target_id)` with exact handle-ID
  lookup before run-ID lookup. Store no payload fields.
- [x] After the existing unpruned terminal tier, use that record for the same
  not-retained refusal. Generalize DB-only copy to avoid inventing an earlier
  session; qualify unknown old handles with expiry and a run-ID fallback.
- [x] Run the new tests, continuation resolution-order pins, mailbox/runtime,
  and the first-pass reliability regression module.

## Completion

- [x] Self-review races and cleanup/error paths; run scoped lint, formatting,
  whitespace, and diagnostic inventory checks.
- [x] Update user guide/review ledger, record exact tests and limitations in
  both Backlog tasks, check criteria, and mark verified tasks Done through CLI.

## Outcome

Both tasks are implemented and marked Done with checked criteria and detailed
notes. The separate targeted runs passed 471 + 195 + 44 = 710 tests. Two
localhost fixtures initially hit sandbox bind restrictions and passed after
permission to run their temporary local server. No full suite or live provider
was invoked. New tests pass lint/format, production lint counts add no findings,
scoped whitespace checks pass, and the reviewed diagnostic inventory passes.

Self-review refined the teardown ordering: cleanup must run before the loop
first stops, eliminating a race with the app's idle-client sweep. A failing
regression reproduced the initial open-pool interval and now passes. Delayed
cleanup retains loop ownership after a bounded join; forced process exit and
permanently blocked loop work remain outside deterministic Python cleanup.

The identity record has a fixed 256-entry limit. Its expiry changes diagnostics
only; durable run IDs still resolve through the existing scoped database tier,
and no payload is retained or continuation enabled by these records.
