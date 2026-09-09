# Fleet admission and wake budgets implementation plan

Delivery/recovery supplement: [ADR-135](../../../backlog/decisions/135-fleet-completion-delivery-and-crash-recovery.md)
and [the delivery plan](2026-09-08-fleet-delivery-and-recovery.md) now settle
TASK-32020/32021. Their per-conversation scheduling and durable attempt contract
replace this plan's earlier drain/global-serialization assumptions.

For agentic workers: use the executing-plans skill to implement these Backlog
tasks inline, with the named behavioral checks before marking each task done.

Goal: Bound actual local fleet occupancy and automatic follow-up work while
preserving manual priority, completed results, and current approval authority.

Architecture: One app-owned runtime capacity object tracks execution owners
and their still-running operations. AgentRunsDB owns causal chains and atomic
budget reservations. Trusted submit origin connects the controller, bridge,
service, wake coordinator, and provider-call seams. UI consumes metadata only.

Tech stack: Python 3.11+, threading/asyncio, existing SQLite and Textual; no new
dependencies.

Spec: `backlog/decisions/134-fleet-admission-and-automatic-work-budgets.md`
ADR required: yes
ADR path: `backlog/decisions/134-fleet-admission-and-automatic-work-budgets.md`
Reason: Shared admission ownership, durable budget policy, and execution limits.

## Global constraints

- Use the exact defaults, origin rules, reset policy, uncertainty handling,
  and local-versus-remote boundary in ADR-134.
- Keep the current working-tree repairs and unrelated Console extraction.
- No whole-suite runs without user opt-in; no provider calls in automated probes.
- Never infer origin from a prompt, result, tool argument, or viewed session.
- Acquire before work starts; release only after the owned work actually ends.
- Preserve existing approval/risk/tool-catalog gates and durable billing counts.
- Recheck schema/ADR allocation against current refs/worktrees at implementation.
- Update effective configuration and user documentation only as each slice ships.
- Each task starts with its Backlog record moved In Progress and its own plan.
  The unchecked tasks below are implementation work, not completed fixes.

## TASK-32034: Owned operations and bounded tool workers

Files: create `tldw_chatbook/Agents/execution_capacity.py` and
`Tests/Agents/test_execution_capacity.py`; modify `Agents/agent_service.py`,
`Chat/console_agent_bridge.py`, the trusted Console submit-context seam, and
`Tests/Chat/test_fleet_client_lifecycle.py`. All paths are under the repository.

The module owns `WorkOrigin` (MANUAL/AUTOMATIC), `RuntimeCapacity`, and an
`ExecutionOwner` per run. The owner's identity is assigned before dispatch and
bound to the DB run ID when available. Its root claim and owned operations must
all settle before its child reservation can be released. A primary owner needs
no child slot but retains its tool operations in the same runtime ledger.

```python
# Public behavior exercised by the first failing test.
owner = capacity.begin_execution(origin=WorkOrigin.AUTOMATIC,
                                 conversation_id="c")
owner.bind_run("run-1")
operation = owner.reserve_tool()
operation.mark_stopping()
owner.finish_root()
assert capacity.snapshot().stopping_tool_workers == 1
with pytest.raises(CapacityRefused):
    owner.reserve_tool()
operation.finish()  # called from the worker's actual finally
assert capacity.snapshot().tool_workers == 0
```

- [x] Add barrier tests for that lifecycle, eight occupied workers, automatic
  refusal at six, manual admission into the two reserved slots, duplicate
  release, failed thread start, and admission racing actual completion.
- [x] Wire the shared runtime owner through service construction. Capture the
  trusted origin at submission and pass it through immutable turn context or a
  named execution-context argument. All children inherit that captured origin.
- [x] Wrap `_call_with_timeout` worker creation, including start failure, with
  operation ownership. A timeout/cancel marks stopping; only worker `finally`
  frees it. The zero-timeout inline invocation also reserves/relinquishes a
  tool slot so configuration cannot bypass capacity.
- [x] Register model lifeline ownership before driver start. Release at the
  actual cleanup/driver completion seam even if `shutdown()` returns early.
  Preserve the caller-owned client distinction from ADR-130.
- [x] Run the new owner suite, fleet runtime, tool timeout, and lifeline suites.
  Replace the timed-out-worker baseline with enforcement assertions; retain the
  pre-change JSON as historical evidence. Verify existing tool refusals reach
  the Console without logging arguments or adding billing entries.

Implementation notes (2026-09-08): completed under ADR-134. Trusted origin uses
named dispatch arguments. Inline children use a separate model scope from fleet
settlement counters. Actual owners remain in a body-free runtime snapshot until
all operations finish. Capacity reads current settings before tool admission.
New production tests are in `Tests/Agents/test_tool_worker_capacity.py` and
`Tests/Chat/test_fleet_execution_ownership.py`; existing lifecycle tests also run.

## TASK-32035: Runtime-wide child admission

Files: extend `Agents/execution_capacity.py`; modify `Agents/agent_service.py`
and `Chat/console_agent_bridge.py`; add `Tests/Chat/test_fleet_runtime_admission.py`.

Child admission consumes the TASK-32034 owner contract. A successful child
reservation creates an ExecutionOwner whose root completion can remain pending
behind owned operations. The bridge must inject the same RuntimeCapacity into
successive services and retain it through retired-service cleanup.

```python
automatic = [capacity.begin_execution(origin=WorkOrigin.AUTOMATIC,
                                     conversation_id=str(i), child=True)
             for i in range(4)]
assert capacity.begin_execution(origin=WorkOrigin.AUTOMATIC,
                                conversation_id="next", child=True).refused
manual = capacity.begin_execution(origin=WorkOrigin.MANUAL,
                                  conversation_id="manual", child=True)
assert not manual.refused
# Marking a DB row cancelled cannot invoke manual.finish_root().
```

- [x] Reproduce seven simultaneous admissions across different conversations;
  assert only six launch, with no row/thread or consumed spawn slot on refusal.
- [x] Place runtime admission before handle reservation. Unwind the runtime
  lease when conversation admission or thread start fails; never acquire by
  summing unlocked per-conversation snapshots.
- [x] Route normal spawn, finished-child continuation, inline/skill children,
  and survivor cleanup through this same owner. Do not release on handle prune
  or forced terminal status. Keep provider/lifeline cleanup under its owner.
- [x] Test same-conversation and cross-conversation races, lowering limits,
  manual reserve, retries after actual release, and closed/reopened sessions.
- [x] Verify fleet/continuation/coordinator and bridge admission suites. Update
  only the affected baseline expected counts, settings, and user guide.

Implementation notes (2026-09-08): completed under ADR-134. Six child owners,
two manual reserves, and current settings apply across all dispatch paths.
Child roots finish at the fleet worker's actual final cleanup; owned operations
can retain the slot beyond it. A typed pre-dispatch refusal also corrects the
pure loop's previously unrefunded unnamed-spawn counter. Existing tool-result
serialization and actual failed-child accounting remain unchanged. Verified
451 regression tests and 93 admission/continuation/runtime/boundary tests.

## TASK-32036: Durable chain identity and reservations

Files: modify `DB/AgentRuns_DB.py` and add its next migration; create
`Agents/automatic_work_budget.py` for typed limits/results and
`Tests/DB/test_automatic_work_budget.py`. Current schema is v13; verify that the
next version is still v14 before allocating it.

Use a chain table keyed by an opaque ID and a unique root submission ID, storing
conversation scope, immutable limit snapshot, first automatic start/deadline,
status, and metadata-only pause reason. Add `agent_runs.work_chain_id`. A
reservation table has a unique ID, chain FK, kind, amount, state, nullable
actual usage, and timestamps. Generation/launch/call reservations are counted
before dispatch; token reservations cover the prepared input and bounded output.

The implemented API lives on `db.automatic_work`, an `AutomaticWorkLedger`
sharing AgentRunsDB's thread-local connections. Its `create_chain`, `attach_run`,
`reserve`, `commit`, `release`, `settle`, and `snapshot` methods keep the durable
contract together without further expanding the run-history class. Reservation
mutations require the same `owner_id`; only the first successful commit returns
`True` and grants dispatch authority. Snapshots expose immutable `used`,
`reserved`, and `available` mappings, uncertainty, and `pause_reason`.

```python
ledger = db.automatic_work
first = ledger.reserve(chain_id, reservation_id="wake-attempt-1",
                       owner_id="runtime-owner", kind="generation", amount=1)
again = ledger.reserve(chain_id, reservation_id="wake-attempt-1",
                       owner_id="runtime-owner", kind="generation", amount=1)
assert first == again
assert ledger.snapshot(chain_id).reserved["generation"] == 1
assert ledger.commit("wake-attempt-1", owner_id="runtime-owner")
assert not ledger.release("wake-attempt-1", owner_id="runtime-owner")
assert ledger.snapshot(chain_id).used["generation"] == 1
```

- [x] Write real SQLite transaction/reopen tests before migration and APIs.
  Cover unknown/legacy lineage, cross-conversation parents, duplicate IDs with
  conflicting amounts, rollback, contention at the final slot, and late results.
- [x] Implement transactionally bounded reservations and idempotent commit,
  release, and usage settlement. Refuse unknown chains; never create one as a
  fallback from a result's supplied identity.
- [x] Attach a chain when explicit manual work is accepted. Preserve it on old
  survivors and all automatic descendants; a new manual submission cannot
  reparent them. Recovery retains unresolved charges and uncertainty.
- [x] Verify migration/reopen and budget accounting suites. No reservation
  amount may enter the existing provider billing or saved run-budget counter.

Implementation notes (2026-09-08): schema v14 and its standalone v13 upgrade
preserve legacy history without allocating automatic allowance. FULL-synchronized
ledger transactions cover reservations and the complete wake-attempt state
machine. Recovery is explicit; opening a handle does not invalidate live owners.
The original deadline and process-tagged monotonic anchor survive handle reopen;
a replacement process preserves elapsed time when establishing its new anchor.
Unknown usage retains the estimate and requires review. Token overages block
admission of every resource, without changing provider billing.

Accepted manual turns establish chain identity off the event loop using the
durable conversation ID. Bridge/service children inherit it; plain wakes carry
it in private stream context. The coordinator separates pending chains and
accepts only its exact live authorization token. Actual reservation/attempt
admission at the runtime dispatch seams is implemented by TASK-32037 below.

## TASK-32037: Wake enforcement and visible pause

Files: modify `Chat/console_fleet_wake.py`, `Chat/console_chat_controller.py`,
`Chat/console_agent_bridge.py`, `Agents/agent_service.py`, the prepared provider
request seam, and `UI/Console_Modules/agent.py`. Add
`Tests/Chat/test_automatic_wake_budget.py` and
`Tests/UI/test_console_automatic_work_pause.py`.

Requires the preceding capacity/ledger contracts plus TASK-32020/32021's
completed delivery/fairness and crash-policy decisions. Use the typed refusal
reason and snapshots from the budget/owner modules; no second budget ledger in UI.

```python
from Tests.Chat.test_console_fleet_wake import (
    _controller_rig, _terminal_subagent_run, _drain, _survivor, _settle,
)

async def test_fourth_wake_preserves_results(tmp_path):
    chacha, app, db, store, session, gateway, bridge, controller = _controller_rig(tmp_path)
    wake = controller.fleet_wake
    chain_id = db.automatic_work.create_chain(session.id, root_submission_id="manual-1")
    try:
        for index in range(4):
            parent, child = _terminal_subagent_run(db, session.id)
            db.automatic_work.attach_run(parent, chain_id)
            db.automatic_work.attach_run(child, chain_id)
            wake.on_fleet_drained(_drain(session.id, _survivor(child, session_id=session.id)))
            if index < 3:
                assert await _settle(lambda: not wake.has_pending(session.id))
        assert await _settle(
            lambda: db.automatic_work.snapshot(chain_id).pause_reason == "generation_budget"
        )
        assert len(gateway.payloads) == 3
        assert wake.has_pending(session.id)
        assert not db.get_run(child)["wake_delivered_at"]
        assert wake._retry_timer is None
    finally:
        controller._disposed = True
        db.close()
        chacha.close()
```

- [ ] Build the real-loop harness used by that test, then test calls/tokens/time
  exhausting before generations. Include both agent and plain-provider paths.
- [ ] Reserve at scheduling and commit at actual acceptance. Select one causal
  chain per wake batch; preserve other chains' pending results. User draft/queue
  claims and the one-reserved-primary-slot rule run before automatic dispatch.
- [ ] Reserve prepared-request input plus capped output before every automatic
  model call, sharing the chain remainder across children. Settle known usage;
  retain uncertain reservations and block further work when estimates overrun.
- [ ] Expose capacity/stopping/paused state through the existing fleet surface.
  Prove painted copy and actual manual recovery/cancel actions without changing
  approval grants or auto-clearing undelivered results.
- [ ] Exercise every crash window from TASK-32021, mixed old/new chains, changing
  limits, clock anomalies, restart with legacy rows, and delayed cleanup.
- [ ] Update effective settings and guide, run targeted regression/lint/format/
  diagnostic checks, self-review, and complete the task's AC and notes via CLI.

TASK-32034 through TASK-32037 are implemented. The accepted limits are enforced
at runtime. Historical baseline measurements are retained alongside the current
admission and dispatch regressions.


Implementation notes (TASK-32037, 2026-09-08): runtime acceptance and physical
provider/helper fences enforce the shared allowance across plain turns, agent
turns, and surviving children. Individual results use a fixed 250 ms window;
two automatic conversations retain a manual primary reserve and rotate fairly.
Startup recovery discovers run history without badge authority. The fleet's
wrapped pause notice keeps Run history and explicit manual continuation visible.

Review-driven adjustments: schema v15 records the native runtime owner so stale
completed-parent callbacks cannot resume after replacement; executor/client waits
recheck authority before dispatch; preparation cancellation releases Console
occupancy. The inspector gained an optional wrapped notice after rendered tests
showed that status rows were clipped. These implement existing ADR-134/135
contracts; no new ADR was required. Final test counts and limitations are in the
Backlog task and orchestration review ledger.
