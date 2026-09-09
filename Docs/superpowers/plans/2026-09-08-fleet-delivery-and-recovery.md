# Fleet completion delivery and crash recovery

Design tasks: TASK-32020 and TASK-32021.
Implementation tasks: TASK-32036 and TASK-32037.

ADR required: yes
ADR path: backlog/decisions/135-fleet-completion-delivery-and-crash-recovery.md
Reason: Changes cross-module completion timing, scheduling, and durable execution
acknowledgement. Amends ADR-129/134 and supplements the existing
[budget plan](2026-09-08-fleet-admission-and-wake-budgets.md).

## Design and evidence

- [x] Trace individual settlement, final drain, submission acceptance, provider
  dispatch, completed-turn stamping, and mark-based reconstruction.
- [x] Run deterministic two-child, real approval, and failed-stamp probes;
  record measurements and limitations in the review ledger.
- [x] Choose individual notifications with a fixed 250 ms batching window,
  two automatic primary slots subject to one manual reserve, and round-robin
  eligible conversations. Preserve final-drain usage reconciliation.
- [x] Specify attempt claims and a required durable acceptance fence, keeping
  uncertain attempts charged and paused across recovery.
- [x] Correct existing exactly-once wording; review the crash matrix against
  both controller dispatch paths and update implementation task criteria.

## TASK-32036: durable state before runtime enforcement

1. Extend the next AgentRunsDB migration with immutable chains, reservations,
   wake attempts, and unique source-run claims. Follow ADR-135's metadata-only
   attempt fields and state transitions. Explicit startup recovery is separate
   from database construction or handle reopen.
   Use FULL-synchronized transactions for automatic-work mutations; the existing
   NORMAL per-step connection policy cannot be the durable execution fence.
2. Add typed atomic claim/accept/abort/complete/recover APIs. A duplicate token
   identifies the same operation; it must not grant a second dispatch. Validate
   conversation, lineage, owner, and terminal status in the transaction.
3. Bind accepted explicit submissions to immutable chains; pass identity as
   trusted context through bridge/service/children and plain-provider turns.
   Keep old survivors on their old chain.
4. Use real SQLite tests for schema upgrades, rollback at each transition,
   separate-connection contention, reopen without recovery, explicit startup
   recovery, duplicate callbacks, and cross-conversation or mixed-chain refusal.
   Run the existing AgentRunsDB migration tests after the schema bump.

Implementation notes (2026-09-08): TASK-32036 supplies the v14 schema and
`db.automatic_work` ledger (`claim_wake`, `accept_wake`, `abort_wake`,
`complete_wake`, `recover`). Mutations use the run DB's connections under FULL
synchronization. Owners, exact run claims, immutable limits, and process-tagged
clock anchors are metadata only. Only the first successful acceptance grants
dispatch authority; runtime integration is the following task.

## TASK-32037: enforce on actual dispatch paths

1. Write failing real bridge/controller tests: a ready child wakes while its
   sibling remains blocked; the final usage fold still waits for both; a second
   conversation can wake while the first waits for real approval, with no
   approval decision or manual-slot theft.
2. Add a typed individual-settlement fanout and wire wake intake separately from
   drain consumers. Use fixed coalescing deadlines and result-ID dedupe.
3. Replace singular delivery state with per-conversation attempt ownership and
   a total cap of two, bounded by primary occupancy minus the manual reserve.
   Update authorization checks and UI/session lookups together; do not retain
   a singular accessor that silently reports an arbitrary active attempt.
4. Integrate the required acceptance fence before the first billable helper,
   main provider generation, or tool work. Reserve exact prepared input plus
   resolved output allowance at each provider boundary. Cover agent/plain
   paths, childless wakes, and helper generations. A best-effort UI hook cannot
   authorize dispatch or determine refund.
5. Complete/abort by typed outcome, never a generic exception or missing return.
   Make mark projection repairable from the durable ledger, and preserve the
   exact batch when new completions arrive during bookkeeping.
6. Add body-free status and explicit manual recovery to the existing extracted
   fleet module. Keep screen growth within the existing architecture contract.
7. Exercise every row of ADR-135's crash matrix, finite multi-generation budgets,
   fair ready-candidate selection, configuration reductions, and cancellation.
   Re-run relevant fleet, admission, approval, runtime/headless, usage-fold,
   database, and UI tests. Full-suite verification still requires user opt-in.

## Scope and review

No tool authority or billing changes. No persistent steering mailbox or peer
messaging is included. Default budgets remain ADR-134's accepted values. Record
implementation evidence separately from the design baseline; do not describe
future delivery/budget/recovery behavior as active until its actual paths pass.


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
