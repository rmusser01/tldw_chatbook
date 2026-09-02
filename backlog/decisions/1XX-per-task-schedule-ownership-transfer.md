# ADR-1XX: Per-task schedule ownership transfer, local recurring_question execution, and results sync-down

Status: Proposed
Date: 2026-09-02
Related Task: [TASK-18940](../tasks/task-18940%20-%20Server-offloaded-scheduled-agent-tasks-execution-seam.md)
Amends: ADR-077 §1 ("execution follows ownership") — reframes it from a
screen-era, permanent split to a per-task, transferable one
Related: [spec-2026-08-31-schedules-handoff-parity.md](../docs/spec-2026-08-31-schedules-handoff-parity.md),
[plan-2026-09-02-schedules-handoff-pr5.md](../docs/plan-2026-09-02-schedules-handoff-pr5.md)

## Decision

A scheduled task's owner (`local` or `server:<user_id>`) is no longer fixed
for the task's lifetime — it is a per-task property the user can transfer in
either direction, on a task-by-task basis, while every other task keeps its
current owner untouched. This amends ADR-077 §1's framing ("a scheduled task
executes on exactly one side... `owner_id="server:<user_id>"` definitions are
... never dispatched by the client's SchedulerLoop") from a permanent,
account-wide split into a per-task one: single-owner execution still holds at
every instant (unchanged, and now machine-checked — see the invariant below),
but which side owns a given task can change.

1. **Transfer state machine** (spec §6). `transfer_state ∈ {NULL,
   to_server_pending, to_server_sent, to_server_failed, from_server_pending}`
   on both `reminder_tasks` and `automation_definitions`. Local→server:
   queue (still executes locally), disarm-then-send on the actual push
   attempt, convert to a server mirror on ack, `to_server_failed` re-arms
   locally with the server's rejection reason and is retryable in place
   (Task 6 fix round; `begin_transfer_to_server` on a failed row CASes it
   back to pending and replaces the queued mutation with a fresh, error-free
   payload). Server→local: an immediate **dormant** local copy is created
   (`from_server_pending`, excluded from every armable-row query) while the
   release (delete for a reminder, archive for a definition) is queued; the
   server keeps executing until the release acks, at which point the copy
   arms and the mirror is torn down (deleted outright, with its sync
   mapping, on the ack for a reminder; archived in place for a
   definition). The reminder mirror's teardown is deliberately NOT left
   to the next pull's full-set reconciliation: that scan only DELETES a
   row carrying a local tombstone, so a released mirror would instead
   become a standing "the server deleted this row" conflict beside the
   already-armed local copy (final review I4).

2. **§3's invariant, machine-checked.** *At most one side is armed for a
   given task at any instant.* This is not just prose: a `RuleBasedState
   Machine` property test
   ([Tests/Scheduling/test_transfer_invariant.py](../../Tests/Scheduling/test_transfer_invariant.py))
   drives randomized interleavings of begin/push/cancel/release/recover
   against the real `ScheduledTasksDB` + `SchedulingService`/`SyncEngine` (a
   stateful fake server client stands in for the network) and asserts the
   invariant after every step, scoped to the reminder primitive because that
   is where 100% of the shared CAS/DB transfer machinery
   (`set_transfer_state`/`clear_transfer_state`/`convert_row_to_server_mirror`/
   `create_local_copy_from_mirror`) is exercised identically for both
   primitives. `Tests/Scheduling/test_transfer_end_to_end.py` (this PR)
   complements it with directed, full-stack walkthroughs of both primitives,
   the definition preview→create network shape, and the schedule-vocabulary
   translation at each ownership boundary.

3. **Cancel is state-keyed, not mutation-keyed** (spec §6.3): unattempted or
   definitively-failed transfers (`to_server_pending`/`to_server_failed`)
   clear in place with nothing sent; `to_server_sent` (a push already in
   flight, or already acked) is too late to cancel and the UI offers a
   reverse transfer instead; an unpushed or failed release
   (`from_server_pending` on the dormant copy) deletes the copy with the
   server untouched; an already-acked release is likewise too late. Cancel
   keys off `row["transfer_state"]`, never off whether a pending mutation
   still exists — a release that fails server-side settles by clearing its
   own mutation while leaving the dormant copy's `from_server_pending` state
   in place, so mutation-absence cannot stand in for "nothing to cancel."

4. **Refusals are honest and per-task** (spec §6.4): no server connection or
   identity; a family the target side cannot execute (`agent_task` always
   refuses locally in v1; `recurring_question` refuses locally when
   `compute_local_health` is not `ready`, quoting the reason verbatim); a
   transfer already in progress on the row; a lifecycle outside
   `{configured, paused}` (nothing left to execute on `archived`/`solved`).
   An imminent or already-past one-time `run_at` **warns rather than
   refuses** — server behavior on a past `run_at` is unverified, and the
   transfer can outlive the moment. Local-only fields (a reminder's
   `timeout_seconds`) are stated as non-transferring in the confirm dialog,
   never silently dropped.

5. **Local `recurring_question` execution** (spec §7; shipped in handoff
   PR-2, #2295) makes local ownership a real, working destination rather
   than a paper one: the server's pure validators/classifiers are ported
   with a fixture-parity contract against the server repo, dispatch follows
   the `BriefingJobHandler` spawn shape (synchronous claim-check, the run
   itself an independent `asyncio.Task`) with an overlap guard, and
   execution reuses `RAG_Search/simplified` and `chat_api_call` with the
   same provider/model precedence the server's `resolve_execution_target`
   uses — the property a pinned model survives handoff in both directions
   depends on.

6. **Results sync-down** (spec §5.1/§10; shipped in handoff PR-3, #2297):
   server-run results pull into the local results store on the same
   bounded newest-pages walk the definitions pull uses (the server exposes
   no `updated_at` filter on `/results`), and local `review_state` changes
   push back through a pending-mutation replay that survives a same-cycle
   stale echo (the push and the same sync's own results pull racing to
   reflect it) without reverting the user's review.

## Context

ADR-077 fixed a real problem (double execution) with a framing that assumed
ownership was decided once, at the account or screen level, for the life of
a task. That framing was correct for "the server is now capable of
execution" but became a limitation once both sides could execute the same
families: a user with a mix of tasks they want offloaded and tasks they want
to keep running on their own machine has no way to express that per-task,
only by moving their entire account's default. This program (schedules-
handoff PR-1 through PR-5) makes ownership a property of the task instead —
requiring local execution parity for the families that can move (PR-2, so
"transfer to local" is not a dead end), sync of what runs where and what it
found (PR-3), authoring that targets either side up front (PR-4), and the
transfer machine itself, both directions, with the single-owner invariant
preserved not by policy but by a CAS-guarded state machine and a property
test that tries to break it.

## Consequences

- ADR-077 §1's "never dispatched by the client's SchedulerLoop" now means
  "never dispatched while `owner_id` says server" — still absolute at any
  instant, no longer a permanent property of the row. The owner filter
  ADR-077 introduced is unchanged; a transferred row is still filtered
  correctly by it because ownership itself is what the transfer mutates.
- The transfer mechanism is create-on-target + tombstone-on-source,
  eventually-atomic (spec §2 decision 5) — a crash between send and ack is
  recoverable (`recover_inflight_transfers`: hash-idempotent blind retry for
  definitions, list-and-match on `link_id` for reminders) but not
  instantaneously atomic. Zero server-side changes were required for v1.
- `agent_task` transfer/local-execution stays out of scope (ADR-077 §4's
  side-effect-free phase-1 boundary, unchanged); the transfer refusal gate
  is already family-aware so it drops in without rework.
- Definitions DO get transfer UI in this PR, as Automations-tab
  keybindings rather than the Queue tab's row adapter: `M` Move to server,
  `m` Move to local, `y` Retry, `k` Cancel, sharing the reminder flow's
  confirmation, warnings and toasts. What genuinely rides PR-6's row
  adapter is the unified Queue-tab presentation of both families (badges,
  owner column), not the transfer triggers themselves.
- Lifecycle (pause/resume/archive) has a service producer
  (`SchedulingService.set_definition_lifecycle`, which writes locally and
  queues the mutation `SyncEngine._push_definition_lifecycle` replays) but
  no UI caller yet: those affordances belong to the schedules redesign
  program. The seam is reachable and tested rather than dead code, and
  that is the whole of the claim.

## Links

- [Spec: Scheduled-task handoff parity](../docs/spec-2026-08-31-schedules-handoff-parity.md)
- [Plan: schedules-handoff PR-5](../docs/plan-2026-09-02-schedules-handoff-pr5.md)
- [ADR-077 — Server-offloaded scheduled agent tasks](077-server-offloaded-scheduled-agent-tasks.md) (amended here)
- [ADR-018 — Local/server hybrid scheduled-tasks storage and sync](018-local-server-hybrid-scheduled-tasks.md)
- [TASK-18940 — Server-offloaded scheduled agent tasks: execution seam and result pass-back](../tasks/task-18940%20-%20Server-offloaded-scheduled-agent-tasks-execution-seam.md)
- [Tests/Scheduling/test_transfer_invariant.py](../../Tests/Scheduling/test_transfer_invariant.py) — the §3 invariant property test
- [Tests/Scheduling/test_transfer_end_to_end.py](../../Tests/Scheduling/test_transfer_end_to_end.py) — directed full-stack transfer walkthroughs
