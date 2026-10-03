# ADR-143: Local chat schedules and agent timers

Status: Accepted (design, 2026-10-02)
Date: 2026-09-09
Revised: 2026-10-02 (user-authorized design-review corrections)
Task: [TASK-32195](../tasks/task-32195%20-%20Design-local-chat-schedules-and-agent-self-timers.md)
Design: [Local chat schedules and agent timers](../../Docs/superpowers/specs/2026-09-09-chat-schedules-and-agent-timers-design.md)
Amends: ADR-134/135 for explicitly authorized scheduled turns and shared recovery.
Preserves: ADR-018/019 scheduling ownership, ADR-069 project instructions, ADR-079 workspace defaults, ADR-082 private scratch, ADR-131 accounting, ADR-141 goal semantics, ADR-148 run hooks and ADR-150 design-token governance.

## Context

The user approved local scheduled responses in an existing Console chat, including
tools, and requested agent tool access for setting timers. The existing scheduler
already dispatches reminders, watchlists and briefings. The Console runtime
already survives navigation and owns native execution, approvals and finite
automatic accounting. Their missing connection is a durable, authorized future
chat turn, not another goal solver or an in-memory sleep loop.

Recurring work also changes accounting authority: a user may explicitly permit
ongoing daily execution, while an automatically executing model must not gain an
unlimited allowance by creating another schedule. Existing goal/fleet allowances
cannot silently be reset to implement this feature.

## Decision

1. Keep native chat schedule definitions, scheduling grants and occurrence
   records in AgentRunsDB beside the automatic ledger. Accept an occurrence and
   reserve its budget in one FULL-synchronized transaction. Project these records
   into the existing Scheduling queue and Schedules screen; do not duplicate
   definitions in ScheduledTasksDB or execute reminder bodies as prompts.
2. Add an app-owned scheduled-turn coordinator under ConsoleRuntime. The common
   scheduler hands off durable pending work and remains free to dispatch other
   jobs. Native controller/bridge execution, manual priority, physical ownership,
   permissions and exact-request accounting remain authoritative. Add an explicit
   scheduled submission origin; do not masquerade as a manual, fleet or goal turn.
3. Capture the target persisted conversation, provider/resources/tools and
   instructions on create/edit, but read current conversation history before
   acceptance. Revalidate immutable target identity and live authority after
   waits. Automatic work never consumes the user's composer or pending attachments.
   Persist reference-backed selections, never scratch locators, live-session IDs
   or secrets. Obtain current scratch and re-resolve narrowing persona/profile
   policy and deny-only hooks through the existing runtime seams per occurrence.
   Project accepted scheduled request/reply pairs into later provider history
   with machine provenance, without minting manual authority or prompt history.
4. Provide `/schedule` and descriptor-backed `schedule_create/get/list/update/cancel`
   agent tools over one service. Resolve caller identity from trusted run context;
   scope v1 to the native main agent's own conversation. Model arguments cannot
   choose another conversation, grant, engine or approval authority. Mutation
   permissions and exact-call idempotency apply at the actual dispatch boundary.
   Every mutation stores its request hash and result receipt in the same transaction;
   replay returns that result before revision CAS. Expired source identities fail
   closed rather than creating fresh work.
5. Human setup may explicitly authorize recurrence until paused, subject to
   finite per-occurrence and per-conversation daily limits. A human-origin agent
   turn may create a finite grant through normal tool permissions. Automatic
   successor timers inherit a grant and its remaining total allowance; they
   cannot create or replenish one. Other automatic origins without a grant may
   propose a schedule for human review, not activate fresh spending authority.
   Bind human grants to the reviewed root instructions. Automatic turns cannot
   replace that body, even with follow-up or stored tool permissions; exact-task
   human review can. Follow-up authorization permits descendant instructions
   within the existing allowance and resource/timing ceilings.
6. Count all descendant scheduled work against the same grant and conversation
   window. Existing billing is distinct from budget-token admission. Daily
   windows are an explicit part of standing human authorization, with durable
   reservations and rollback detection; restart, cancellation, new timer IDs and
   midnight settlement do not erase accepted or uncertain charges.
7. Use unique occurrence identities and one pending/active occurrence per
   schedule. Coalesce overdue work into one catch-up, preserve manual capacity,
   and never replay accepted work whose effect is uncertain. Definition revisions
   fence pending edits; instructions-only edits transfer unaccepted nominal work
   without a new allowance, timing edits supersede it, and accepted work is
   immutable. Early one-time Run-now consumes the sole timer; overdue one-time
   Resume requires explicit retiming. Skip slots due during accepted execution
   until physical settlement, then resume at the next future cadence slot.
   Notify the shared queue after each committed cursor/retry change.
   A completed one-time root controls its grant independently: human Pause/Resume
   affects eligible descendants without repeating that root. Retiming an accepted
   one-time timer is refused until settlement rather than silently discarded.
8. Persist definitions across restart. Acquire a strict database-scoped OS lock
   before automatic recovery/dispatch, using the existing portalocker dependency.
   Preserve advisory profile-instance behavior and unrelated manual chats; all
   native primary paths also use a per-conversation OS lock held until physical
   settlement. A secondary runtime never steals/reclaims live ownership. Move
   constructor orphan recovery behind explicit ownership; schema maintenance
   and blanket recovery require an exclusive maintenance gate; every primary and
   store mutation holds its shared gate, so quiescence is atomic. Share one startup owner/audit;
   recover pending work idempotently and pause uncertain accepted work for review.
   A committed transcript with incomplete accounting is reconciled without a new
   provider call. Maintain safe status projections and private bounded payloads.
   Bound aggregate scheduling state and settled history/receipts while pinning
   live/uncertain dependencies, durable lifetime counters and closed-day watermarks.
   Reserve terminal/control headroom before admission. A tool timeout alone is
   not physical settlement; uncertain effects block the grant without replay.
   Fresh narrowly scoped revocation remains available at full quotas or exhausted
   mutation allowance; it cannot create, resume, retime or enlarge a grant.
9. Execute only local native schedules while Chatbook is running. Navigation
   must not stop them, and first due work must initialize a viewless runtime even
   if Console has never opened. Server scopes, OS background operation, external
   agent-engine timers and durable spawned-agent resumption require separate work.

## Alternatives considered

| Alternative | Tradeoff |
| --- | --- |
| Store definitions in ScheduledTasksDB and execution in AgentRunsDB | Centralizes definition storage but adds cross-database fencing for revocation, claiming and accounting. The projection pattern keeps the execution transaction simpler. |
| Repeat a goal run at every due time | Appropriate for an independently requested recurring solve-and-verify workflow; adds objective lifecycle and completion review to ordinary scheduled replies. |
| Keep a worker asleep until the timer expires | Requires the original invocation to remain alive, consumes ownership, and loses durable timer identity across restart. |
| Let each agent-created timer mint a fresh budget | Allows recursive scheduling to bypass the existing automatic-work limits and loses the user's original scope of authority. |
| Start with a daemon or server | Extends process/account ownership and remote resource access beyond the approved app-running milestone. |
| Treat the advisory profile-instance lock as execution authority | It deliberately fails open and permits concurrent apps. Strict automatic-owner and per-conversation locks preserve that behavior while protecting recovery and conflicting primary work. |
| Let a lease timestamp prove that old work stopped | A delayed process or abandoned tool can still act. OS-held ownership and conservative effect recovery provide separate liveness and settlement evidence. |

## Consequences and review status

The accepted user flow reuses the scheduler and native runtime, while new storage,
grant, occurrence and tool contracts require implementation and targeted combined
tests. Explicit daily allowances make ongoing human schedules usable without
turning finite goal/fleet chains into perpetual grants. Agent-created schedules
remain finite by default and visible to the user.

The main-agent scope was retained through the approved design and review. Initial
limits are policy choices, not measured throughput. The user authorized correcting
the [2026-10-02 review](../../Docs/superpowers/reviews/2026-10-02-chat-schedules-and-agent-timers-review.md);
the revision addresses its eight contracts and current runtime integration.
The independent authority/runtime revision check resolved the original eight
findings; subsequent lifecycle/control clarifications are incorporated. The
revised design is accepted and claims no
implementation or live-model result. The
[implementation plan](../../Docs/superpowers/plans/2026-10-02-chat-schedules-and-agent-timers.md)
starts by integrating current dev prerequisites and
reconciles the later TASK-32195 claimant according to add-commit provenance.

Existing canonical references:

- [ADR-018](018-local-server-hybrid-scheduled-tasks.md)
- [ADR-019](019-watchlist-scheduler-migration.md)
- [ADR-069](069-console-project-instruction-local-state-and-preflight.md)
- [ADR-131](131-durable-agent-budget-accounting.md)
- [ADR-134](134-fleet-admission-and-automatic-work-budgets.md)
- [ADR-135](135-fleet-completion-delivery-and-crash-recovery.md)
- [ADR-141](141-native-console-goal-runs.md)
