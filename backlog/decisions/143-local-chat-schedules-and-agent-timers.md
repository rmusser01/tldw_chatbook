# ADR-143: Local chat schedules and agent timers

Status: Proposed
Date: 2026-09-09
Task: [TASK-32195](../tasks/task-32195%20-%20Design-local-chat-schedules-and-agent-self-timers.md)
Design: [Local chat schedules and agent timers](../../Docs/superpowers/specs/2026-09-09-chat-schedules-and-agent-timers-design.md)
Amends: ADR-134/135 for explicitly authorized scheduled turns and shared recovery.
Preserves: ADR-018/019 scheduling ownership, ADR-069 project instructions, ADR-131 accounting, and ADR-141 goal semantics.

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
4. Provide `/schedule` and descriptor-backed `schedule_create/get/list/update/cancel`
   agent tools over one service. Resolve caller identity from trusted run context;
   scope v1 to the native main agent's own conversation. Model arguments cannot
   choose another conversation, grant, engine or approval authority. Mutation
   permissions and exact-call idempotency apply at the actual dispatch boundary.
5. Human setup may explicitly authorize recurrence until paused, subject to
   finite per-occurrence and per-conversation daily limits. A human-origin agent
   turn may create a finite grant through normal tool permissions. Automatic
   successor timers inherit a grant and its remaining total allowance; they
   cannot create or replenish one. Other automatic origins without a grant may
   propose a schedule for human review, not activate fresh spending authority.
6. Count all descendant scheduled work against the same grant and conversation
   window. Existing billing is distinct from budget-token admission. Daily
   windows are an explicit part of standing human authorization, with durable
   reservations and rollback detection; restart, cancellation, new timer IDs and
   midnight settlement do not erase accepted or uncertain charges.
7. Use unique occurrence identities and one pending/active occurrence per
   schedule. Coalesce overdue work into one catch-up, preserve manual capacity,
   and never replay accepted work whose effect is uncertain. Definition revisions
   fence pending edits; cancellation retains ownership until physical cleanup.
8. Persist definitions across restart. Share the existing startup owner/audit;
   recover pending work idempotently and pause uncertain accepted work for review.
   A committed transcript with incomplete accounting is reconciled without a new
   provider call. Maintain safe status projections and private bounded payloads.
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

## Consequences and review status

The accepted user flow reuses the scheduler and native runtime, while new storage,
grant, occurrence and tool contracts require implementation and targeted combined
tests. Explicit daily allowances make ongoing human schedules usable without
turning finite goal/fleet chains into perpetual grants. Agent-created schedules
remain finite by default and visible to the user.

The written design proposes initial limits and assumes the main-agent scope
from the unanswered scope question. This ADR remains Proposed until
the written design is reviewed. It claims no implementation or live-model result.

Existing canonical references:

- [ADR-018](018-local-server-hybrid-scheduled-tasks.md)
- [ADR-019](019-watchlist-scheduler-migration.md)
- [ADR-069](069-console-project-instruction-local-state-and-preflight.md)
- [ADR-131](131-durable-agent-budget-accounting.md)
- [ADR-134](134-fleet-admission-and-automatic-work-budgets.md)
- [ADR-135](135-fleet-completion-delivery-and-crash-recovery.md)
- [ADR-141](141-native-console-goal-runs.md)
