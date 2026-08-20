# ADR-072: Server-offloaded scheduled agent tasks — execution seam and result pass-back

Status: Proposed
Date: 2026-08-19
Related Task: [TASK-18940](../tasks/task-18940%20-%20Server-offloaded-scheduled-agent-tasks-execution-seam.md)
Amends: ADR-018 (its "execution remains `execution_unavailable` until server-side automation execution is integrated" clause)
Related client work: ADR-018 (local/server hybrid storage + sync), TASK-18937 (missed-fire accounting), TASK-18938 (Run-now), TASK-18939 (execution timeouts)

## Decision

tldw_server is the execution authority for **server-scoped** scheduled agent
work, in the same role hermes-agent's hosted gateway plays for hermes: the
client authors and inspects scheduled work; the server runs it and passes
results back. The contract is:

1. **Execution follows ownership.** A scheduled task executes on exactly one
   side: `owner_id="server:<user_id>"` definitions are executed by the
   server and are **never dispatched by the client's SchedulerLoop**;
   `owner_id="local"` definitions execute locally (offline-first, ADR-018).
   The client's local scheduler stops arming server-scoped rows — today it
   loads every enabled reminder regardless of owner, which for offloaded
   agent work would be double execution by construction.

2. **The server mirrors its reminders pattern for automation definitions.**
   A server-side scheduler feed arms `configured` (non-paused, non-archived,
   non-disabled) automation definitions into the existing Jobs pipeline; a
   new `agent_task` job consumer executes them with the same durable run
   bookkeeping reminders already have (`run_slot_key` dedupe, terminal run
   statuses). Definition `health` becomes honest: `ready` when armed,
   `execution_unavailable` only when execution genuinely cannot run.

3. **Results pass back as notifications, reusing the feed the client already
   observes.** A completed (or failed, or timed-out) execution creates a
   user notification — the same channel server reminders already deliver
   through — carrying the definition name, the outcome, a **bounded result
   summary**, and a durable run reference. "If so desired" is governed by
   the definition's existing `notification_policy` field. The full result
   stays server-side unless requested; the server's existing
   `metadata_only` redaction policy for agent-task input messages is
   honored end-to-end.

4. **Phase 1 executes side-effect-free work only.** The first execution
   seam covers `recurring_question` and `agent_task` runs that require no
   tools and no unattended side effects. Tool-using agent tasks stay
   `execution_unavailable` until an approval-escalation design exists for
   server-side tool use — the client's nothing-is-auto-approved stance
   applies unchanged to work the user cannot see happening. That follow-up
   (and the approval-policy enforcement the server already validates but
   does not enforce) is explicitly out of scope here and gets its own ADR.

5. **Timeouts map onto one vocabulary.** The server's run statuses gain a
   timeout outcome matching the client's `timed_out` (TASK-18939); the
   client displays what the server reports rather than re-deriving it.

6. **Missed-fire accounting never double-counts.** Client-local
   `missed_at`/`missed_count` (TASK-18937) describe what the *client
   scheduler* missed and apply only to local-owner tasks. For server-scoped
   tasks the server's run rows are authoritative; the client derives
   nothing of its own. Reconnect reconciliation is the server's
   run-slot-dedupe plus the notification feed — occurrences the server ran
   while the client was away arrive once, as notifications, and are never
   re-announced.

7. **Authoring rides the existing control plane.** The client creates,
   previews, pauses, resumes, and archives definitions through the
   server's already-shipped automation API (preview-gated creates,
   payload-hash idempotency, audit history) and mirrors them locally
   through the ADR-018 sync discipline (server-wins, pending mutations,
   tombstones). A server-side "run now" endpoint (mirroring TASK-18938's
   manual dispatch, with the same run-slot dedupe) is a required server
   addition so Run-now works on offloaded tasks; until it exists, Run-now
   honestly refuses on server-scoped definitions.

## Context

TASK-18936's parity audit found the one capability where hermes runs
scheduled agent work and chatbook cannot: `agent_task` automations are
modeled on both sides (families, lifecycle, health, previews, policies —
richer than hermes's Blueprints conceptually) but execute nowhere. The
owner's stated long-term direction is that tldw_server fill exactly the
hosted-gateway role hermes's gateway fills: offload scheduled work, pass
results back.

A 2026-08-19 survey of `tldw_server` `origin/dev` @ `385afa95` (recorded in
TASK-18940) established that the server already has every ingredient except
the dispatch wiring: an APScheduler→Jobs-pipeline→consumer→notification
execution pattern proven by reminders (with durable runs and dedupe), a
complete client-ready control-plane API for automation definitions, and an
audit trail. Nothing on the server dispatches definitions; nothing on the
client asks it to.

Constraints that shaped the decision: the client is local-first (ADR-018) —
offline users must keep local reminders working, which single-owner
execution preserves; the client's approval philosophy admits no unattended
auto-approved side effects; and the result channel must be one the client
already consumes (the notifications feed) rather than a new push
subsystem.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Client executes everything, server only stores | Defeats the offload goal: scheduled agent work would require the client app to be running, and burns the user's local tokens unattended. Hermes parity specifically means server-side execution. |
| Server executes everything, client becomes a thin viewer | Breaks ADR-018's local-first guarantee for offline reminders; local watchlists/briefings already execute locally by design. |
| Both sides execute, dedupe at delivery | The current de-facto shape for reminders (the local loop arms every enabled row regardless of owner) — for agent work this is double execution with nondeterministic ordering, and dedupe after the fact cannot un-run side effects. Single-owner execution is simpler and honest. |
| Client polls a new results endpoint | New polling surface for data the notifications feed already delivers; the client's `ServerNotificationsService` already observes that feed. |
| Full agent_task (tools) execution in phase 1 | Tool use on the server implies unattended, unreviewed side effects — violates the approval stance until an escalation design exists (the server validates `approval_policy` but enforces nothing today). |

## Consequences

- The client's `SchedulerLoop` gains an owner filter: it arms only
  local-owner reminder rows. This **changes current behavior** for
  server-scoped reminders (they stop firing locally and rely on the
  server's copy — the notification arrives via the server feed instead of
  the local dispatch). That is the correct steady state for the hybrid
  model but must be called out in the Schedules user guide and verified
  against a real server account before release.
- Server-side work is required and tracked in the server repo: the
  definition scheduler feed, the `agent_task` consumer, the timeout run
  status, the run-now endpoint, and the result notification. The client
  work (owner filter, definition sync surfacing, run-now refusal,
  result-row rendering) cannot ship meaningfully ahead of it; the seam is
  the contract, and both sides verify against a live server per
  TASK-18940 AC#8.
- `automation_definitions` sync joins reminders' server-wins discipline;
  conflicts land in the existing `sync_conflicts` surface, not a new one.
- Definition health becomes a real signal the Settings/Schedules surfaces
  can rely on (`ready` vs `execution_unavailable`), replacing the current
  permanent `execution_unavailable` lie.
- Tool-using agent tasks remain modeled-but-unexecuted until the
  approval-escalation ADR; the capabilities endpoint keeps advertising
  that honestly.
- TASK-18938's Run-now refuses (with a reason) on server-scoped
  definitions until the server's run-now endpoint exists — refusal is
  honest, silent wrong-side dispatch is not.

## Links

- [TASK-18940 — Server-offloaded scheduled agent tasks](../tasks/task-18940%20-%20Server-offloaded-scheduled-agent-tasks-execution-seam.md) (implementation task; carries the server survey evidence)
- [ADR-018 — Local/server hybrid scheduled-tasks storage and sync](018-local-server-hybrid-scheduled-tasks.md) (amended here on execution availability)
- [TASK-18937 / 18938 / 18939](../tasks/) — the client-side vocabulary this contract composes with (missed-fire, run-now, timeouts)
- Server-side counterparts (tldw_server repo): `scheduled_tasks_control_plane.py`, `scheduled_task_automation_service.py`, `reminders_scheduler.py`, `core/Reminders/reminder_jobs.py` — the pattern this ADR mirrors.
