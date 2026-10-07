# Schedules and workflows

This document describes the unified scheduler (reminders, watchlist checks, briefings, automations), the transfer/ownership model with the server, the Schedules workbench, and the honest state of the Workflows destination (decided in ADR-138, **not implemented**). Meetings is a capture surface outside the scheduler and is covered briefly at the end.

## Authoritative files

| Concern | File | Key symbols |
| --- | --- | --- |
| Loop | `Scheduling/scheduler/loop.py` | `SchedulerLoop` — `tick`, `_dispatch_due`, `_still_armable`, lateness causes (away/busy/stalled) |
| Queue | `Scheduling/scheduler/queue.py` | `PriorityQueue` rebuilt from DB + projections; drops server-owned and transferring rows |
| Handlers | `Scheduling/scheduler/handlers/` | `ReminderHandler`, `WatchlistCheckHandler`, `BriefingJobHandler`, `AutomationDefinitionHandler`; briefing/automation handlers are fire-and-forget off the tick |
| Trigger model | `Scheduling/schedule_compute.py` | `compute_next_run_at` — one_time, interval (60 s floor), daily, weekly, cron; **advance-from-now** (elapsed slots skipped, never replayed); never raises |
| Constants | `Scheduling/constants.py` | poll 30 s, missed-fire grace 60 s, handler timeout 300 s |
| Models | `Scheduling/models.py` | `ReminderTask`, `AutomationDefinition`, `AutomationRun`/`Result`, `AutomationFamily` (`recurring_question`, `agent_task` — enum-only, no executor) |
| DB | `Scheduling/db/scheduled_tasks_db.py` + `db/migrations/` | schema v7: reminders, automation definitions/previews/runs/results, audit events, sync tables, run ledger (`scheduled_task_runs`), incidents (`task_incidents`) |
| Service | `Scheduling/services/scheduling_service.py` | `SchedulingService` — CRUD, run-now, server transfer state machine, `recover_inflight_transfers` |
| Server sync | `Scheduling/services/sync_engine.py`, `server_client.py` | replays `pending_mutations`; typed server error hierarchy |
| Automations | `Scheduling/automation_execution.py`, `automation_health.py`, `recurring_question_scope.py` | `execute_recurring_question` = library RAG search + grounded answer; read-time health (never persisted) |
| Projections | `Scheduling/services/watchlist_projection.py`, `briefing_projection.py` | synthetic tasks from SubscriptionsDB cadences |
| Liveness | `Scheduling/scheduler_heartbeat.py`, `tldw_chatbook/emergency_stop.py` | heartbeat JSON each tick; global emergency-stop file checked before dispatch (unreadable reads as **stopped** — fail-safe) |
| UI | `UI/Screens/scheduling/schedules_workbench.py` + siblings | three-pane workbench, results inbox, conflicts tab, sync status |
| Workflows | `UI/Screens/workflows_screen.py`, `Workflows/` package | local authoring surface + authoring/document/draft/session services — see below |

## Scheduler flow

1. **Define**: the workbench form → `SchedulingService.create_reminder` / `save_definition` → DB rows (`schedule_kind` one_time/cron for reminders; a `schedule` JSON dict for automations). Mutations call `request_reload()` so a new task arms on the next tick rather than the next periodic reload.
2. **Arm**: every poll interval (30 s) the loop freezes `now` and pops everything due (`next_run_at <= now`); server-scoped rows (`owner_id` starting `server:`) and rows in dormant transfer states are refused at both queue load and dispatch time.
3. **Execute**: emergency-stop check → run-ledger row (reminder/briefing types) → optional handler preflight (bounded 10 s) → the handler awaited inline under `wait_for` with the effective timeout (row override > 300 s default; ≤ 0 disables). Long handlers (briefings, automations) spawn `asyncio.Task`s held in strong-ref sets so a multi-minute LLM call cannot stall the serial tick; an overlap guard degrades too-frequent intervals to back-to-back runs, never concurrent.
4. **Record**: `mark_reminder_dispatched` advances the schedule from **now** (missed slots skipped), records `missed_at`/`missed_count` past the grace window with an attributed cause (`away`/`busy`/`stalled`), closes the ledger row, and dispatches notifications. Startup reconciliation fails `running` ledger rows left by a prior process.

## What can be scheduled

- **Reminders** — one-time or cron-recurring; notification dispatch.
- **Watchlist checks and briefings** — projected from SubscriptionsDB (see [subscriptions-watchlists.md](./subscriptions-watchlists.md)); flag-gated.
- **Automations** — family `recurring_question`: a library RAG search plus a grounded answer on a cadence; findings land in `automation_results` and the results-tab inbox. Family `agent_task` exists in the enum only — no executor is registered.
- **Meetings are not schedulable** — no handler, no projection, no task type.

## Server ownership and transfer (ADR-077/112)

`owner_id = "server:<user_id>"` rows are the server's to execute — the local loop never arms them, and manual run-now refuses them and any row mid-transfer. Transfers CAS `transfer_state` (e.g. `to_server_sent`); `recover_inflight_transfers` un-sticks crashed transfers at startup. Server-scoped history is server-authoritative — the local run ledger skips those rows.

## Workflows: local authoring shipped; run gated off

ADR-138 (`backlog/decisions/138-portable-workflow-definitions-and-local-execution.md`) governs portable workflow definitions and local execution. The current state on this branch:

- **Authoring is live.** The `Workflows/` package provides the model (`Revision` — immutable saved definitions with portable identity and lineage; `Draft`), a step catalog (contracts, field specs, discovery), the `DocumentService`, a draft-session service with confirmation versions, expressions, and `WorkflowAuthoring`. The app exposes them lazily (`ensure_workflow_authoring()` populating `app.workflow_documents` / `app.workflow_drafts`); the screen degrades to an "authoring owner unavailable" panel when they are absent. `WorkflowsScreen` is a library/navigator/editor authoring surface with save-revision, validate, import, and export.
- **Run is explicitly unavailable in this authoring release** — the Run button ships disabled with the roadmap copy "Sequential v1; branching v2; parallel v3." Execution machinery exists underneath (one app-owned, **in-memory** sequential workflow session with no execution persistence, blocking local step effects for a retained worker, session permissions), matching the approved first-sequential-workflow execution design — but the UI does not expose running.
- The `Image|Video_Generation/workflows/*.json` files are provider pipeline payloads, unrelated to this destination. `Docs/User_Guide/workflows.md` is a self-declared stub.

Specs: `Docs/superpowers/specs/2026-09-14-workflows-authoring-compatibility.md`, `2026-09-14-workflows-authoring-dev.md`, `2026-09-16-workflows-first-run-design.md` ("first sequential workflow: execution design checkpoint").

## Meetings (brief)

`UI/Screens/meetings_screen.py` (F7) records a call/room with a live labelled transcript (`Audio/meeting_owner.py`, `meeting_session.py`); Stop saves raw audio + segment transcript and queues the recording for Library ingest (diarization) as a searchable media item. Server-side meetings listing is a separate scope service. Relation to the scheduler: none.

## Config keys (`[scheduling]`)

`sync_interval_seconds` (300) + sync retry knobs, `scheduler_poll_interval_seconds` (30), `missed_fire_grace_seconds` (60), `handler_timeout_seconds` (300), `reminder_catchup_hours` (24), `watchlist_checks_enabled` (true), `watchlist_checks_shadow` (false), `briefing_schedules_enabled` (true), `daily_report_demo_banner_dismissed` (false). DB path override `scheduled_tasks_db_path` (default `<user_data>/tldw_chatbook_scheduled_tasks.db`).

## Failure behaviors

| Case | Behavior |
| --- | --- |
| Emergency-stop file unreadable | Reads as stopped — holds work (fail-safe) |
| Handler timeout | Cancels the handler, records `timed_out`, **schedule still advances** |
| Failed preflight | Distinct outcome + grouped incident; does **not** consume the occurrence (one-time stays armed) |
| Heartbeat/ledger/incident write failure | Never breaks dispatch (guarded, debug-logged) |
| Callback breakage | `on_reminder_dispatched` / `on_queue_changed` failures never fail a dispatch |
| Junk TOML values | `coerce_positive_float` fallback (bools rejected) |

## Governing decisions

ADR-018 (`018-local-server-hybrid-scheduled-tasks.md`), ADR-019 (watchlist scheduler migration), ADR-077 (`077-server-offloaded-scheduled-agent-tasks.md`), ADR-099 (`099-schedule-editor-shape.md`), ADR-112 (`112-per-task-schedule-ownership-transfer.md`), ADR-116 (`116-schedules-inspector-editing.md`), ADR-138 (workflows — unimplemented). User guide: `Docs/User_Guide/schedules.md` (the authoritative behavior copy for liveness, preflight, incidents, lateness, run-now, transfers, timeouts).

## Verified gotchas

1. The scheduler worker is a **coroutine** worker (never `thread=True`) — the watchlists in-flight guard is lock-free on the one-event-loop invariant.
2. Missed-fire cause attribution is falsifiable: `busy` is backed by the previous tick's measured dispatch span.
3. `automation_health` is read-time only; persisting it would drift from the run ledger.
4. A handler timeout advances the schedule — a wedged handler can burn occurrences by design (bounded by the timeout).
