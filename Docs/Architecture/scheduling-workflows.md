# Schedules and workflows

This document describes the unified scheduler (reminders, watchlist checks, briefings, automations), the transfer/ownership model with the server, the Schedules workbench, and the session-scoped execution exposed by the Workflows destination (ADR-138). Meetings is a capture surface outside the scheduler and is covered briefly at the end.

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
| Dreams | `Scheduling/services/dreams_projection.py`, `Scheduling/scheduler/handlers/dream_track_handler.py`, `Dreams/track_service.py` | `DreamsProjection`, `DreamTrackHandler`, `run_track_check` — local projected discovery/question-track jobs; tracked runs and due clocks live in DreamsDB |
| Liveness | `Scheduling/scheduler_heartbeat.py`, `tldw_chatbook/emergency_stop.py` | heartbeat JSON each tick; global emergency-stop file checked before dispatch (unreadable reads as **stopped** — fail-safe) |
| UI | `UI/Screens/scheduling/schedules_workbench.py` + siblings | three-pane workbench, results inbox, conflicts tab, sync status |
| Workflows | `UI/Screens/workflows_screen.py`, `Workflows/` package | local authoring surface + authoring/document/draft/session services — see below |

## Scheduler flow

1. **Define**: the workbench form → `SchedulingService.create_reminder` / `save_definition` → DB rows (`schedule_kind` one_time/cron for reminders; a `schedule` JSON dict for automations). Mutations call `request_reload()` so a new task arms on the next tick rather than the next periodic reload.
2. **Arm**: every poll interval (30 s) the loop freezes `now` and pops everything due (`next_run_at <= now`); server-scoped rows (`owner_id` starting `server:`) and rows in dormant transfer states are refused at both queue load and dispatch time.
3. **Execute**: emergency-stop check → run-ledger row (reminder/briefing types) → optional handler preflight (bounded 10 s) → the handler awaited inline under `wait_for` with the effective timeout (row override > 300 s default; ≤ 0 disables). Long handlers (briefings, automations) spawn `asyncio.Task`s held in strong-ref sets so a multi-minute LLM call cannot stall the serial tick; an overlap guard degrades too-frequent intervals to back-to-back runs, never concurrent. For detached jobs, this timeout bounds the awaited handler dispatch, not the spawned job's execution lifetime.
4. **Record reminders**: `mark_reminder_dispatched` advances the schedule from **now** (missed slots skipped), records `missed_at`/`missed_count` past the grace window with an attributed cause (`away`/`busy`/`stalled`), closes the ledger row, and dispatches notifications. Startup reconciliation fails `running` ledger rows left by a prior process.

The automation definition form admits Save synchronously before starting its
`automation-form-save` worker. While the admitted save is pending, its button
shows **Saving…** and is disabled; repeated button or direct Save actions retain
the original operation instead of cancelling a waiter whose SQLite thread may
still commit. Validation errors, server refusals and failed writes keep the draft
open and restore Save for retry. Successful local saves and offline server saves
retain the service's `saved` and `queued` outcomes respectively (ADR-018).

Hosted Results review actions (`r`/`d`/`o`/`a`) admit one mutation
synchronously before worker dispatch. Further review gestures show a busy refusal
until the mutation and repaint settle, so they cannot cancel an admitted per-row
Mark all read batch. Admission captures the selected result and its definition,
or the scope's current unread IDs; cursor movement, refreshed definition lookups,
and later arrivals cannot retarget it. A batch retains the existing per-row
success/failure count and failed rows remain unread for retry. Review-plus-outbox
transactions and each row's owner still belong to `SchedulingService` (ADR-018).

Reminder bulk toggle and confirmed deletion share one synchronous workbench
admission before worker submission. The batch freezes each row's ID and owner,
and toggle also freezes its desired enabled value. Further bulk gestures and
mark/clear gestures receive a busy refusal until mutation and refresh submission
settle; cursor navigation remains available. Completion removes only the captured
marks and preserves the existing per-row success/refusal/failure counts. Worker
submission failure retains marks for retry, and every settlement releases
admission. Service ownership, transfer guards and offline outcomes remain under
ADR-018; this screen-local guard does not establish physical cancellation of a
remote request or extend the batch lifetime beyond the workbench.

## What can be scheduled

- **Reminders** — one-time or cron-recurring; notification dispatch.
- **Watchlist checks and briefings** — projected from SubscriptionsDB (see [subscriptions-watchlists.md](./subscriptions-watchlists.md)); flag-gated.
- **Automations** — family `recurring_question`: a library RAG search plus a grounded answer on a cadence; findings land in `automation_results` and the results-tab inbox. Family `agent_task` exists in the enum only — no executor is registered.
- **Dreams discovery and question-track checks** — local, opt-in `[dreams]` projections; see below.
- **Meetings are not schedulable** — no handler, no projection, no task type.

`DreamsProjection` reads `[dreams] enabled` live and emits the local `dreams:cycle` task plus one `dream_track:<id>` task per **active question** item. Page-tracking uses the existing subscription/watchlist projection; paused, retired and page items emit no Dreams track task. `app_service_wiring.py` attaches the projection and `dream_track_check` handler in the same post-`_ui_ready` slice. `DreamTrackHandler` spawns checks into a strong-reference task set so they do not hold up the serial tick; `app_lifecycle.py` cancels them through the handler's separate bounded shutdown seam. `run_track_check` guards per-item overlap and records track runs plus `last_checked` in DreamsDB; that persisted clock drives the next projection instead of `mark_reminder_dispatched`.

## Server ownership and transfer (ADR-077/112)

`owner_id = "server:<user_id>"` rows are the server's to execute — the local loop never arms them, and manual run-now refuses them and any row mid-transfer. Transfers CAS `transfer_state` (e.g. `to_server_sent`); `recover_inflight_transfers` un-sticks crashed transfers at startup. Server-scoped history is server-authoritative — the local run ledger skips those rows.

## Workflows: local authoring and session execution

ADR-138 (`backlog/decisions/138-portable-workflow-definitions-and-local-execution.md`) governs portable workflow definitions and local execution. The current state on this branch:

- **Authoring is live.** The `Workflows/` package provides the model (`Revision` — immutable saved definitions with portable identity and lineage; `Draft`), a step catalog (contracts, field specs, discovery), the `DocumentService`, a draft-session service with confirmation versions, expressions, and `WorkflowAuthoring`. The app exposes them lazily (`ensure_workflow_authoring()` populating `app.workflow_documents` / `app.workflow_drafts`); the screen degrades to an "authoring owner unavailable" panel when they are absent. `WorkflowsScreen` is a library/navigator/editor authoring surface with save-revision, validate, import, and export.
- **Saved revisions can run.** `_run_saved()` opens setup and confirmation, then uses `app.ensure_workflow_session()` to compose one app-owned, in-memory sequential session. Run is enabled after the required services and saved-revision checks; an unsaved draft prompts for which revision to run. The supported execution path is keyless llama.cpp to Local Note, with existing session permissions and worker ownership. Execution is session-only, without persisted run recovery; the roadmap copy describes later branching/parallel scope.
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
| Handler timeout | Cancels the awaited handler; **reminder schedules still advance**. This does not bound work already detached into a spawned task. |
| Failed preflight | Distinct outcome + grouped incident; does **not** consume the occurrence (one-time stays armed) |
| Heartbeat/ledger/incident write failure | Never breaks dispatch (guarded, debug-logged) |
| Callback breakage | `on_reminder_dispatched` / `on_queue_changed` failures never fail a dispatch |
| Junk TOML values | `coerce_positive_float` fallback (bools rejected) |

## Governing decisions

ADR-018 (`018-local-server-hybrid-scheduled-tasks.md`), ADR-019 (watchlist scheduler migration), ADR-077 (`077-server-offloaded-scheduled-agent-tasks.md`), ADR-099 (`099-schedule-editor-shape.md`), ADR-112 (`112-per-task-schedule-ownership-transfer.md`), ADR-116 (`116-schedules-inspector-editing.md`), ADR-138 (workflow definitions and session execution), ADR-196 (`196-dreams-daily-discovery-and-tracking.md`; Dreams discovery/tracking). User guide: `Docs/User_Guide/schedules.md` (the authoritative behavior copy for liveness, preflight, incidents, lateness, run-now, transfers, timeouts).

## Verified gotchas

1. The scheduler worker is a **coroutine** worker (never `thread=True`) — the watchlists in-flight guard is lock-free on the one-event-loop invariant.
2. Missed-fire cause attribution is falsifiable: `busy` is backed by the previous tick's measured dispatch span.
3. `automation_health` is read-time only; persisting it would drift from the run ledger.
4. A reminder handler timeout advances its schedule — a wedged reminder handler can burn occurrences by design (bounded by the handler timeout).

Dreams startup uses a bounded first-dispatch wait plus a separate per-task guard
installed by `track_reconcile.first_dispatch_gate`. Timeout/error releases the
wait, while `DreamsDispatchGate.permits` retains refusal for pending Dreams-owned
subscriptions and linked reminders until persisted reconciliation is verified.
Unclassified watchlists stay held if ownership could not be read; unrelated
scheduler types and positively classified unrelated sources remain eligible.
Held occurrences do not enter the run ledger or advance their due clock.
