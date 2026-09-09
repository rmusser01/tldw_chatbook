---
id: TASK-18940
title: 'Server-offloaded scheduled agent tasks: execution seam and result pass-back'
status: In Progress
assignee:
  - '@robert'
created_date: '2026-08-19 11:05'
updated_date: '2026-08-19 11:05'
labels:
  - scheduling
  - agents
  - server
  - architecture
dependencies:
  - task-18937
  - task-18938
  - task-18939
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The strategic follow-up from the TASK-18936 parity audit, carrying the owner's stated long-term direction: **tldw_server is to be treated the same as hermes's hosted gateway — scheduled work is offloaded to the server to execute, and results are passed back to the client if so desired.** This closes the one capability where hermes runs scheduled agent work today and chatbook cannot: `agent_task` automations are fully modeled locally (ADR-018: families, lifecycle, health, preview, approval policies) but execution is `execution_unavailable` because ADR-018 deferred execution to server integration that never landed.

Scope: define and implement the first server-execution seam for `agent_task` automations — the client creates/previews a definition (already built), the server owns execution when the account is server-scoped (`owner_id="server:<user_id>"`), and completed results flow back through the existing sync/notification channels into the client's workbench (and optionally Console, reusing the existing handoff machinery). Local execution remains the offline fallback for local-owner definitions — this task adds the server path, it does not remove the local-first one. Hermes-precedent capabilities to fold into the contract: durable execution-audit history (the `AutomationAuditEvent` model exists), per-task model selection passed with the definition, and continuity/missed-fire reconciliation when the client reconnects (consumes TASK-18937's accounting).

This is architecture-first work: an ADR defining the client↔server execution contract (request shape, claim/lease semantics or server-side queue ownership, result-delivery channel, approval policy for server-side tool use, failure/timeout semantics, and what "passed back if so desired" means concretely — notification vs Console handoff vs workbench result row) must precede implementation. Sequenced after 18937–18939 so missed-fire, run-now, and timeout semantics exist as shared vocabulary in the contract.

**Server-side survey (2026-08-19, `tldw_server2` checkout, `origin/dev` @ `385afa95`)** — what the ADR builds on:

- **Reminders already execute server-side through a real jobs pipeline**: `app/services/reminders_scheduler.py` (APScheduler) enqueues due reminders into the core Jobs pipeline (`JobManager`, domain `notifications`, type `reminder_due`); `app/core/Reminders/reminder_jobs.py` consumes them with **durable run bookkeeping** (`create_reminder_task_run` rows, `run_slot_key` dedupe, terminal statuses `succeeded`/`skipped`/`failed`) and delivers via **user notifications** (`create_user_notification`, kind `reminder_due`, `dedupe_key=f"task:{task_id}:{run_slot_utc}"`). Gated by `REMINDERS_SCHEDULER_ENABLED` env.
- **The unified control plane exists and is client-ready**: `endpoints/scheduled_tasks_control_plane.py` exposes GET "" (unified task list across reminders/watchlist jobs/automation definitions), full CRUD on `/reminders`, and a complete **automation-definitions API** — capabilities, previews (24h TTL, payload-hash idempotency), definitions CRUD, audit history, pause/resume/archive/duplicate. Schemas in `scheduled_tasks_automation_schemas.py`; DB layer `Scheduled_Tasks_DB.py` (DefinitionRow/PreviewRow/AuditEventRow).
- **The gap is exactly where the client audit found it**: automation definitions on the server default `health="execution_unavailable"` (`DEFAULT_DEFINITION_HEALTH` in `scheduled_task_automation_service.py`) — definitions are validated, redaction-policy'd (agent_task input messages get `metadata_only` redaction), and audited, but **nothing dispatches them**. There is no APScheduler feed from automation definitions into the Jobs pipeline, and no `agent_task` job consumer. The modeling half is done on both sides; the execution half is missing on both sides.
- **Implications for the contract**: (1) the natural server execution shape mirrors the reminders pattern — a scheduler feed from automation definitions → Jobs pipeline → an `agent_task` consumer → notification delivery with a run-slot dedupe key, which composes with the client's missed-fire accounting (18937) for reconnect reconciliation; (2) the server's Jobs pipeline already has run lifecycle events/SLA plumbing the timeout semantics (18939) can align with; (3) the client's `missed_count`/`missed_at` being client-local (18937 decision) needs an explicit reconciliation rule in the ADR: server run rows are authoritative for what ran; client missed-state describes what the *client scheduler* missed, and the two must not double-count; (4) approval policy for server-side agent tool use must build on the definition's `approval_policy` field, which the server already validates but nothing enforces yet.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An ADR (amending/superseding the relevant ADR-018 clause) defines the server-execution contract: definition upload, execution ownership, result-delivery channel(s), approval policy for server-side tool use, failure/timeout semantics, and reconnect reconciliation — drafted and accepted before implementation begins
- [ ] #2 A server-scoped `agent_task` definition can be created/previewed locally, submitted for server execution, and its `health`/lifecycle transitions honestly (no more permanent `execution_unavailable` for server owners)
- [ ] #3 Completed server executions deliver results back to the client through at least one concrete channel (workbench result row, notification, or Console handoff — per the ADR), with the delivery visible in the UI and durable
- [ ] #4 Execution-audit history is durable end-to-end (client-visible audit trail of server executions, reusing `AutomationAuditEvent`)
- [ ] #5 Local-owner definitions keep today's local behavior unchanged — the local-first path is not regressed (pinned by tests)
- [ ] #6 Missed-fire/run-now/timeout semantics (18937–18939) reconcile correctly across a client reconnect (e.g. occurrences the server ran while the client was away are reported once, not re-derived or duplicated)
- [ ] #7 Per-task model selection rides the definition payload (hermes parity), bounded to the providers the server account can use
- [ ] #8 Live verification against a real tldw_server instance (per lessons-live-verification) — at minimum one real server-executed automation with result pass-back observed end-to-end; what was and was not verified live is recorded honestly
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes.
ADR path: backlog/decisions/077-server-offloaded-scheduled-agent-tasks.md (drafted as 072, renumbered twice at merge time as concurrent branches claimed 072–076; accepted 2026-08-23 with both judgment decisions approved by the owner; amends ADR-018's "execution remains execution_unavailable until server-side automation execution is integrated" clause).
Reason: cross-system service contract (client↔server execution ownership, result delivery, approval policy for server-side tool use) — squarely in ADR-required territory, and the owner has stated the long-term direction this task exists to realize.

1. Draft ADR-076: execution contract, result-delivery channels, approval policy, reconciliation semantics
2. Server client + service layer: definition submission, execution status polling/push, result retrieval
3. Client UI: health/lifecycle honesty for server owners; result row/notification/Console handoff per ADR
4. Audit trail wiring (AutomationAuditEvent end-to-end)
5. Reconciliation with 18937–18939 semantics; per-task model payload
6. Live verification against a real server; docs (schedules.md is a stub — this task should also give it its real content)
<!-- SECTION:PLAN:END -->

## Implementation Notes

**Progress log (task remains In Progress — slices land incrementally):**

- **Foundations (merged):** TASK-18937 (missed-fire accounting: `grace_seconds`/`missed_at`/`missed_count`, no-double-count rule), TASK-18938 (Run-now semantics), TASK-18939 (`timeout_seconds` + `TIMED_OUT` status) all landed on dev ahead of this task.
- **ADR-077 accepted** (`backlog/decisions/077-server-offloaded-scheduled-agent-tasks.md`): single-owner execution (server-scoped rows never dispatch locally), notification pass-back as the phase-1 result channel, phase-1 side-effect-free families only, `timed_out` as the shared vocabulary, control-plane authoring, run-now endpoint. Owner accepted both judgment decisions (no-double-count attribution; agent_task deferred with message-redaction preserved).
- **Server side (merged on tldw_server dev):** TASK-13020 (scheduler feed arming per-occurrence DateTrigger jobs), TASK-13021 (`agent_task_jobs.py` consumer with run-slot dedupe, timeout, notification pass-back; phase-1 unwired families skip with `family_not_wired_for_execution:<family>`), TASK-13022 (`scheduled_task_runs` table + statuses + result_summary), TASK-13110 (`POST /definitions/{id}/run` run-now endpoint with idempotency + lifecycle refusals). `agent_task` execution deliberately unwired (input.message redacted at rest — phase-2 design filed as server issue #2805).
- **Client slice 1 (PR #1986, merged 2026-08-29):** single-owner execution — `is_server_scoped_owner` predicate in `Scheduling/scheduler/queue.py` filters server-scoped rows from both `PriorityQueue.load` paths; `SchedulerLoop.tick`/`run_reminder_now` and `SchedulingService.run_reminder_now` refuse them; workbench Run-now shows a refusal toast using the shared predicate. Local-owner path pinned by tests (no regression). Files: `queue.py`, `loop.py`, `services/scheduling_service.py`, `UI/Screens/scheduling/schedules_workbench.py`, `Docs/User_Guide/schedules.md` callout, `Tests/Scheduling/test_owner_filter.py`. Diagnostic inventory pin regenerated for the two refusal logs.

**Remaining slices:** client surfacing/sync of server definitions (AC#2), result rendering from the notification feed (AC#3), per-task model selection on the definition payload (AC#7), live end-to-end verification with both server env gates enabled (AC#8), schedules.md real content.
