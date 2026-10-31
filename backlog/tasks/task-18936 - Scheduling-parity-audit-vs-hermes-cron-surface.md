---
id: TASK-18936
title: 'Scheduling parity audit vs hermes-agent cron surface'
status: Done
assignee: []
created_date: '2026-08-19 10:25'
updated_date: '2026-10-11 00:00'
labels:
  - scheduling
  - audit
  - parity
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Audit of chatbook's Scheduling module against hermes-agent's cron surface, performed 2026-08-19 in `.worktrees/hermes-parity-audit` (branch `task/hermes-parity-audit`). The owner's claim to verify: "the scheduling module should be just as, if not more, powerful than hermes's solution." This task records the verified inventory, the verdict per capability, and the gaps that justify follow-up work. Hermes's cron-relevant capabilities (from release notes v0.13–v0.20.4): cron hardening waves (configurable timeout, manual-run attachments, missed-fire surfacing), cron continuity flags, durable execution-audit history, self-heal (EMFILE recovery, stale-claim reconciliation, wedged-job re-arm), Automation Blueprints (parameterized templates rendered per-surface), per-job model pickers, cron media-send, Cron Blueprints page in desktop, pluggable CronScheduler + managed-cron provider for scale-to-zero, and cron jobs as first-class sidebar entities.

**Chatbook inventory (verified in code):** local-first hybrid architecture per ADR-018 — `ScheduledTasksDB` (own SQLite + schema track, `reminder_tasks` with `missed_at` column, automation definitions, audit events, sync mappings/tombstones/conflicts); `SchedulerLoop` (30s poll, in-memory `PriorityQueue`, handler registry with honest startup configuration reporting); three registered handlers (reminder → NotificationDispatchService, watchlist_job, briefing_job); `SchedulingService` (CRUD + server sync with network-then-transaction boundary and server-wins conflict resolution); `SchedulesWorkbench` (routed three-pane UI: Queue/Conflicts tabs, task detail, inspector, create/edit/delete/enable-disable/marks/sync bindings per ADR-031); reminder form with cron presets and a live humanized cron preview; automation-definition domain model (families `recurring_question`/`agent_task`, lifecycle, health, preview, policies for visibility/notification/approval, audit events).

**Verdict per capability:**

- **Core scheduling engine (poll, dispatch, one-time + cron recurring, timezone-aware next-run)** — PARITY. chatbook's loop is honest about misconfiguration (orphaned-task reporting from TASK-1210) in ways hermes's notes don't claim; croniter + ZoneInfo timezone handling matches.
- **Storage durability + audit** — PARITY OR BETTER locally (dedicated schema, audit events, tombstones); hermes's durable execution-audit history is matched in concept.
- **Sync/offline** — chatbook BETTER for the local-first case (hermes cron is gateway-bound; chatbook reminders execute offline by design per ADR-018). Hermes's crash-window caveat on create is documented and accepted.
- **Missed-fire surfacing** — GAP. `missed_at` exists as a column and `last_status="missed"` is written on handler failure, but there is no missed-fire catch-up policy (a reminder due while the app was closed is simply late; recurring tasks re-derive next-run from *now*, skipping missed occurrences silently) and no user-facing "missed runs" surface. Hermes explicitly ships missed-fire surfacing and continuity flags.
- **Manual run / "Run now"** — GAP. No run-now action on any task in the workbench; hermes has manual-run (with attachments).
- **Retry/backoff, run controls** — GAP (partially by design). The deprecated `SchedulesScreen` renders retry/pause/resume buttons but all are disabled ("not wired yet"); the routed workbench has no run-control actions at all. Hermes has configurable timeouts and self-heal.
- **Execution-timeout configurability** — GAP. No per-task timeout knob (`[scheduling] scheduler_poll_interval_seconds` is the only knob found).
- **Automation Blueprints (parameterized templates)** — chatbook has the richer underlying *definition* model (families, policies, previews), but execution is `execution_unavailable` (ADR-018: server-side execution not yet integrated) — so hermes is AHEAD on actually running scheduled agent work, while chatbook is ahead on modeling. NET GAP in practice: reminders/briefings/watchlists run; `agent_task` automations do not execute locally.
- **Per-job model picker** — GAP (moot until agent_task execution exists; hermes's per-task model + thinking-depth from the kanban board is analogous).
- **Pluggable scheduler / scale-to-zero** — GAP (hermes-only concept, low relevance to a local-first TUI; note and skip).
- **First-class surfacing (sidebar/palette entity, Blueprints page)** — chatbook has a routed destination with keyboard model; hermes's extra surfaces are desktop-app concepts. Rough PARITY for a TUI.

**Owner direction (2026-08-19, post-audit):** tldw_server is to be viewed/treated the same as hermes's hosted gateway — the long-term goal is offloading tasks/scheduled work to the server to handle, passing results back to the client if so desired. Local execution remains the offline fallback (ADR-018's core), but the module's ambition is server-executed scheduled work with client-side result delivery, not merely local reminders.

**Recommended follow-ups (each deserving its own task if accepted):** (1) missed-fire catch-up policy + surfacing (decide: run-once-then-continue vs mark-missed; surface a "missed while away" state in the Queue tab and detail pane); (2) "Run now" action on any task, honoring the same handler path (also gives the workbench honest retry semantics: retry = run-now after a failure); (3) per-task timeout knob with a documented default; (4) local execution of `agent_task` automations (the big one — closes the only capability where hermes runs scheduled agent work and chatbook cannot; needs its own ADR since ADR-018 deferred execution to the server — under the owner's direction this task becomes the first server-offload execution seam: define the client↔server contract for server-executed agent tasks with results passed back).

**Missed-fire characterization (AC#3 — probe evidence, 2026-08-19):** a probe script exercising the REAL startup path (`ScheduledTasksDB` seeded in a temp dir → `PriorityQueue.load()` → `pop_due(now)` → `mark_reminder_dispatched`, exactly the loop's first tick) confirmed, at commit `34831c776`:

- **Overdue one_time (due 2h before open):** dispatched once, immediately, 2h late; task then disabled, `next_run_at=None`, `last_status="completed"`, `missed_at` NULL. The lateness is invisible — the row looks like an on-time success.
- **Overdue recurring (hourly, 2 owed occurrences):** dispatched ONCE (not twice); `next_run_at` re-derived from dispatch time (13:00, not from schedule); `last_status="completed"`, `missed_at` NULL. Elapsed occurrences are silently collapsed — no catch-up, no count, no surfacing.
- **What writes `last_status="missed"`:** only the handler-failure branch in `SchedulerLoop.tick` (`mark_reminder_dispatched(..., success=False)`). Even there, `missed_at` is never written — the column exists in the schema (v0→v1 migration line 36) but no code path populates it.
- **Net:** "missed" in this codebase means "dispatch attempted and handler raised", never "occurrence elapsed un-dispatched". Hermes's missed-fire surfacing and continuity flags have no equivalent here.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 This audit's inventory claims are re-verified against the implementing code at implementation time (file paths, handler registry, workbench actions) and corrections recorded here — at minimum: `tldw_chatbook/Scheduling/{models,db/scheduled_tasks_db,services/scheduling_service,scheduler/loop,scheduler/queue}.py`, `tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py`, handler wiring in `app.py` (~line 6829) — DONE 2026-10-11 at dev `9f8533686c`; corrections recorded in the Re-verification note below (schema v7; six handler keys, not three; wiring moved to `app_service_wiring.py` ~2725-2830/3560; missed-fire/run-now/timeout gaps closed by TASK-18937/18938/18939)
- [x] #2 Each identified gap has a decision recorded: file a follow-up task, fold into an existing task, or explicitly reject with a reason (e.g. pluggable scheduler/scale-to-zero as not relevant to local-first) — DONE 2026-08-19: missed-fire → TASK-18937; run-now/retry → TASK-18938; timeout knob → TASK-18939; server-offloaded agent_task execution (owner's stated long-term goal, tldw_server as hermes-style hosted gateway) → TASK-18940 (ADR required, amends ADR-018); pluggable CronScheduler/scale-to-zero → REJECTED, hosted-gateway concept with no local-first relevance; per-job model picker → folded into TASK-18940 AC#7
- [x] #3 The missed-fire gap is precisely characterized: what happens today when the app is closed across a due one_time reminder and across N missed recurring occurrences (verified against `mark_reminder_dispatched` and `_compute_next_run_at`, not asserted from reading alone — a seeded-DB test is acceptable evidence) — DONE 2026-08-19: probe evidence recorded in the Description (overdue fires once late and looks like on-time success; recurring collapses N owed occurrences to 1; `missed_at` never written by any path)
- [x] #4 Findings and decisions are summarized to the owner and the "as powerful as hermes" claim is answered with evidence (verdict table above is the starting point, corrected as needed) — DONE 2026-10-11: corrected verdict table in the Owner summary below; answer is now "as powerful or better on every local-execution axis; the one net gap (server-executed agent work) is TASK-18940, in progress"
- [x] #5 Any follow-up tasks created carry IDs swept against all remotes + worktrees at creation time (lessons-backlog-hygiene) — DONE 2026-10-11 sweep re-run: TASK-18937/18938/18939/18940 exist exactly once each in `backlog/tasks/` (no duplicates in tasks/ or drafts/); 18937-18939 are Done with implementation evidence, 18940 In Progress
<!-- AC:END -->

### Re-verification (2026-10-11, dev `9f8533686c`)

The original inventory was written 2026-08-19; the module has moved. Corrections, all verified in code this date:

1. **Handler registry is six keys, not three.** `app_service_wiring.py` builds `handlers = {"reminder" (ReminderHandler→NotificationDispatchService), "watchlist_job" (WatchlistCheckHandler, conditional), "briefing_job" (briefing_handler, conditional), "dreams_cycle", "automation_definition"}` at wiring (~lines 2769-2821) and adds `"dream_track_check"` later (~3560). The audit's "three registered handlers" is stale. Handler wiring is no longer in `app.py` (~6829); it lives in `tldw_chatbook/app_service_wiring.py`.
2. **Schema is v7, not the early track.** `ScheduledTasksDB._CURRENT_SCHEMA_VERSION = 7`; `reminder_tasks` now carries `missed_at`, `missed_count`, and `timeout_seconds` (columns written at the dispatch seam, per TASK-18937/18939).
3. **The missed-fire GAP is closed** (TASK-18937, Done): `mark_reminder_dispatched` writes `missed_at` + `missed_count` for elapsed undispatched occurrences; Queue tab and detail pane render a distinct "missed while away" state (◇ glyph) derived from `missed_at`, not the status enum. The probe findings in the Description above describe pre-18937 behavior and are retained as historical evidence.
4. **The run-now/retry GAP is closed** (TASK-18938, Done): `SchedulerLoop.run_reminder_now` shares the `dispatch_reminder` seam with `tick`; workbench exposes `action_run_task_now` and `action_pause_resume` — the audit's "no run-control actions at all" is stale.
5. **The timeout GAP is closed** (TASK-18939, Done): `[scheduling] handler_timeout_seconds` (default 300, ≤0 disables) + per-task `reminder_tasks.timeout_seconds`.
6. **Automation execution has partially landed since the audit.** `automation_definition` rows are real queue rows dispatched locally (the `recurring_question` family per the wiring comment block); `agent_task` family execution remains the server-seam work of TASK-18940 (In Progress). The audit's blanket "execution is `execution_unavailable`" claim is therefore now true only for the `agent_task` family.
7. **New module files not in the audit's inventory** (all verified present): `recovery.py`, `task_incidents.py`, `automation_execution.py`, `automation_health.py`, `automation_preview.py`, `scheduler_heartbeat.py`, `schedule_compute.py`, `schedule_input_parsing.py`, `schedule_vocabulary.py`, `slot_keys.py`, plus `handlers/{automation,dreams,dream_track,reminder,watchlist_check,briefing}_handler.py` and the `schedules/` workbench sub-package (`conflicts_tab.py`, `definition_detail.py`, `results_tab.py`, `task_detail.py`, `unified_rows.py`, `workbench_host_screen.py`, `sync_status_widget.py`).

### Owner summary (2026-10-11) — "as powerful as hermes?" corrected verdict table

| Capability | Audit verdict (2026-08-19) | Current verdict (2026-10-11) |
|---|---|---|
| Core engine (poll, dispatch, cron, tz) | PARITY | PARITY (unchanged; honest startup reporting retained) |
| Storage durability + audit | PARITY OR BETTER | BETTER (schema v7; incidents + heartbeat added) |
| Sync/offline local-first | BETTER | BETTER (unchanged) |
| Missed-fire surfacing | GAP | **PARITY** — closed by TASK-18937 |
| Manual run / Run now | GAP | **PARITY** — closed by TASK-18938 |
| Retry/backoff/run controls | GAP | **PARITY** — run-now doubles as retry; pause/resume live |
| Execution-timeout configurability | GAP | **PARITY** — closed by TASK-18939 |
| Automation Blueprints execution | NET GAP (execution_unavailable) | PARTIAL — `recurring_question` executes locally; `agent_task` awaits TASK-18940 (server seam, In Progress) |
| Per-job model picker | GAP (folded into 18940 AC#7) | GAP — pending TASK-18940 |
| Pluggable scheduler / scale-to-zero | GAP | REJECTED (hosted-gateway concept; no local-first relevance) |
| First-class surfacing | Rough PARITY | PARITY (routed workbench + sub-package panes) |

**Answer to the owner's claim:** on local-execution axes the module is now as powerful as or better than hermes's cron surface — every gap the audit identified except server-executed agent work has been closed by the 18937-18939 follow-ups. The single remaining net gap is server-offloaded `agent_task` execution (hermes runs scheduled agent work via its gateway; chatbook's server seam is TASK-18940, In Progress), which is also the owner's stated long-term direction (tldw_server as the hosted gateway). Local reminders execute offline by design per ADR-018 and remain the offline fallback.

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no.
ADR path: N/A.
Reason: this is an audit/verification task; any follow-up that changes scheduling execution semantics (notably local agent_task execution or missed-fire policy) requires its own ADR at that task's start — ADR-018 remains the governing decision for the module.

1. Re-verify inventory claims against current code; correct this file where reality differs
2. Characterize missed-fire behavior with a seeded-DB test or equivalent evidence
3. Record per-gap decisions (follow-up tasks filed with swept IDs, or documented rejections)
4. Summarize verdict table to the owner
<!-- SECTION:PLAN:END -->

## Implementation Notes

- Approach: audit-closure only — no code changes. The 2026-08-19 audit work (inventory, missed-fire probe, follow-up filing) was already complete for ACs #2/#3; this session closed the remaining ACs #1/#4/#5.
- AC#1: re-verified every inventory claim against dev `9f8533686c`; recorded seven corrections in the "Re-verification (2026-10-11)" section (handler count 3→6, wiring location app.py→app_service_wiring.py, schema v1-era→v7, three gap closures landed via 18937/18938/18939, partial automation execution, new module files).
- AC#4: corrected verdict table + owner answer recorded in the "Owner summary (2026-10-11)" section.
- AC#5: re-swept TASK-18937/18938/18939/18940 — each exists exactly once in backlog/tasks/, no drafts or duplicates.
- Modified files: this task file only.
