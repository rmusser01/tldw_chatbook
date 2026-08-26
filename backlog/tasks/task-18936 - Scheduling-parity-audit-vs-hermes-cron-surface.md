---
id: TASK-18936
title: 'Scheduling parity audit vs hermes-agent cron surface'
status: To Do
assignee: []
created_date: '2026-08-19 10:25'
updated_date: '2026-08-19 10:25'
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
- [ ] #1 This audit's inventory claims are re-verified against the implementing code at implementation time (file paths, handler registry, workbench actions) and corrections recorded here — at minimum: `tldw_chatbook/Scheduling/{models,db/scheduled_tasks_db,services/scheduling_service,scheduler/loop,scheduler/queue}.py`, `tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py`, handler wiring in `app.py` (~line 6829)
- [x] #2 Each identified gap has a decision recorded: file a follow-up task, fold into an existing task, or explicitly reject with a reason (e.g. pluggable scheduler/scale-to-zero as not relevant to local-first) — DONE 2026-08-19: missed-fire → TASK-18937; run-now/retry → TASK-18938; timeout knob → TASK-18939; server-offloaded agent_task execution (owner's stated long-term goal, tldw_server as hermes-style hosted gateway) → TASK-18940 (ADR required, amends ADR-018); pluggable CronScheduler/scale-to-zero → REJECTED, hosted-gateway concept with no local-first relevance; per-job model picker → folded into TASK-18940 AC#7
- [x] #3 The missed-fire gap is precisely characterized: what happens today when the app is closed across a due one_time reminder and across N missed recurring occurrences (verified against `mark_reminder_dispatched` and `_compute_next_run_at`, not asserted from reading alone — a seeded-DB test is acceptable evidence) — DONE 2026-08-19: probe evidence recorded in the Description (overdue fires once late and looks like on-time success; recurring collapses N owed occurrences to 1; `missed_at` never written by any path)
- [ ] #4 Findings and decisions are summarized to the owner and the "as powerful as hermes" claim is answered with evidence (verdict table above is the starting point, corrected as needed)
- [ ] #5 Any follow-up tasks created carry IDs swept against all remotes + worktrees at creation time (lessons-backlog-hygiene)
<!-- AC:END -->

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
