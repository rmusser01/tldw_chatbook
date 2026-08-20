---
id: TASK-18937
title: 'Scheduling: missed-fire policy and surfacing'
status: Done
assignee:
  - '@robert'
created_date: '2026-08-19 11:05'
updated_date: '2026-08-19 11:05'
labels:
  - scheduling
  - parity
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close the missed-fire gap found by the TASK-18936 parity audit (hermes ships missed-fire surfacing and cron continuity flags; chatbook has neither). Verified current behavior (probe evidence in TASK-18936): a reminder due while the app was closed fires once, late, and records `last_status="completed"` — lateness is invisible; an overdue recurring task collapses N owed occurrences into one late dispatch with next-run re-derived from dispatch time; `missed_at` exists in the schema but no code path writes it; `"missed"` status means only "handler raised".

Decide and implement a missed-fire policy: on reopen, tasks whose stored `next_run_at` is materially in the past should be surfaced honestly — record the actual owed-occurrence count and lateness, write `missed_at` when occurrences elapsed undispatched, and show a "missed while away" state in the Schedules Queue tab and task-detail pane (distinct from failed: the work never ran, as opposed to ran and raised). Catch-up semantics (re-running every missed occurrence vs run-once-then-continue, the current implicit behavior) are a product decision to record in the task before implementation; the recommendation is run-once-then-continue for reminders (matches user expectation of "at least it told me") with the missed count surfaced, not replayed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The missed-fire policy is decided and recorded in this task before implementation (catch-up vs run-once; what counts as materially late; one_time vs recurring rules) — decided 2026-08-19, recorded in the Implementation Plan section
- [x] #2 On reopen, a task with elapsed undispatched occurrences records honest state: owed-occurrence count and/or lateness persisted, and `missed_at` is written on that path (the column stops being dead schema) — `missed_at` + new `missed_count` (schema v2) written at the dispatch seam; tests `test_overdue_one_time_records_missed_state` / `test_overdue_recurring_counts_skipped_occurrences`
- [x] #3 The Queue tab and task-detail pane show a distinct "missed while away" state that is visually and semantically separate from failed (never-ran vs ran-and-raised), with the owed count where applicable — ◇ glyph on queue rows; detail-pane notice with skipped-count copy; both derived from `missed_at`, not the status enum
- [x] #4 The recurring next-run re-derivation rule is made a deliberate, documented choice (from dispatch time vs from schedule) — documented as a consequence of run-once-then-continue in the policy record and in `Docs/User_Guide/schedules.md`
- [x] #5 Behavior is pinned by tests using the real `ScheduledTasksDB` + `PriorityQueue` + dispatch path (seeded overdue one_time and recurring cases per the TASK-18936 probe), not a reimplementation — `Tests/Scheduling/test_missed_fire.py` (14 tests), all through the real loop/DB/service seams
- [x] #6 Docs updated: `Docs/User_Guide/schedules.md` (still a stub — extend it) documents the policy and the missed/failed distinction — "Missed while away" section added, grace-knob documented
<!-- AC:END -->

## Implementation Notes

Implemented 2026-08-19 in `.worktrees/hermes-parity-audit` (branch `task/hermes-parity-audit`).

**Approach.** Detection happens at one seam — the dispatch. When `SchedulerLoop.tick` dispatches a reminder whose stored `next_run_at` is more than `missed_fire_grace_seconds` (default 60s = 2× the 30s poll) in the past, `mark_reminder_dispatched` records the lateness: `missed_at` = the owed occurrence's scheduled time, `missed_count` = occurrences strictly between the owed one and the dispatch (exclusive both ends — an occurrence landing exactly at dispatch time coincides with the notification the user just got). An on-time dispatch clears both, so the state describes the last dispatch and self-heals. Run-once-then-continue: no replay of skipped occurrences.

**Prerequisite fix discovered during implementation.** Nothing pushed newly created/edited reminders into the live scheduler queue — only the periodic reload (~30 min at defaults) picked them up. Left alone, that delay would have manufactured false "missed while away" reports the moment a mid-session task dispatched. Fixed: `SchedulingService.on_queue_changed` callback (fired from create/update/delete/server-persist, exception-guarded) wired in `app.py` to `SchedulerLoop.request_reload()`, honored before every loop iteration. The app.py wiring resolves the loop lazily (same getter discipline as `BriefingJobHandler`'s `chachanotes_db_getter`) since the loop is constructed after the service.

**Schema.** v1→v2 migration adds `missed_count INTEGER NOT NULL DEFAULT 0` (`missed_at` existed since v1 but no code path ever wrote it). Two migration fixes surfaced by tests: `schema_version` is a bare table whose v0→v1 seed uses INSERT OR IGNORE, so v2 must DELETE-then-INSERT the single row (INSERT OR REPLACE would add a second row that `LIMIT 1` hides); and `v0_to_v1.rollback` now clears ALL version rows, not just version 1 — a fresh DB holds only "2". `missed_count` is client-local: the service never maps it to or from server payloads.

**UI.** Queue rows get a ◇ marker (alongside the existing ● mark glyph); the detail pane gets a plain-text (markup-free — titles are untrusted) warning-colored notice with the scheduled time and skipped count; `missed` in the queue filter matches late-dispatch rows as well as failed ones. `_was_missed_while_away()` is deliberately NOT a `TaskStatus`: "missed" as a status means ran-and-raised, which is orthogonal to late dispatch — a late dispatch can complete successfully.

**Verification.** `Tests/Scheduling/` fully green (284 passed) including 14 new missed-fire tests (migration up/rollback, one-time and recurring accounting, self-heal, grace boundary, handler-failure distinctness, 30-day every-minute bounded count, reload flag, callback firing + broken-callback survival, model mapping, helper derivation, queue regression). `Tests/UI/ -k sched` green (95 passed). `app.py` import-checked. Not verified live against a running app (backlog-docs-only worktree, no TTY session) — the UI assertions are covered by the existing Textual harness tests, and the dispatch-path behavior is covered end-to-end through the real `SchedulerLoop.tick`.

**Files modified:** `tldw_chatbook/Scheduling/db/scheduled_tasks_db.py`, `tldw_chatbook/Scheduling/db/migrations/v1_to_v2.py` (new), `tldw_chatbook/Scheduling/db/migrations/v0_to_v1.py` (rollback fix), `tldw_chatbook/Scheduling/db/schema.py` (unchanged — canonical DDL still v0→v1), `tldw_chatbook/Scheduling/models.py`, `tldw_chatbook/Scheduling/scheduler/loop.py`, `tldw_chatbook/Scheduling/services/scheduling_service.py`, `tldw_chatbook/app.py`, `tldw_chatbook/config.py`, `tldw_chatbook/UI/Screens/scheduling/task_detail.py`, `tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py`, `Tests/Scheduling/test_missed_fire.py` (new), `Tests/Scheduling/test_scheduled_tasks_db.py` + `test_migrations.py` (version assertions), `Docs/User_Guide/schedules.md`.

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no.
ADR path: N/A.
Reason: execution-policy refinement within ADR-018's existing local-first module; no schema/interface boundary change beyond populating an existing column. If the chosen policy turns out to require new sync semantics (missed-state reconciliation with the server), raise it then.

**Policy decided 2026-08-19 (before implementation):**

1. **Catch-up: NO — run-once-then-continue.** One late dispatch covers the interval; undispatched occurrences are counted and surfaced, never replayed.
2. **Detection seam: dispatch time.** When the loop dispatches a reminder whose stored `next_run_at` is more than a grace period in the past, the dispatch is "late" and the row records it. One seam covers app-reopen, reload, and mid-session creation.
3. **Grace:** new `[scheduling] missed_fire_grace_seconds` (default 60 = 2× the 30s poll). While the app runs, dispatch lands within one poll; beyond 2× means the scheduler wasn't running at the scheduled time. **Prerequisite fix:** reminders created/updated mid-session must reach the live queue immediately — today nothing pushes them (only the 60-tick ≈ 30-min periodic reload), which would manufacture false "missed" reports; fixed via a `request_reload()` flag on the loop wired from the service through an `on_queue_changed` callback.
4. **Persisted state (schema v2):** new `missed_count INTEGER`; `missed_at` (existing, dead) populated. On a late dispatch: `missed_at` = scheduled time of the earliest owed occurrence (the stored `next_run_at`), `missed_count` = owed occurrences − 1 for recurring (skipped before this one), 0 for one_time (fired late, nothing skipped). On an on-time dispatch both clear — the state describes the *last* dispatch and self-heals.
5. **Recurring next-run re-derivation stays from dispatch time** — now a deliberate, documented consequence of run-once-then-continue (not an accident of `mark_reminder_dispatched`).
6. **Distinct from failed:** `last_status="missed"` (handler raised — ran-and-raised) semantics unchanged; "missed while away" is derived from `missed_at`/`missed_count` and rendered as its own state. `missed_count` is client-local accounting: not pushed to the server, not expected in server responses.

Implementation steps:

1. Migration v1→v2 (`missed_count` column) + migration chain in `_initialize_schema`
2. DB: `mark_reminder_dispatched(..., scheduled_at=, grace_seconds=)` computes + writes/clears missed state
3. Model: `ReminderTask.missed_count`; loop passes scheduled time + grace; `request_reload()` flag honored each loop iteration
4. Service `on_queue_changed` callback (create/update/delete/persist-server-response) wired in `app.py`; config knob
5. UI: detail-pane Last Run + missed notice; Queue-tab missed indicator
6. Tests (real DB/queue/loop paths) + `schedules.md` docs
<!-- SECTION:PLAN:END -->

## Review Round (PR #1832, Qodo — 2026-08-19)

Ten findings; nine fixed, one declined with reasoning:

- **Fixed — sync staleness (the substantive bug):** `SchedulingService.sync_now()` now fires `on_queue_changed`; a pull that inserts/updates/deletes reminders reaches the live queue on the next tick instead of the periodic ~30-minute reload.
- **Fixed — silent truncation:** `_count_missed_occurrences` past its 100,000 cap now stores the sentinel `-1`, rendered as "more than 100,000 occurrence(s) were skipped" — never a capped exact.
- **Fixed — rollback drops indexes:** both rollbacks recreate the three v1 indexes after the table rebuild (pinned by `test_rollbacks_preserve_the_v1_indexes`).
- **Fixed — validation:** new `Scheduling/constants.py` (single source for the 30/60/300 defaults) + `coerce_positive_float`; junk TOML values degrade to documented defaults instead of crashing or classifying every dispatch late.
- **Fixed — repeated literals:** app wiring and loop defaults now reference the named constants.
- **Fixed — log context:** the on_queue_changed failure log carries owner + callback qualname.
- **Fixed — migration type annotations:** `_MigrationCapableDB` Protocol under TYPE_CHECKING (no import cycle).
- **Fixed — UI integration coverage:** new `Tests/UI/test_schedules_missed_notice.py` (7 tests) mounts the real TaskDetail and pins notice copy, overflow copy, clearing, and retry-label variants.
- **Fixed (CI, self-found):** the migration chain broke `:memory:` databases (every connection is a fresh empty DB; my `get_schema_version()` between migrations raised). Migrations are now memory-correct: each checks its own applicability structurally and skips on an empty connection. Also rebuilt the CSS bundle — the 18937 TaskDetail CSS was never regenerated into `widget_defaults_scoped.tcss`.
- **Declined — env-var override for the grace knob:** `get_cli_setting` has no env layer and no `[scheduling]` knob uses one (`scheduler_poll_interval_seconds`, `briefing_schedules_enabled` are TOML-only); adding one for just this key would break the section's convention. The underlying robustness concern is addressed by the coercion fix above.

Verification after the round: Scheduling + pragma suites 316 passed; UI sched + CSS-guard suites 98 passed.
