---
id: TASK-37032
title: Keep full reminder-form saves independent across submissions
status: To Do
created_date: 2026-10-27 18:00
labels:
- textual-patterns
- schedules
- correctness
priority: medium
references:
- tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py
- tldw_chatbook/Scheduling/services/scheduling_service.py
- Docs/superpowers/qa/2026-10-27-textual-book-after-pr3251.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep an accepted full reminder-form Save from silently losing its completion and local mirror update when another reminder form is saved. Each accepted independent mutation needs a truthful outcome.

The real Queue edit key and ReminderForm Save callback use one exclusive `schedules-save-reminder` group. With A held after a controlled external effect but before its response, saving B cancels A. Serial service/SQLite controls paint `EditedA, EditedB`; overlap paints `A, EditedB` although both controlled external effects occurred. Cancellation does not prove remote rollback. No real-server, same-reminder or offline outcome was dynamically qualified.

**Scope and related owners:** [TASK-36728](<task-36728 - Keep-inline-reminder-edits-independent-across-reminder-identities.md>) explicitly owns inline reminder detail edits, not full forms. [TASK-36036](<task-36036 - Keep-admitted-Schedules-reminder-bulk-actions-and-newer-marks-intact.md>) owns marked bulk actions; [TASK-35865](<task-35865 - Prevent-repeated-automation-Save-from-creating-duplicate-definitions.md>) owns duplicate automation-definition Saves. [TASK-19559](<task-19559 - exclusive=True without group= makes every worker on a node cancel its siblings.md>) owns default-group mistakes (this callback already names a group), and [TASK-19870](<task-19870 - Mutation-workers-refresh-inline-instead-of-dispatching-into-the-loader-group.md>) owns refresh dispatch after a completed mutation. [TASK-33283](<task-33283 - PERF-24-Watchlists-and-Schedules---incremental-panes-DB-off-the-event-loop.md>) and [TASK-33460](<task-33460 - Release-departed-Workflows-and-Schedules-screens-promptly-task-contexts-pin-them.md>) own performance/retention, not this mounted full-form overlap.

**ADR assessment:** No new ADR is expected for a bounded repair preserving existing ownership and service contracts. Existing [ADR-018](<../decisions/018-local-server-hybrid-scheduled-tasks.md>), [ADR-033](<../decisions/033-application-session-state-ownership.md>), [ADR-036](<../decisions/036-application-service-composition-lifecycle.md>), [ADR-031](<../decisions/031-tui-keybinding-and-footer-hint-conventions.md>) apply; reassess if implementation changes those boundaries.

Evidence, source pins, book pages, controls and limitations: [post-merge Textual audit](../../Docs/superpowers/qa/2026-10-27-textual-book-after-pr3251.md). Audited merged dev: `416feb50ee667fdc95cb269bf50f8b1d5b3ff8ff`. The supplied book is reference data.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Two distinct full reminder forms saved while the first response is pending cannot silently cancel one accepted mutation; each accepted submission retains its captured reminder ID, service owner and payload, or a later submission is explicitly refused before acceptance.
- [ ] #2 Every accepted save has truthful acknowledgement or failure handling; its local mirror and visible Queue reflect the acknowledged outcome, and cancellation is never represented as proof of remote rollback.
- [ ] #3 Serial full-form edits continue to persist and repaint both rows; preserve validation, transfer guards, existing recoverable failure feedback and offline queued behavior affected by the repair.
- [ ] #4 Mounted private-profile tests drive actual full-form keys and Save callbacks over the real SchedulingService and SQLite, with ordinary and cross-reminder overlap controls, visible outcome assertions, bounded gate release and drained native work.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
