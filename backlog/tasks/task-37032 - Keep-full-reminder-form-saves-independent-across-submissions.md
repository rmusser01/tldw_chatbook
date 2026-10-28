---
id: TASK-37032
title: Keep full reminder-form saves independent across submissions
status: In Progress
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
assignee:
- '@codex'
updated_date: 2026-10-28 00:30
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
- [x] #1 Two distinct full reminder forms saved while the first response is pending cannot silently cancel one accepted mutation; each accepted submission retains its captured reminder ID, service owner and payload, or a later submission is explicitly refused before acceptance.
- [x] #2 Every accepted save has truthful acknowledgement or failure handling; its local mirror and visible Queue reflect the acknowledged outcome, and cancellation is never represented as proof of remote rollback.
- [x] #3 Serial full-form edits continue to persist and repaint both rows; preserve validation, transfer guards, existing recoverable failure feedback and offline queued behavior affected by the repair.
- [x] #4 Mounted private-profile tests drive actual full-form keys and Save callbacks over the real SchedulingService and SQLite, with ordinary and cross-reminder overlap controls, visible outcome assertions, bounded gate release and drained native work.
- [x] #5 Repeated full-form Saves for one captured owner and reminder settle in submission order or explicitly refuse the later Save before acceptance; an older acknowledgement cannot replace the newer local Queue value or delete its offline mutation, while different reminders and creates remain independent.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce I1 with mounted same-reminder Queue edit and real SchedulingService/SQLite, holding an accepted first response before a second full-form Save. 2. Serialize admitted same-owner/reminder form workers through existing native worker groups and settlement, capturing service, owner, ID and copied payload before admission; retain independent distinct reminders and creates. 3. Prove meaningful RED and GREEN for online and newer-offline outcomes, then run the affected nine existing form cases and scoped source/import/static checks without allowance increases. 4. Update architecture and implementation notes with actual policy, null/API limits and evidence; remain In Progress pending controller review/CI. ADR required: no. ADR path: N/A. Reason: same mounted UI owner directly implements backlog/decisions/018-local-server-hybrid-scheduled-tasks.md, backlog/decisions/033-application-session-state-ownership.md, backlog/decisions/036-application-service-composition-lifecycle.md and backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md; no new service, storage, runtime or retry-policy boundary.
<!-- SECTION:PLAN:END -->
## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented independent native full reminder-form Save workers with captured service/ID/owner and copied scalar/datetime form payloads. Creates retain separate admitted writes; real repeated Enter controls prove the form still submits once. Refused deleted/transfer-locked updates now use existing failure feedback and Queue refresh. A real offline form exposed inherited datetime JSON failure; the authorized bounded scheduling_service.py repair projects existing server-write fields and ISO datetimes, preserving nulls/partial payloads, local next-run clocks and top-level idempotency metadata. Real recurring/one-time replay traverses SyncEngine, SchedulingServerClient and ServerNotificationsService with controlled API transport and verifies durable queue settlement. Docs/Architecture/scheduling-workflows.md and incident-backed testing lesson updated. No new ADR: existing backlog/decisions/018-local-server-hybrid-scheduled-tasks.md, 033-application-session-state-ownership.md, 036-application-service-composition-lifecycle.md and 031-tui-keybinding-and-footer-hint-conventions.md apply. Targeted evidence: expected cross-reminder/create/refusal RED, all nine new scenario behaviors qualified through passing private-child receipts, 14 existing identity/bulk private cases and 12 focused legacy/service controls pass; two exact legacy UI modules use approved existing bootstrap-profile fixture. Six Python files format clean; seven inherited E402 violations remain. Size5532 and both import guards pass; unchanged worker inventory reports three inherited failures with exact-base detector output equality. Five unchanged native visual/path guard baselines are source-qualified from Task1. No full suite or policy/budget increase. Full commands, hashes, actual parent/child XML counts and qualifications: .superpowers/sdd/2026-10-27-textual-37031-37034/task-2-report.md. Status remains In Progress pending controller review and current-head CI.
Final review I1 follow-up: full-form updates now group by captured owner/reminder and await all prior same-node/group native workers before the service call; separate reminders and creates remain independent, and no admitted response is cancelled by a later Save. Mounted same-ID online and newer-offline current-source RED both reverted SQLite to First instead of Second (2 parent and 2 actual child failures; no errors/skips). Final full-form module GREEN: 11 parent and 11 actual child passes, no failures/errors/skips, including the nine original controls, correct newest Queue/SQLite, retained newer offline mutation and captured owner. Five scoped source/import guards pass: workbench5530 under unchanged5532 cap, Evals4211 untouched, preimport558/558, boot680/686 and first-send93/94; no new imports or allowance increases. Owned Python Ruff/format and diff checks pass; other callback/method ASTs and scoped worker-detector findings match FIX_BASE0e844e6e3a, retaining the existing loader finding. Existing M1 null transport omission and M2 broader static/output debt remain explicit. Docs/Architecture/scheduling-workflows.md records this same-screen full-form settlement policy and limits. ADR required:no; ADR path:N/A; existing backlog/decisions/018-local-server-hybrid-scheduled-tasks.md, 033-application-session-state-ownership.md, 036-application-service-composition-lifecycle.md and 031-tui-keybinding-and-footer-hint-conventions.md apply. No service/schema/retry/runtime change. Full commands, hashes and actual XML proof: .superpowers/sdd/2026-10-27-textual-37031-37034/final-fix-report.md. Remains In Progress pending controller scoped review/current-head CI.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->