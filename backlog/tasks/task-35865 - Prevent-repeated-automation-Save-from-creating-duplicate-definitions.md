---
id: TASK-35865
title: Prevent repeated automation Save from creating duplicate definitions
status: In Progress
created_date: 2026-10-20 19:23
labels:
- scheduling
- textual-patterns
- reliability
priority: high
documentation:
- Docs/superpowers/qa/2026-10-20-textual-book-reanalysis.md
assignee:
- '@codex'
updated_date: 2026-10-20 19:42
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
One automation Save must produce one admitted save operation. Repeated Save gestures while a physical write is pending must not create duplicate definitions or report a single success over two persisted rows.

### Evidence and bounded scope

At merged dev 1ffc1ef0d6, automation_definition_form.py:941-974 starts an exclusive automation-form-save worker without pending admission. scheduling_service.py:2751 awaits the physical create in asyncio.to_thread. An isolated real-form/service/SQLite probe held the first create, waited for the normal button active effect to clear, and clicked Save again: two create calls persisted two distinct IDs with the same name while the form returned saved. Natural incidence under ordinary storage latency was not measured.

Book basis: Chapter 7 printed p.149 / PDF p.158; thread workers printed p.326 / PDF p.335. The book supplies background-worker ownership principles; the duplicate-write finding comes from current-code/runtime evidence.

### Existing ownership and decisions

TASK-19559 is the completed analogous Study durable-write correction, not coverage of this form. TASK-33622.15 owns modal quit behavior; TASK-33283 owns loop I/O and redraw cost. Neither owns repeated Save admission.

Related tasks: [TASK-19559](./task-19559%20-%20exclusive%3DTrue%20without%20group%3D%20makes%20every%20worker%20on%20a%20node%20cancel%20its%20siblings.md); [TASK-33622.15](./task-33622.15%20-%20CtrlQ-under-a-modal-in-flight-operations-and-the-video-save-picker-can-still-lose-work.md); [TASK-33283](./task-33283%20-%20PERF-24-Watchlists-and-Schedules---incremental-panes-DB-off-the-event-loop.md).

ADR required: no new ADR expected.
ADR paths: [018-local-server-hybrid-scheduled-tasks](../decisions/018-local-server-hybrid-scheduled-tasks.md); [099-schedule-editor-shape](../decisions/099-schedule-editor-shape.md); [116-schedules-inspector-editing](../decisions/116-schedules-inspector-editing.md).
Reason: No new ADR is expected for a localized UI admission repair preserving service, storage, ownership and sync semantics. ADR-018 is Accepted; ADR-099 and ADR-116 are Proposed context, not newly adopted by this task.

[Post-merge audit and evidence](../../Docs/superpowers/qa/2026-10-20-textual-book-reanalysis.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Two real Save gestures after the normal button active effect clears, while the first physical create is held, persist exactly one definition and publish one successful outcome.
- [ ] #2 Pending Save is visibly represented; repeated button and keyboard/direct action paths cannot replace or duplicate the admitted operation.
- [ ] #3 Validation/refusal and a genuinely failed save retain the form draft and permit a subsequent successful retry without duplicate rows.
- [ ] #4 Local create/edit, server-owned queued/refused outcomes, and existing modal close/quit guards retain their established behavior; no schema, sync-policy or generic worker-framework change is introduced.
- [ ] #5 Deterministic private-profile tests hold and release the real SQLite write, wait for physical settlement, and retain ordinary Save, failure and retry controls.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/018-local-server-hybrid-scheduled-tasks.md
Reason: bounded form admission fix; preserve existing writer, ownership, validation and outcome contracts. ADR-099/116 remain Proposed context.

1. Add real mounted form/service/SQLite duplicate Save, same-turn admission, and failure/retry regressions; record expected product failures first.
2. Admit Save synchronously and retain pending ownership until the existing operation settles, preserving close/quit and local/server outcomes.
3. Run targeted form/ownership checks and scoped static checks; update Schedules documentation.
4. Obtain Superpowers spec/quality review before controller marks acceptance criteria complete.

Detailed plan: Docs/superpowers/plans/2026-10-20-textual-book-followup-fixes.md (Task 1).
<!-- SECTION:PLAN:END -->
