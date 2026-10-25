# Reminder edit identity and responsive quiz Start Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete TASK-36728 and TASK-36729, open a PR against dev, and qualify its CI and review feedback.
**Architecture:** Keep reminder exclusivity within the captured reminder and field. Run the shell quiz Start in a screen-owned worker, with shared controller admission so shell/inner gestures do not cancel or duplicate an admitted attempt. Retain existing service and persistence owners.
**Tech Stack:** Python 3.12, Textual 8.x, SQLite; no new dependencies.
**Spec:** `backlog/tasks/task-36728 - Keep-inline-reminder-edits-independent-across-reminder-identities.md`; `backlog/tasks/task-36729 - Keep-Study-navigation-responsive-while-shell-Start-quiz-awaits-the-server.md`; `Docs/superpowers/qa/2026-10-24-textual-book-after-pr3243.md`.
**Execution:** Native inline for two bounded fixes, with one fresh Superpowers whole-branch reviewer. User authorized implementing both specified task contracts, PR creation and CI/review follow-through; no additional design approval is needed for these routine choices.

## Global Constraints

- Python >=3.12; Textual >=8.0.0,<9. No new dependencies.
- Preserve captured mutation identities and values; cancelling an await is not evidence of remote rollback.
- Preserve same-reminder later intent, local/server validation, offline/queued and transfer restrictions.
- Preserve question-list, draft, panel-retirement and attempt ownership contracts.
- Workers for operations >100ms; no UI-thread waiting on pending server responses.
- Use existing $ds-* tokens and component patterns; no visual redesign or stylesheet changes.
- Targeted local verification only, per AGENTS.md; no full local test sweep. Required natural PR CI remains mandatory.
- Superpowers requesting-code-review replaces Qodo. Do not invoke Qodo.
- Do not change CI budgets, timeouts, shard counts or existing coverage allowances.

ADR required: no.
ADR path: backlog/decisions/018-local-server-hybrid-scheduled-tasks.md; backlog/decisions/116-schedules-inspector-editing.md (Proposed context); backlog/decisions/011-chatbook-workbench-ui-system.md; backlog/decisions/033-application-session-state-ownership.md.
Reason: localized UI scheduling and publication corrections; no storage, ownership, runtime or sync-policy boundary change.

## Review Focus

1. A failure from reminder A must not paint or clear reminder B's reused detail row; pin success/error/refusal after selection moves.
2. Same-reminder later edits retain the existing final-intent behavior; pin serial and overlapping saves with actual service/SQLite.
3. Repeated shell and inner Start gestures during one pending request must preserve one admitted attempt; pin both entry paths and their overlap.
4. Start failure or refusal must leave a retry usable; pin the real controller/service boundary and ordinary success.
5. Dashboard navigation remains delivered before held Start finishes, and completion does not force the user back to Quizzes; pin actual mounted events.

### Task 1: TASK-36728 — reminder edit identity

**Files:** Modify `tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py`; create `Tests/UI/test_schedules_reminder_edit_identity.py`; update `Docs/Architecture/scheduling-workflows.md` and the task record.
**Interfaces:** Consumes existing `ReminderFieldEditRequested(task, payload, row)` and `SchedulingService.edit_reminder_fields(task.id, payload)`. Produces the same service calls with independent worker ownership per reminder/field and feedback scoped to the captured target. No interface consumed by Task 2.

- [ ] Write mounted real-service/private-SQLite tests for independent reminders, same-reminder latest intent, serial success and failure/refusal while the detail row changes. Assert literal persisted values and displayed errors, not worker group strings.
- [ ] Run the new file on unchanged production code. Expected: independent-reminder and stale-feedback assertions fail; ordinary controls pass. Retain full output.
- [ ] Include captured reminder identity in the existing worker group; scope row error/clear publication to the reminder currently displayed. Preserve existing refresh and service paths.
- [ ] Run the whole new file and affected existing inline-edit controls; scoped lint/format and source guards. Expected: zero new failures; qualify inherited issues separately.
- [ ] Record evidence/ADR links and commit this task's code, tests and documentation.

### Task 2: TASK-36729 — responsive quiz Start

**Files:** Modify `tldw_chatbook/UI/Screens/study_screen.py`, `tldw_chatbook/UI/Study_Window.py`, `tldw_chatbook/UI/Study_Modules/quizzes_handler.py`; create `Tests/UI/test_study_quiz_start_responsiveness.py`; update `Docs/Architecture/study.md` and the task record.
**Interfaces:** Consumes the existing `StudyQuizzesController.start_attempt()` and `QuizScopeService.start_attempt(...)`. Produces identical attempt/session behavior with non-blocking shell dispatch and shared pending admission. Does not change Task 1 interfaces.

- [ ] Write mounted production StudyScreen/StudyWindow tests at a held fake server API boundary: shell/inner navigation, repeated cross-entry gestures, ordinary success, refusal and failure/retry. Verify actual attempt state and current section.
- [ ] Run the new regressions before production edits. Expected: shell navigation fails while the response is held; repeat/cancellation controls expose any existing admission gap.
- [ ] Dispatch the shell continuation through a named screen-owned worker. Admit one attempt in the shared controller before its first await; do not cancel that admitted start through repeated inner worker dispatch. Release admission on settlement/failure/cancellation.
- [ ] Run the whole new file plus existing quiz attempt retry and relevant Study shell tests, scoped lint/format and source guards. Expected: zero new failures.
- [ ] Record evidence/ADR links and commit this task's code, tests and documentation.

## Integration and delivery

- [ ] Append every new UI regression selector to the PR census without changing the existing prefix; register new split-file completeness requirements and raise only the literal census floor.
- [ ] Run targeted census/queue/ID contracts, changed-source lint/format and required source guards; preserve existing budgets and coverage.
- [ ] Generate the full branch review package and dispatch a fresh Superpowers reviewer. Resolve substantive findings with red/green evidence and record every scope ruling.
- [ ] Finalize task acceptance criteria/notes/status through Backlog CLI once their definition of done is met.
- [ ] Rebase latest dev if necessary, push with an explicit lease, create and attach the PR against dev, and monitor natural CI/review feedback to green.
- [ ] Save durable QA receipts, archive this plan's scratch ledger before its cleanup, and report PR/test outcomes and any rulings or deferred minors.
