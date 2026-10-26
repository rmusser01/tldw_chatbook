# Textual follow-up fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Complete TASK-36930 through TASK-36932, open a PR against dev, and resolve review/CI failures.

**Architecture:** Reuse controller-local context generations, existing SQLite connection ownership and screen-owned worker admission. No new service boundary, dependency or durable state.

**Tech Stack:** Python 3.12, Textual 8.x, asyncio, SQLite, pytest.

**Spec:** The three Backlog task descriptions and acceptance criteria; Docs/superpowers/qa/2026-10-25-textual-book-after-pr3248.md records reproductions.

## Global Constraints

- Work only in the existing attached codex worktree. No stash, destructive resets, unrelated files or global Git configuration changes.
- Targeted verification only; no full local suite without user opt-in. Keep all privacy/lifecycle/network guards.
- Read relevant architecture, canonical ADRs, testing/live-verification/backlog lessons and design-language/component-patterns before edits.
- Mark each task In Progress and add its plan via Backlog CLI before code. Do not mark Done before evidence, static checks, review and notes.
- Root owns the shared CI census, plan, task completion and PR. Implementers own only their assigned production/test/architecture files; never dispatch helpers.

## Review Focus

- ABA context changes and replaced/detached answer panels must not restore old quiz publication.
- Thread-affine in-memory SQLite and awaitable alternate stores must retain their contracts.
- Store errors and runtime-policy refusal must remain nonfatal and never produce false success.
- Repeated Lore Save admission and selection/panel replacement must not duplicate or retarget durable writes; cancellation before worker entry must leave an explicit Save retry available, while running native work retains admission until settlement.
- New drafts edited during a save must survive completion, and result reloads must not paint a different selection.

### Task 1: TASK-36930 Study Submit publication

**Files:** Modify `tldw_chatbook/UI/Study_Modules/quizzes_handler.py` and `Docs/Architecture/study.md`; create `Tests/UI/test_study_quiz_submit_lifetime.py`. Existing adjacent tests may be extended if they give stronger actual UI coverage.

**Interfaces:** Consumes existing controller/service APIs only; produces corrected publication/admission behavior without a new shared interface. No other task consumes this task.

**Implementation:** Capture Submit mode, scope, generation, attempt and answer-panel identity before awaiting the existing service. Reject obsolete publication after every service await, including failures, and report prior-context completion truthfully without touching current state. Preserve nonfinal progression and definite pre-write retry semantics. Test runtime reset, scope ABA, attempt replacement and panel replacement, ordinary grading and failure/retry using mounted Study and actual local SQLite.

ADR required: no
ADR path: existing canonical ADRs 011, 033 linked by TASK-36930.
Reason: existing-boundary correction.

- [ ] Write behavioral regression cases in `Tests/UI/test_study_quiz_submit_lifetime.py` using production mounted surfaces and private real SQLite where applicable.
- [ ] Run the focused new regression selection. Expected: assertion failure reproducing the task defect, not setup/teardown errors.
- [ ] Implement the bounded correction in `tldw_chatbook/UI/Study_Modules/quizzes_handler.py` using the established local pattern.
- [ ] Run `Tests/UI/test_study_quiz_submit_lifetime.py` and adjacent owner tests with the private-profile harness. Expected: all selected tests pass, including actual child receipts.
- [ ] Run scoped ruff/format checks, update `Docs/Architecture/study.md` and self-review the diff. Expected: no newly introduced lint/format errors or unrelated changes.
- [ ] Report exact RED/GREEN commands and results; root integrates and commits only owned files after task review.

### Task 2: TASK-36931 Watchlists inbox store dispatch

**Files:** Modify `tldw_chatbook/UI/Subscription_Modules/notifications_inbox_controller.py` and `Docs/Architecture/notifications.md`; create `Tests/UI/test_notifications_inbox_responsiveness.py`. Existing adjacent tests may be extended if they give stronger actual UI coverage.

**Interfaces:** Consumes existing controller/service APIs only; produces corrected publication/admission behavior without a new shared interface. No other task consumes this task.

**Implementation:** Dispatch synchronous file-backed ClientNotificationsDB list/read/dismiss calls with asyncio.to_thread before evaluation. Keep thread-affine in-memory stores valid and retain awaitable seams, policy gates, nonfatal errors and existing loader serialization. Test real private file-backed durable state and controlled blocking/competing writes, in-memory compatibility and refusal/error paths. Do not repair Notifications pane rebuilding (TASK-33283).

ADR required: no
ADR path: existing canonical ADRs 011, 033, 042 linked by TASK-36931.
Reason: existing-boundary correction.

- [ ] Write behavioral regression cases in `Tests/UI/test_notifications_inbox_responsiveness.py` using production mounted surfaces and private real SQLite where applicable.
- [ ] Run the focused new regression selection. Expected: assertion failure reproducing the task defect, not setup/teardown errors.
- [ ] Implement the bounded correction in `tldw_chatbook/UI/Subscription_Modules/notifications_inbox_controller.py` using the established local pattern.
- [ ] Run `Tests/UI/test_notifications_inbox_responsiveness.py` and adjacent owner tests with the private-profile harness. Expected: all selected tests pass, including actual child receipts.
- [ ] Run scoped ruff/format checks, update `Docs/Architecture/notifications.md` and self-review the diff. Expected: no newly introduced lint/format errors or unrelated changes.
- [ ] Report exact RED/GREEN commands and results; root integrates and commits only owned files after task review.

### Task 3: TASK-36932 Personas Lore save operation

**Files:** Modify `tldw_chatbook/UI/Screens/personas_screen.py`, the existing `tldw_chatbook/UI/Persona_Modules/persona_save_completion.py` completion helper and `Docs/Architecture/character-chat.md`; create `Tests/UI/test_personas_lore_save_responsiveness.py`. Keep the screen within its existing module-size ceiling by placing bounded operation/completion logic beside the existing save helpers. Root lowers the measured Personas row in `Tests/Architecture/test_module_size_ratchet.py` and registers exact new cases in the existing UI census; no cap or shard-count increase.

**Interfaces:** Consumes existing controller/service APIs only; produces corrected publication/admission behavior without a new shared interface. No other task consumes this task.

**Implementation:** Release the screen handler promptly by admitting one screen-owned, noncancelling Lore settings save worker. Capture manager, book ID, expected version, payload and original panel before dispatch. Guard publication against retired selection/panel, preserve newer edits, and report captured-book success/failure truthfully. Test mounted held and ordinary saves using actual manager/private SQLite, repeated activation, conflicts, replacement, stale failures and pre-entry versus running-native cancellation. Preserve existing dirty/transition owners.

ADR required: no
ADR path: existing canonical ADRs 007, 011, 033 linked by TASK-36932.
Reason: existing-boundary correction.

- [ ] Write behavioral regression cases in `Tests/UI/test_personas_lore_save_responsiveness.py` using production mounted surfaces and private real SQLite where applicable.
- [ ] Run the focused new regression selection. Expected: assertion failure reproducing the task defect, not setup/teardown errors.
- [ ] Implement the bounded correction in `tldw_chatbook/UI/Screens/personas_screen.py` using the established local pattern.
- [ ] Run `Tests/UI/test_personas_lore_save_responsiveness.py` and adjacent owner tests with the private-profile harness. Expected: all selected tests pass, including actual child receipts.
- [ ] Run scoped ruff/format checks, update `Docs/Architecture/character-chat.md` and self-review the diff. Expected: no newly introduced lint/format errors or unrelated changes.
- [ ] Report exact RED/GREEN commands and results; root integrates and commits only owned files after task review.
