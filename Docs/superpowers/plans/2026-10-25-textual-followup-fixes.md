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

- [x] Write behavioral regression cases in `Tests/UI/test_study_quiz_submit_lifetime.py` using production mounted surfaces and private real SQLite where applicable.
- [x] Run the focused new regression selection. Expected: assertion failure reproducing the task defect, not setup/teardown errors.
- [x] Implement the bounded correction in `tldw_chatbook/UI/Study_Modules/quizzes_handler.py` using the established local pattern.
- [x] Run `Tests/UI/test_study_quiz_submit_lifetime.py` and adjacent owner tests with the private-profile harness. Expected: all selected tests pass, including actual child receipts.
- [x] Run scoped ruff/format checks, update `Docs/Architecture/study.md` and self-review the diff. Expected: no newly introduced lint/format errors or unrelated changes.
- [x] Report exact RED/GREEN commands and results; root integrates and commits only owned files, then obtains task-scoped review of the immutable revision.

### Task 2: TASK-36931 Watchlists inbox store dispatch

**Files:** Modify `tldw_chatbook/UI/Subscription_Modules/notifications_inbox_controller.py`, the existing mutation-worker admission in `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`, and `Docs/Architecture/notifications.md`; create `Tests/UI/test_notifications_inbox_responsiveness.py`. Existing adjacent tests may be extended if they give stronger actual UI coverage.

**Interfaces:** Consumes existing controller/service APIs only; produces corrected publication/admission behavior without a new shared interface. No other task consumes this task.

**Implementation:** Dispatch synchronous file-backed ClientNotificationsDB list/read/dismiss calls with asyncio.to_thread before evaluation. Keep thread-affine in-memory stores valid and retain awaitable seams, policy gates, nonfatal errors and existing loader serialization. Test real private file-backed durable state and controlled blocking/competing writes, in-memory compatibility and refusal/error paths. Keep admitted finite mutations alive through their completion-to-refresh path; retain latest-wins ownership for the loader. Add real SQLite A-held/B-completes-first coverage for mark-read and dismiss after the whole-branch review identified cancellation at this newly asynchronous boundary. Do not repair Notifications pane rebuilding (TASK-33283).

ADR required: no
ADR path: existing canonical ADRs 011, 033, 042 linked by TASK-36931.
Reason: existing-boundary correction.

- [x] Write behavioral regression cases in `Tests/UI/test_notifications_inbox_responsiveness.py` using production mounted surfaces and private real SQLite where applicable.
- [x] Run the focused new regression selection. Expected: assertion failure reproducing the task defect, not setup/teardown errors.
- [x] Implement the bounded correction in `tldw_chatbook/UI/Subscription_Modules/notifications_inbox_controller.py` using the established local pattern.
- [x] Run `Tests/UI/test_notifications_inbox_responsiveness.py` and adjacent owner tests with the private-profile harness. Expected: all selected tests pass, including actual child receipts.
- [x] Run scoped ruff/format checks, update `Docs/Architecture/notifications.md` and self-review the diff. Expected: no newly introduced lint/format errors or unrelated changes.
- [x] Report exact RED/GREEN commands and results; root integrates and commits only owned files, then obtains task-scoped review of the immutable revision.

### Task 3: TASK-36932 Personas Lore save operation

**Files:** Modify `tldw_chatbook/UI/Screens/personas_screen.py`, the existing `tldw_chatbook/UI/Persona_Modules/persona_save_completion.py` completion helper and `Docs/Architecture/character-chat.md`; create `Tests/UI/test_personas_lore_save_responsiveness.py`. Keep the screen within its existing module-size ceiling by placing bounded operation/completion logic beside the existing save helpers. Root lowers the measured Personas row in `Tests/Architecture/test_module_size_ratchet.py` and registers exact new cases in the existing UI census; no cap or shard-count increase.

**Interfaces:** Consumes existing controller/service APIs only; produces corrected publication/admission behavior without a new shared interface. No other task consumes this task.

**Implementation:** Release the screen handler promptly by admitting one screen-owned, noncancelling Lore settings save worker. Capture manager, book ID, expected version, payload and original panel before dispatch. Guard publication against retired selection/panel, preserve newer edits, and report captured-book success/failure truthfully. Test mounted held and ordinary saves using actual manager/private SQLite, repeated activation, conflicts, replacement, stale failures and pre-entry versus running-native cancellation. Preserve existing dirty/transition owners.

ADR required: no
ADR path: existing canonical ADRs 007, 011, 033 linked by TASK-36932.
Reason: existing-boundary correction.

- [x] Write behavioral regression cases in `Tests/UI/test_personas_lore_save_responsiveness.py` using production mounted surfaces and private real SQLite where applicable.
- [x] Run the focused new regression selection. Expected: assertion failure reproducing the task defect, not setup/teardown errors.
- [x] Implement the bounded correction in `tldw_chatbook/UI/Screens/personas_screen.py` using the established local pattern.
- [x] Run `Tests/UI/test_personas_lore_save_responsiveness.py` and adjacent owner tests with the private-profile harness. Expected: all selected tests pass, including actual child receipts.
- [x] Run scoped ruff/format checks, update `Docs/Architecture/character-chat.md` and self-review the diff. Expected: no newly introduced lint/format errors or unrelated changes.
- [x] Report exact RED/GREEN commands and results; root integrates and commits only owned files, then obtains task-scoped review of the immutable revision.


## Delivery checks

- [x] Finish the fresh whole-branch Superpowers review and address actionable findings.
- [x] Create and attach the PR against dev; verify hosted CI and current-head review feedback.
- [x] Complete task acceptance criteria and Implementation Notes via Backlog CLI after CI.

Verification record: `Docs/superpowers/qa/2026-10-25-textual-followup-fixes.md`.


## Whole-branch review follow-up

- [x] Reproduce and correct exclusive mutation-worker cancellation losing the late native write refresh (TASK-36931 AC2/4); preserve nonfatal policy/store behavior and loader ownership.
- [x] Add the optional direct obsolete-history-exception witness (TASK-36930 AC1/4) without changing production semantics.
- [x] Correct the historical TASK-36931 affected-source path through the task CLI; complete one scoped re-review of the entire fix wave.

ADR required: no
ADR path: existing canonical ADRs 011, 033, 042.
Reason: completion of the existing finite operation and UI refresh contract; no new architecture.


## CI guard integration follow-up

- [x] Express the protected post-await Lore library-pane lookup as a safe optional query, without increasing the worker-contract census.
- [x] Review all changed diagnostic statements and regenerate only the intended inventory owner rows (two moved calls, one new ID-only presentation warning; unchanged persistent sink topology).
- [x] Verify both exact guard scripts, the existing nine Lore regression cases and a focused review of this CI delta before final hosted qualification.

ADR required: no
ADR path: existing canonical ADRs 007, 011, 033 and existing persistent-diagnostic governance.
Reason: bounded lookup safety and regenerated source-derived metadata; no new service, sink or storage contract.


## CI qualification repairs (existing TASK-36695 and TASK-34000.14)

Final-head attempts exposed two independently controlled issues. TASK-36695 owns a real competing-render final sample publication race: canonical Undo is complete while the editor can remain stale after ordinary settling. TASK-34000.14 owns a test observation gap: an attached replacement Notes list may have a zero compositor region before layout. The original three tasks remain complete; reuse these existing owners without duplicate tasks.

ADR required: no
ADR path: existing canonical ADR-118, ADR-033 and ADR-011.
Reason: existing-owner applicability correction plus test readiness, without new storage, runtime, UI structure or service boundaries.

- [x] Add deterministic current-source Chunking render-order RED controls, then repair only owned final publication and verify newer-input protection plus adjacent edit behavior.
- [x] Replace Library test fixed delay with bounded current-list paint readiness; retain the original missing-folder negative control and all semantic/page-count assertions.
- [x] Verify actual child/canvas evidence, source guards, scoped static checks, architecture/QA notes and one focused Superpowers review of this CI delta.
- [x] Preserve all 906 current dev census slots and required caps, register the new exact controls, push the reviewed complete change, and remain on PR #3251 until current-head CI is green and review feedback is addressed.


### Focused CI repair review follow-up

- [x] Address the independently reproduced configuration-input regression introduced by the first final-render exception: restrict the exception to the sample echo while preserving the original configuration publication generation fence, or use an equally narrow proven guard. Add a permanent real-key/held-parent regression retaining the second accepted key and canonical result.
- [x] Verify the existing sample convergence and newer-input/status controls on the corrected source; run one scoped rereview of the complete feedback fix before pushing.

ADR required: no
ADR path: existing ADR-118 and ADR-033.
Reason: preserve the existing configuration authoring contract while correcting the same-owner sample publication defect; no new state/store/service boundary.


## Integration onto dev386 (required CI admission)

Dev advanced to `38623632e9` while the focused review ran. Preserve all newer upstream changes; thirteen reviewed files remain byte-identical, and the Personas added/deleted lines for this PR match the reviewed patch exactly alongside upstream search changes. Verify the nine Lore cases under the combined screen/new bootstrap fixture, the bounded Library cases, relevant architecture/census contracts and exact source guards before one scoped integration review. Preserve all 906 dev census entries, append the same 44 exact cases (floor 950), and keep 17 shards/1,200-second caps. Consolidate dev's adjacent 906/905 floor assignments into the one exact combined floor; no checker logic change. Pin the measured combined Personas size at 16,859, below current dev's 16,883 ceiling, preserving upstream pane/state ratchets.

ADR required: no
ADR path: existing ADR-007, ADR-011, ADR-033 and ADR-118.
Reason: mechanical integration and current-source verification; existing boundaries and reviewed feature behavior remain.


### Dev386 integration review correction (existing TASK-33910.35)

- [x] Preserve a permanent mounted regression of held native Lore Save, search read S, newer Save refresh R, actual list Enter while R is held, and first matching selection after R publishes. Retain the intended old-row Enter RED and independently clean teardown.
- [x] Complete first-match intent at actual current Lore-row publication, rechecking generation/query/mode after the row-update await. Remove unconditional landing when a superseded Lore search merely returns; let the authoritative Save refresh finish the same existing intent. Keep captured Save, window ownership, errors and other-mode behavior intact; no new timer/state owner.
- [x] Run the focused new overlap plus nine existing Lore cases and directly affected search/window controls, exact source guards and measured module/census checks, then one scoped rereview of the complete integration fix before push.

ADR required: no
ADR path: existing ADR-007, ADR-011 and ADR-033.
Reason: repair publication completion within the existing Lore list and first-match intent owners; no storage, service or new UI architecture.

The same integration correction also keeps the background Lore-save active-row marker from taking the cursor or creating a load-through intent after a newer search. Use the existing `mark_active_row(..., move_cursor=False, load_through=False)` options in `persona_save_completion.py`; no new owner/API. The mounted overlap has two query arms: captured book filtered out, and captured Second still included while First is the first match. Both must keep Enter inert while R is held, then retain the actual first match after R and save completion.


Canonical decisions for the CI integration corrections: [ADR-007](../../../backlog/decisions/007-personas-workbench-route-consolidation.md), [ADR-011](../../../backlog/decisions/011-chatbook-workbench-ui-system.md), [ADR-033](../../../backlog/decisions/033-application-session-state-ownership.md) and [ADR-118](../../../backlog/decisions/118-chunking-lab-local-execution-and-recovery.md). The final measured Personas owner after the publication correction is 16,870 lines, below current dev's 16,883 ceiling; no allowance is increased relative to the current target.
