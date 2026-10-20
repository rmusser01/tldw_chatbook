# Textual book follow-up fixes implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Complete TASK-35865 through TASK-35870 without changing existing persistence, runtime, or UI ownership boundaries.

**Architecture:** Keep each correction in its current screen/widget owner. Admit writes synchronously and refuse overlapping Results mutations with visible busy feedback; suppress programmatic input messages at publication and reject stale initialization; retain Rule draft state while updating only list presentation; derive each surface's footer/help from its existing action context.

**Tech Stack:** Python 3.12+, pinned Textual 8.x (installed 8.2.8), SQLite, existing pytest/private-profile helpers. No new dependencies.

**Spec:** The six reviewed Backlog records TASK-35865..35870 and `Docs/superpowers/qa/2026-10-20-textual-book-reanalysis.md`; each task's full acceptance criteria are authoritative. The user authorized implementation of these six records.

ADR required: no.
ADR paths: existing `018-local-server-hybrid-scheduled-tasks.md`, `033-settings-commit-models-three-honestly-labeled.md`, `033-application-session-state-ownership.md`, `042-watchlists-reader-first-ia.md`, `031-tui-keybinding-and-footer-hint-conventions.md`, `152-roleplay-mode-keys-single-letters-nav-layer-reserved.md`, `118-chunking-lab-local-execution-and-recovery.md`, `161-component-pattern-library.md` in backlog/decisions.
Reason: bounded corrections within existing UI/service owners. Scheduling ADR-099/116 remain Proposed context. Reassess before introducing storage/sync policy, new authority, or a shared service contract.

## Global Constraints

- Preserve local/server ownership, persistence writers, review/outbox transactions, validation, confirmation, and quit/close contracts; no schema or sync/conflict-policy changes.
- No generic worker framework, global action registry, root application state, dependency, or broad decomposition.
- Read `backlog/docs/design-language.md` and `backlog/docs/component-patterns.md` before UI edits. Reuse tokens/classes and existing component patterns; no visual redesign or new global key binding.
- Textual 8.x semantics control behavior. Keep printable typing, widget identity/focus/selection, and intentional guidance/refusal paths.
- Tests use existing private-profile conventions and real SQLite for DB behavior. Never use the real user profile. Test modules belong under Tests so conftest loads before app imports.
- Run only targeted tests; the user has not authorized a full suite. Do not weaken assertions, widen timeouts, bypass hooks/guards, or add broad warning filters to obtain green output. Disclose inherited warnings/failures with evidence.
- Native commands here use the primary venv Python and TEMP/TMP resolved to this worktree's `.test-tmp`; set PYTHONIOENCODING=utf-8. Do not use apply_patch inside `.test-tmp` because it changes its ACL.
- Implementers own only their task's product/tests/docs, may commit their exact files, never dispatch agents, and never mark the Backlog task Done. Controller owns task lifecycle, census integration and review gates.
- Keep every existing CI census entry and all limits. Controller admits new regression nodes after implementation, checking measured placement; no duplicate or omitted coverage. Update size/import ratchets only to justified measured values with no slack or waiver.
- Use Superpowers task and whole-branch review in place of Qodo. Retain worktree for PR iteration; do not merge the follow-up PR without user authorization.

## Review Focus

1. A second Save before its worker starts or while its physical write runs cannot produce two definitions; failure/retry leaves admission usable (Task 1).
2. Results selection/scope changes cannot retarget an accepted operation; overlap receives a visible refusal and the admitted batch settles (Task 2).
3. Stale initialization messages after a genuine input edit cannot revert the user's current text or stage an untouched setting (Task 3).
4. Raw invalid Rule text and original editing identity survive refresh/section return; Save/Cancel clear only that draft (Task 4).
5. Focus-stable busy/readiness transitions and narrow F1 entry project the actual action scope without new DB reads (Tasks 5-6).

## File structure and integration

Tasks 1-2 touch different Schedules UI owners and share only `Docs/Architecture/scheduling-workflows.md`; execute sequentially. Task 3 changes Settings and its measured size ratchet if necessary. Task 4 changes RulesPane and the Watchlists screen's existing view-state message handoff. Tasks 5-6 change separate action/help owners. Controller alone changes shared census/Backlog records and the final qualification report.

### Task 1: TASK-35865 - automation Save admission

**Spec:** `backlog/tasks/task-35865 - Prevent-repeated-automation-Save-from-creating-duplicate-definitions.md` (all five ACs).
**Files:** modify `tldw_chatbook/UI/Screens/scheduling/forms/automation_definition_form.py`; extend `Tests/UI/test_automation_definition_form.py`; update Schedules architecture/user documentation as needed.
**Interfaces:** keep `_run_save(self) -> None`, `_save_async(self, payload: dict[str, Any], owner: str) -> None` and `SchedulingService.save_definition` outcomes. No service/storage API changes.
ADR required: no; existing Accepted ADR-018; Proposed ADR-099/116 are context only.
- [x] Add `test_double_save_while_insert_pending_creates_one_definition` using real form/service/SQLite. Hold the first physical insert, wait for Button active-effect removal, press Save again; assert one persisted row and one successful outcome. Include same-turn pre-worker admission and genuine failure/retry controls.
- [x] Run the new nodes first and record expected duplicate/admission failures before changing product code. Fix fixture/collection errors before treating them as red evidence.
- [x] Add minimal synchronous admission and visible pending state around the existing Save dispatch; retain ownership until the admitted operation settles, prevent replacement by repeated gestures, and reset safely for genuine retry.
- [x] Run the affected form tests plus directly affected modal/ownership checks; verify local create/edit and queued/refused outcomes, scoped Ruff/format and whitespace. Record commands, red/green output and any inherited noise in the report.
- [x] Update relevant docs, self-review, and commit only owned files. Leave task status In Progress for controller review.

### Task 2: TASK-35866 - Results mutation admission

**Spec:** `backlog/tasks/task-35866 - Keep-later-Results-actions-from-cancelling-an-admitted-review-batch.md` (all five ACs).
**Files:** modify `tldw_chatbook/UI/Screens/scheduling/results_tab.py`; extend `Tests/UI/test_schedules_results_tab.py`; update Schedules documentation.
**Interfaces:** existing ResultsHostScreen actions and `review_automation_result`/outbox primitives remain. Refuse overlaps visibly while busy; do not add a command queue.
ADR required: no; existing ADR-018; ADR-116 remains Proposed context.
- [x] Add a real-keyboard/SQLite `test_mark_all_read_survives_overlapping_dismiss`: hold the first result read, press a then d, release and settle; expect three read rows and an explicit busy refusal, then a subsequent deliberate d works. Pin captured target/scope and single-row/solved/failure controls.
- [x] Record expected cancellation failure on original code.
- [x] Admit Results mutations synchronously in the existing host, retain the accepted target/scope, refuse later competing actions and release pending state on all outcomes. Preserve per-row errors/counts and transaction semantics.
- [x] Run the affected hosted Results tests and relevant service/outbox cases, scoped static checks and whitespace; record exact evidence.
- [x] Update docs, self-review and commit owned files; leave Backlog completion to controller.

### Task 3: TASK-35867 - remaining Console input event origins

**Spec:** `backlog/tasks/task-35867 - Keep-remaining-Console-input-refreshes-out-of-user-drafts.md` (all five ACs).
**Files:** modify `tldw_chatbook/UI/Screens/settings_screen.py`; create `Tests/UI/test_settings_remaining_input_event_origin.py`; update `Docs/Architecture/app-shell.md` and measured size expectations only if needed.
**Interfaces:** five AGENT_BUDGET_FIELDS, tool_result_display_chars, sidechat_model, sidechat_prompt_template keep current draft/validation/writer APIs.
ADR required: no; Settings ADR-033 and ADR-161.
- [ ] Add mounted private-profile tests matching the audit: only paste threshold 120 staged, first eight backing values changed, then changed again; expect draft exactly `{'paste_collapse_threshold': 120}` and every untouched display follows the second refresh. Add queued-stale/genuine typing, mount/return and compact/wide real-writer Save/Revert coverage.
- [ ] Record expected draft pollution/stale display failures before product changes.
- [ ] Use native publication prevention for programmatic assignments and correct stale/no-op event handling for initialization; remove obsolete temporary sync flags. Preserve invalid text and the first genuine edit after refresh, including a previous display value.
- [ ] Run new regressions, affected budget/side-chat tests and existing paste/parallel origin regressions; test Input identity/focus/selection, size/import guards and scoped static checks.
- [ ] Update owner documentation, self-review and commit owned files; do not extract another section or alter runtime budgets.

### Task 4: TASK-35868 - Watchlists Rule draft preservation

**Spec:** `backlog/tasks/task-35868 - Preserve-unsaved-Watchlists-rule-fields-across-refresh-and-section-return.md` (all five ACs).
**Files:** modify `tldw_chatbook/UI/Watchlists_Modules/rules_pane.py`, `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`; create `Tests/UI/test_watchlists_rule_drafts.py`; update Watchlists architecture documentation.
**Interfaces:** retain current pane-to-screen form-state ownership/message flow and save/cancel mutation APIs; draft remains ephemeral view state, never a new persistent owner.
ADR required: no; exact `033-application-session-state-ownership.md`, ADR-042 and ADR-161.
- [ ] Add New/Edit tests that actually type changed fields before a changed Rules snapshot; establish settled focus/cursor/selection first. Include raw invalid threshold text and original target identity. Add a mounted screen Rules -> another section -> Rules return plus Save/Cancel/validation cases.
- [ ] Record expected value/identity/section-return failures on original code.
- [ ] Update only list presentation on list changes; retain raw form draft at the existing narrow owner and carry it across section reconstruction. Preserve condition guidance and selection/initial-seeding contracts.
- [ ] Run new tests plus existing Rules pane/scoped-rebuild/seeding tests and measured size/import/static guards; record exact evidence.
- [ ] Update docs, self-review and commit owned files; no storage/sync policy or broad workbench rewrite.

### Task 5: TASK-35869 - Evals action scope

**Spec:** `backlog/tasks/task-35869 - Make-Evals-grid-footer-and-F1-help-follow-active-keyboard-scope.md` (all five ACs).
**Files:** modify `tldw_chatbook/UI/Screens/evals_screen.py` and only necessary ResultsGrid integration in `UI/Evals/results_grid.py`; create `Tests/UI/test_evals_grid_shortcuts.py`; update Evals architecture/user documentation.
**Interfaces:** l/b/s/e remain ResultsGrid-owned actions; existing ShortcutContext and WorkbenchHelpState projections consume one truthful context without new reads on focus/key dispatch.
ADR required: no; exact keyboard ADR-031 and ADR-152.
- [ ] Add mounted actual-Evals tests selecting a real run group, moving focus via F6/Tab among grid/rail/inspector/text controls and exercising the real grid actions. Cover compact/wide footer/F1 and action dispatch/typing.
- [ ] Record expected stale hints/missing-help failures on original code.
- [ ] Derive supported grid hints from mounted/focused/readiness context; cache only existing selection classification if needed to avoid focus-time DB calls. Project the same facts into F1 with honest scope requirements, preserving widget binding ownership.
- [ ] Run new tests and affected existing Evals help/grid/export checks plus relevant static/performance guards; record exact evidence.
- [ ] Update docs, self-review and commit owned files; no global registry or new shell binding.

### Task 6: TASK-35870 - Chunking Lab action/help state

**Spec:** `backlog/tasks/task-35870 - Keep-Chunking-Lab-shortcut-hints-honest-during-busy-and-editing-states.md` (all four ACs).
**Files:** modify `tldw_chatbook/UI/Screens/chunking_lab_screen.py`; create `Tests/UI/test_chunking_lab_shortcuts.py`; update `Docs/Chunking_Lab.md` as appropriate.
**Interfaces:** keep current coordinator/recovery/runtime APIs, native text-focus action gates and safe Escape behavior; use existing WorkbenchHelpPanel if a Lab-specific help projection is needed.
ADR required: no; keyboard ADR-031/152 and Lab ADR-118.
- [ ] Add actual mounted Lab tests for focus-stable idle -> busy -> idle, guarded/recovery/leaving and candidate readiness changes, plus Input/TextArea/Select focus and compact/wide F1. Use existing deterministic local coordinator/runner fixtures, not provider/network calls.
- [ ] Record expected misleading hints/help failures on original code.
- [ ] Share a small surface-local action-support projection between controls/footer/help, refresh on state and focus changes, and preserve useful refusal paths and printable typing.
- [ ] Run new tests and affected Lab screen/recovery/results tests plus relevant static/import guards; record exact evidence.
- [ ] Update docs, self-review and commit owned files; leave completed TASK-31645 and stale pre-renumber TASK-31428 unchanged.

## Controller closeout

- [ ] After each task, obtain spec-compliance and quality review against its full commit range; resolve findings before the next task. Record every scoped exclusion/ruling and verify unchanged-contract requirements.
- [ ] Only after evidence and review satisfy each AC, update its Backlog status/checkboxes/implementation notes via CLI, preserving metadata and ADR links.
- [ ] Admit new regressions to existing PR CI without dropping any of the 408 previous entries or raising limits; verify the current census floor, exact coverage and measured placement. Do not claim TASK-35834's separate three-run/headroom acceptance.
- [ ] Run focused combined checks, Backlog guards, link checks, scoped lint/format and generated-artifact checks applicable to actual changes. No full suite unless the user requests one.
- [ ] Obtain one fresh whole-branch Superpowers review on the most capable model, resolve its scoped findings, create/attach a follow-up PR against dev, inspect its CI and address feedback. Preserve the worktree and do not merge without user authorization.
