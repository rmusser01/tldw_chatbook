# Textual save acknowledgement and chapter freshness implementation plan

> **For agentic workers:** Execute the four independent task plans with the Superpowers TDD and review workflows. Shared census, diagnostic metadata, Git integration and PR work belong to the root integrator.

**Goal:** Preserve newer editable state when earlier operations finish, and publish audiobook chapters only for the current source.

**Architecture:** Keep existing screen/widget/service ownership. Save completion adopts actual persisted identity/version and acknowledges only submitted fields; create completion clears only unchanged submitted authoring fields. Chapter-source edits invalidate older publication before any debounce dispatch.

**Tech Stack:** Python >=3.12, Textual >=8,<9, existing SQLite/Persona file/configuration services; no new dependencies.

**Spec:** The four linked task acceptance contracts and [source-qualified audit](../qa/2026-10-21-textual-book-after-pr3199.md), accepted for implementation by the user's burn-down request.

**ADR required:** no new ADR.
**ADR paths:** `backlog/decisions/046-roleplay-chat-display-identity-and-template-provenance.md`, `033-application-session-state-ownership.md`, `033-settings-commit-models-three-honestly-labeled.md`, `011-chatbook-workbench-ui-system.md`, `150-design-token-system-and-design-language.md`, `161-component-pattern-library.md`; ADR044 is secondary background with decision2 superseded by176.
**Reason:** Bounded repairs inside existing owners implement those decisions without new schemas, storage, services, global draft stores or mutation lifetimes.

## Global constraints

- Preserve existing actual record identity/version, checked storage/config authority, runtime/service boundaries and explicit Save/Rename/Revert semantics.
- Keep later raw drafts and current editing context; never restore obsolete focus after the user moves elsewhere.
- Use existing private-profile mounted tests and real stores. Controlled waits delay original entry points then delegate; do not fabricate success or bypass admission.
- Every test invocation uses a fresh native private TEMP/TMP and bootstrap/basetemp. Preserve red/green receipts. No full local suite unless requested; targeted integration and hosted required CI provide verification.
- No new CSS literals or stylesheet edits; existing token/component rules apply. Root alone edits shared census and diagnostic pins, assigns task status/ACs, commits and pushes.

## Review focus

1. Later edits during create and update adopt persisted identity/version but stay unsaved; a subsequent Save updates the same record.
2. No-edit success and failed saves/creates keep their existing behavior and retry correctly.
3. Current focus/caret/selection survives media completion, including a deliberate focus move and category reload.
4. Debounce gaps, empty sources and slow-A/newer-B completion cannot publish stale chapters or stale success notifications.
5. Test controls use actual persistence/event routing and bounded settlement; fixture errors, artificial local suspension and historical receipts are qualified honestly.

## TASK-36097: Character and Persona Save

**Contract and detailed implementation plan:** [TASK-36097](../../../backlog/tasks/task-36097%20-%20Keep-newer-Character-and-Persona-edits-dirty-when-earlier-saves-finish.md).
**Files:** tldw_chatbook/UI/Screens/personas_screen.py; Widgets/Persona_Widgets/personas_character_editor_widget.py; persona_profile_editor_widget.py; existing personas_pane_messages.py; UI/Persona_Modules/persona_save_completion.py (completion extraction preserves the existing size ceiling).
**Test:** `Tests/UI/test_personas_pending_save_drafts.py`.
**Interfaces:** Retain existing service signatures and local owner boundaries; no dependencies on another task's new API.

- [x] Read the task plan, governing ADRs, existing implementations and retained audit reproduction.
- [x] Write mounted tests covering all four ACs; run `python -m pytest Tests/UI/test_personas_pending_save_drafts.py -q --tb=short` under the private native test environment. Verify the actual intended assertion fails before product changes; separately verify ordinary controls.
- [x] Implement the smallest owner-local correction. Follow the task's explicit snapshot/source contract rather than a new generic framework.
- [x] Re-run that module and adjacent owner tests, then scoped Ruff lint/format. Record exact commands, sources, red/green receipts and qualifications.
- [x] Independent Superpowers review and integration checks; fix findings, complete task notes/ACs and mark Done via CLI only when DoD holds.

## TASK-36098: Image and Video Settings Save

**Contract and detailed implementation plan:** [TASK-36098](../../../backlog/tasks/task-36098%20-%20Preserve-newer-Image-and-Video-Settings-drafts-after-pending-saves.md).
**Files:** tldw_chatbook/UI/Screens/settings_screen.py and focused UI/Settings_Modules/media_gen_save.py (existing screen owns drafts/writes; helpers preserve its size ceiling). No panel module changes required.
**Test:** `Tests/UI/test_settings_media_pending_save_drafts.py`.
**Interfaces:** Retain existing service signatures and local owner boundaries; no dependencies on another task's new API.

- [x] Read the task plan, governing ADRs, existing implementations and retained audit reproduction.
- [x] Write mounted tests covering all four ACs; run `python -m pytest Tests/UI/test_settings_media_pending_save_drafts.py -q --tb=short` under the private native test environment. Verify the actual intended assertion fails before product changes; separately verify ordinary controls.
- [x] Implement the smallest owner-local correction. Follow the task's explicit snapshot/source contract rather than a new generic framework.
- [x] Re-run that module and adjacent owner tests, then scoped Ruff lint/format. Record exact commands, sources, red/green receipts and qualifications.
- [x] Independent Superpowers review and integration checks; fix findings, complete task notes/ACs and mark Done via CLI only when DoD holds.

## TASK-36099: Study quiz/question create

**Contract and detailed implementation plan:** [TASK-36099](../../../backlog/tasks/task-36099%20-%20Keep-newer-Study-quiz-drafts-when-earlier-creates-complete.md).
**Files:** tldw_chatbook/UI/Study_Modules/quizzes_handler.py; existing Study_Window handlers.
**Test:** `Tests/UI/test_study_quiz_create_drafts.py`.
**Interfaces:** Retain existing service signatures and local owner boundaries; no dependencies on another task's new API.

- [x] Read the task plan, governing ADRs, existing implementations and retained audit reproduction.
- [x] Write mounted tests covering all four ACs; run `python -m pytest Tests/UI/test_study_quiz_create_drafts.py -q --tb=short` under the private native test environment. Verify the actual intended assertion fails before product changes; separately verify ordinary controls.
- [x] Implement the smallest owner-local correction. Follow the task's explicit snapshot/source contract rather than a new generic framework.
- [x] Re-run that module and adjacent owner tests, then scoped Ruff lint/format. Record exact commands, sources, red/green receipts and qualifications.
- [x] Independent Superpowers review and integration checks; fix findings, complete task notes/ACs and mark Done via CLI only when DoD holds.

## TASK-36100: Audiobook source freshness

**Contract and detailed implementation plan:** [TASK-36100](../../../backlog/tasks/task-36100%20-%20Reject-audiobook-chapter-results-after-their-source-text-changes.md).
**Files:** tldw_chatbook/UI/STTS_Window.py.
**Test:** `Tests/UI/test_audiobook_chapter_source_freshness.py`.
**Interfaces:** Retain existing service signatures and local owner boundaries; no dependencies on another task's new API.

- [x] Read the task plan, governing ADRs, existing implementations and retained audit reproduction.
- [x] Write mounted tests covering all four ACs; run `python -m pytest Tests/UI/test_audiobook_chapter_source_freshness.py -q --tb=short` under the private native test environment. Verify the actual intended assertion fails before product changes; separately verify ordinary controls.
- [x] Implement the smallest owner-local correction. Follow the task's explicit snapshot/source contract rather than a new generic framework.
- [x] Re-run that module and adjacent owner tests, then scoped Ruff lint/format. Record exact commands, sources, red/green receipts and qualifications.
- [x] Independent Superpowers review and integration checks; fix findings, complete task notes/ACs and mark Done via CLI only when DoD holds.

Review refinements within TASK-36098 retain expanded nondefault sections and explicit Clear-then-retype history, including incremental typing and an equal global staging event. Existing dirty predicates continue to use `is_dirty`/`dirty_keys`; no new owner or framework is introduced.

## Integration and delivery

- [x] Preserve the full dev census prefix and append bounded regression selectors. Measure collection/runtime, check shard capacities and unchanged workflow timeouts.
- [x] Regenerate only required diagnostic/size metadata after checking exact source ownership deltas; no blanket baseline relaxations.
- [x] Run targeted combined UI/architecture/governance and Backlog guards, scoped lint/format and whitespace/source checks. Review the whole branch with `superpowers:requesting-code-review` instead of Qodo.
- [ ] Commit and push the reviewed changes; create and attach a PR against dev. Watch required CI and Perf Guard, inspect review threads, verify feedback before fixes, and continue until checks pass and actionable feedback is addressed. The user requested a PR, not a merge.


## Rebase CI follow-up: native served-relay import

The inherited dev Perf Guard and PR3205 both measure 1032 modules against 1031 at UI-ready; the native Windows reproduction measures 1033 because windows_files is also resident. app_served._served_relay_wrap_pending_approval imports the served publisher before returning the original callback for a native app with no publisher. Preserve that early-return behavior while avoiding its unnecessary Web_Server package/publisher imports.

ADR required: no
ADR path: backlog/decisions/097-boot-budget-ratchets.md; backlog/decisions/236-served-desktop-approval-relay.md
Reason: Mechanical deferral inside the existing TASK-35033 native/served seam; no behavior, authority, lifetime, service contract or budget increase.

1. Record current hosted and native census failures; add a fresh-process native wrapper regression and a real served publisher call-through control.
2. Verify the native import regression fails, then move the publisher import below the existing no-publisher return.
3. Verify focused publisher tests and real UI-ready census, lint/format, and independent Superpowers review. Push only while auto-merge is disabled, then re-arm after current-head review/thread checks and monitor CI through merge.

Verification: the new native-wrapper regression failed before the import move while its served positive control passed. Native full UI-ready census decreased from 1033 to 1031 without changing the 1031 ceiling (19 focused publisher/census tests passed in 79.792s); that legacy subprocess emitted a cp1252 stderr-reader warning, so clean stderr is not claimed. Final formatted source retains the 125-line module ceiling; all 18 publisher cases and both size checks passed (20 tests, 1.369s). Ruff lint and formatting pass. Historical incoming-dev Perf run 39744284548 already had the same Linux 1032/1031 breach as rebased PR run 39796132452. Fresh hosted verification is required after this follow-up push.
