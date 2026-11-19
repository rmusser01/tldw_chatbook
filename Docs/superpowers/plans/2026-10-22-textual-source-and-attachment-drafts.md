# Textual source and attachment drafts implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax.

**Goal:** Complete TASK-36130, TASK-36131 and TASK-36132, open a PR against dev, and address current-head CI/review feedback.
**Architecture:** Repair source/acknowledgement ownership within the existing widgets and handlers. Full editable audiobook source remains authoritative; submitted attachment and field acknowledgements preserve later user edits. No new global state, service boundary or dependency.
**Tech Stack:** Python ≥3.12, Textual 8.x (≥8.0.0,<9), existing SQLite/config services.
**Spec:** The three linked Backlog tasks and `Docs/superpowers/qa/2026-10-22-textual-book-after-pr3205.md`; user explicitly requested their implementation. Their outcome criteria are binding.

## Global Constraints

- Preserve path validation, storage admission, stale-editor fences and actual persisted identity/version.
- No provider/synthesis/network effects in regression tests; actual storage/services must not be replaced by fabricated success.
- Definite failure retry stays under existing service contracts; uncertain-write/idempotency policy is outside scope.
- Use explicit UTF-8 for every text read/write. Native tests use private 0700 profile/TEMP/TMP/bootstrap before pytest starts; `private_profile_test` functions accept `request`.
- Run targeted local tests only, plus genuine current-head hosted CI. No full local suite without explicit user opt-in.
- Do not raise module-size, boot or timeout ceilings. Preserve the committed UI census prefix/shard mapping; root owns shared census and diagnostic pins.
- Read design-language/component rules before UI edits; no visual token/keybinding change is planned.
- Use Superpowers requesting-code-review instead of Qodo. Create a PR against dev; merging the new PR is not part of this request.

ADR required: no
ADR path: backlog/decisions/033-application-session-state-ownership.md; backlog/decisions/011-chatbook-workbench-ui-system.md; backlog/decisions/046-roleplay-chat-display-identity-and-template-provenance.md
Reason: Bounded corrections within accepted source/editor ownership; no storage, runtime, service or cross-screen contract changes. ADR-150/161 apply to incidental UI composition.

## Review Focus

- Long import tail and literal ellipsis must survive in editor, source, chapters and Generate intent; cover all three import routes in Task 1.
- A deliberate source edit/clear must remain authoritative while real detection completes; keep TASK-36100 controls in Task 1.
- Later avatar upload must remain dirty through actual persistence completion and existing Stay guards; cover byte/identity/version and subsequent Save in Task 2.
- Partial field edits and raw-whitespace changes must survive while unchanged fields clear; cover independently in Task 3.
- Definite failures must retain drafts for retry without changing service semantics or stealing later focus; cover in Tasks 2 and 3.

### Task 1: Audiobook source integrity (TASK-36130)

**Files:** Modify `tldw_chatbook/UI/STTS_Window.py`; create `Tests/UI/test_audiobook_import_source_integrity.py`; update `Docs/Architecture/speech.md` after implementation. Root owns shared QA/task metadata/census/pins.
**Interfaces:** Consume existing widget events, service signatures and persistence results. Produce the same interface and corrected source/draft state; no later task depends on a new interface.
**Binding task:** `backlog/tasks/task-36130 - Keep-full-audiobook-imports-authoritative-over-shortened-previews.md`.
**ADR required:** no. **ADR path:** backlog/decisions/033-application-session-state-ownership.md; backlog/decisions/011-chatbook-workbench-ui-system.md. **Reason:** existing domain-owner repair.
**Approach:** Keep full imported content in the existing editable TextArea for file, Notes and Conversation imports. Preserve source freshness and existing import/service paths; do not let a shortened programmatic display replace source.

**Acceptance criteria:**
- [ ] #1 Successful audiobook file import preserves all imported text in the working source and actual Generate request, including text beyond character 1000; preview ellipsis is never silently substituted into that source.
- [ ] #2 Chapter detection and its editor/table reflect the complete imported source, including a chapter beyond the preview cutoff; intentional later edits and clears retain TASK-36100 freshness guarantees.
- [ ] #3 Equivalent Notes and Conversation import previews preserve the same full-source guarantee. Intentional source editing has a consistent, explicit relationship to the text submitted by Generate, without silently discarding unseen imported content.
- [ ] #4 Bounded mounted regressions drive real import/change/Generate events for long, exactly-1000 and short sources, demonstrate the defect before the fix and preserve ordinary imports and deliberate edits without provider or synthesis effects.

- [ ] Read full binding task, relevant ADRs, architecture, design/component rules and testing lessons. Trace the existing callers before editing.
- [ ] Read the task, shared source-change handler and all three import callbacks. Extend the retained real import/Generate witness into bounded maintained regressions: long file, exactly 1000 and short control; real Notes/Conversation service routes; deliberate edit/clear before detection completes. Record meaningful RED assertions, then keep the complete source in the existing editor. Verify editor/model, chapter tail and emitted Generate content separately. No synthesis or external provider calls.
- [ ] Run the new targeted regression file through `.test-tmp/task-36130-36132/run-targeted.py` with unique labels. Keep JUnit/log/argv receipts and exact RED/GREEN results. Run only relevant adjacent controls; qualify pre-existing failures without weakening guards.
- [ ] Run scoped Ruff lint/format and applicable unchanged module ceilings. Compare inherited diagnostics/format debt against the baseline by source/AST rather than reformatting large files.
- [ ] Update the named architecture paragraph and report exact files, source hashes, covering commands/results, AC coverage and limitations. Self-review, then hand the diff to root for independent spec/quality review. Root handles commits and shared task status after review.


### Task 2: Character pending avatar acknowledgement (TASK-36131)

**Files:** Modify `tldw_chatbook/Widgets/Persona_Widgets/personas_character_editor_widget.py; tldw_chatbook/UI/Screens/personas_screen.py`; create `Tests/UI/test_personas_pending_avatar_saves.py`; update `Docs/Architecture/ccp.md` after implementation. Root owns shared QA/task metadata/census/pins.
**Interfaces:** Consume existing widget events, service signatures and persistence results. Produce the same interface and corrected source/draft state; no later task depends on a new interface.
**Binding task:** `backlog/tasks/task-36131 - Preserve-avatars-uploaded-while-an-earlier-Character-Save-is-pending.md`.
**ADR required:** no. **ADR path:** backlog/decisions/046-roleplay-chat-display-identity-and-template-provenance.md; backlog/decisions/033-application-session-state-ownership.md. **Reason:** existing domain-owner repair.
**Approach:** Acknowledge the submitted attachment baseline while retaining later avatar bytes as an unsaved attachment and adopting the actual persisted identity/version. Preserve existing text/greetings acknowledgement and aggregate guards.

**Acceptance criteria:**
- [ ] #1 If a valid avatar upload is staged after an ordinary Character Save captures its submission, completing that Save retains the exact later bytes as an unsaved attachment while adopting the actual persisted record identity and version.
- [ ] #2 The current editor preview/readout and existing aggregate leave/quit draft predicate continue to represent the retained attachment; choosing Stay preserves it without silently acknowledging or replacing it.
- [ ] #3 A subsequent explicit Save persists the retained avatar to the same Character with the correct version and then becomes clean; uploads made before Save persist normally and failed saves retain staged bytes.
- [ ] #4 Mounted real-picker-result and real-store regressions distinguish upload-before-Save, upload-during-pending-Save and same-record retry/failure outcomes while preserving path validation, storage admission, stale-editor fences and existing raw-text/greetings coverage.

- [ ] Read full binding task, relevant ADRs, architecture, design/component rules and testing lessons. Trace the existing callers before editing.
- [ ] Read the task and trace avatar staging, Save capture, persistence, mark_saved and all acknowledgement callers. Extend the retained mounted real-picker/real-SQLite race into maintained regressions for pending update/create as applicable, ordinary upload-before-Save, preview/readout and New/leave Stay, next same-record Save and definite failure/retry. Establish intended RED before the minimal attachment-domain fix; preserve path/admission and editor identity fences.
- [ ] Run the new targeted regression file through `.test-tmp/task-36130-36132/run-targeted.py` with unique labels. Keep JUnit/log/argv receipts and exact RED/GREEN results. Run only relevant adjacent controls; qualify pre-existing failures without weakening guards.
- [ ] Run scoped Ruff lint/format and applicable unchanged module ceilings. Compare inherited diagnostics/format debt against the baseline by source/AST rather than reformatting large files.
- [ ] Update the named architecture paragraph and report exact files, source hashes, covering commands/results, AC coverage and limitations. Self-review, then hand the diff to root for independent spec/quality review. Root handles commits and shared task status after review.


### Task 3: Flashcards submitted-only clearing (TASK-36132)

**Files:** Modify `tldw_chatbook/UI/Study_Modules/flashcards_handler.py`; create `Tests/UI/test_study_flashcard_create_drafts.py`; update `Docs/Architecture/study.md` after implementation. Root owns shared QA/task metadata/census/pins.
**Interfaces:** Consume existing widget events, service signatures and persistence results. Produce the same interface and corrected source/draft state; no later task depends on a new interface.
**Binding task:** `backlog/tasks/task-36132 - Preserve-newer-Flashcards-drafts-when-earlier-creates-complete.md`.
**ADR required:** no. **ADR path:** backlog/decisions/033-application-session-state-ownership.md; backlog/decisions/011-chatbook-workbench-ui-system.md. **Reason:** existing domain-owner repair.
**Approach:** Capture raw deck/card authoring fields before normalization; clear each field after success only when its current value still equals that raw submission. Retain current editing context and the unchanged service, list/status and view-swap contracts.

**Acceptance criteria:**
- [ ] #1 A successful pending deck create persists only its submitted deck name and preserves any exact newer raw name still in the current authoring field, including a whitespace-only edit.
- [ ] #2 A successful pending flashcard create persists only its submitted front, back and parsed tags; independent or combined newer raw edits to any of those fields remain in the form. Completion preserves current editing context, including caret/selection and focus, without stealing focus deliberately moved elsewhere.
- [ ] #3 Unchanged fields clear after one successful creation; a definite failed create preserves current drafts and remains retryable under the existing service contract. Uncertain-write retry/idempotency policy is outside this task.
- [ ] #4 Bounded mounted regressions use the existing create events/workers, controlled service awaits and real StudyScopeService/LocalStudyService with SQLite; live drafts are asserted separately from stored records and unchanged-success/failure-retry controls pass.

- [ ] Read full binding task, relevant ADRs, architecture, design/component rules and testing lessons. Trace the existing callers before editing.
- [ ] Read the task, deck/card create callers and quiz acknowledgement precedent. Add bounded mounted real StudyScopeService/LocalStudyService/SQLite regressions with controlled service awaits: deck name, each card field independently and combined, raw whitespace, focus moved elsewhere, unchanged success and definite failure/retry. Record intended RED; make submitted-only clearing local to existing handlers. Do not introduce an uncertain-write exactly-once promise or repair separate list/status tasks.
- [ ] Run the new targeted regression file through `.test-tmp/task-36130-36132/run-targeted.py` with unique labels. Keep JUnit/log/argv receipts and exact RED/GREEN results. Run only relevant adjacent controls; qualify pre-existing failures without weakening guards.
- [ ] Run scoped Ruff lint/format and applicable unchanged module ceilings. Compare inherited diagnostics/format debt against the baseline by source/AST rather than reformatting large files.
- [ ] Update the named architecture paragraph and report exact files, source hashes, covering commands/results, AC coverage and limitations. Self-review, then hand the diff to root for independent spec/quality review. Root handles commits and shared task status after review.
