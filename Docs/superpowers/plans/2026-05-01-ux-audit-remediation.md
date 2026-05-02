# UX Audit Remediation Plan

Date: 2026-05-01
Status: Current-dev rebaseline after UX remediation PRs #146-#186, plus active Study quiz Review in Chat slice
Branch context: current `dev` / `origin/dev` at `6f1f39fc`; active branch `codex/ux-quiz-review-chat-handoff`
Previous audit baseline: `e2576cae`

## Goal

Fix every UX failure identified during the live application audit while preserving the Textual/TUI product model, power-user speed, and the chat-first direction.

This is not a visual refresh. The work is ordered around workflow completion, recoverability, and information architecture clarity.

## Audit Sources

- `/private/tmp/tldw-chatbook-current-dev-ux-audit-2026-05-01/audit-probe.json`
- `/private/tmp/tldw-chatbook-current-dev-workflow-audit-2026-05-01/workflow-probe.json`
- `Docs/superpowers/specs/2026-04-20-ux-rescue-audit-design.md`
- `Docs/superpowers/specs/2026-04-21-use-in-chat-handoffs-design.md`
- `Docs/superpowers/handoffs/2026-04-30-backend-parity-ux-handoff.md`
- Current-dev source inspection and focused tests on 2026-05-01:
  - `Tests/UI/test_chat_first_handoffs.py`
  - `Tests/UI/test_search_handoffs.py`
  - `Tests/UI/test_media_handoffs.py`
  - selected Notes handoff tests in `Tests/UI/test_notes_screen.py`
  - `Tests/UI/test_chat_screen_state.py`
  - `Tests/UI/test_chat_tab_container.py`
  - existing Chatbooks, screen navigation, Ingest, Study, and Search/RAG tests

## Current Dev Rebaseline

Verified on current `dev` at `6f1f39fc`:

- Phase 0 and Phase 1 are merged: the shared shell/Chatbooks trap, Ingest default source, quiz empty mapping response, Chat save-state, Search/RAG thread mutation, and Search primary-action reachability regressions are covered.
- Phase 2 is merged: Chat has provider readiness and first-run orientation coverage.
- Phase 3 clear/dismiss is merged: staged Chat context can be cleared before send, sent context cards are retained, and source handoff focused tests remain in place.
- Phase 6 startup-log polish is partially merged: splash import, optional OpenAI TTS mapping fallback, and NLTK falsy-download logging regressions are covered.

Current source state plus this branch:

- Phase 4 still needs broader live smoke replay and any non-handoff source actions where policy state can still block a visible action.
- Phase 5 top-level IA labels are merged: visible navigation now uses `Library`, `Models`, and `Speech` while preserving route IDs.
- Phase 5 Media empty-state copy is merged: Media now points users toward Ingest and selected-item requirements for analysis/save/export.
- Phase 5 Study empty-state cleanup is merged: flashcards and quizzes now separate no-content guidance for global/local vs workspace scopes while preserving backend-unavailable states.
- Phase 5 Search empty-state cleanup is merged: initial and zero-result panes now provide visible guidance for plain search, RAG collections, and Chat handoff flow.
- Phase 5 Notes empty-state cleanup is merged: local, server, and workspace scopes now provide visible creation/import routes.
- Phase 5 CCP/Library empty-state cleanup is merged: conversations, characters, personas, prompts, dictionaries, and world/lore books now explain creation/import routes and their relationship to Chat.
- Phase 5 Chatbooks empty-state cleanup is merged: empty Chatbooks explains portable context packs, import/create recovery, Chat reuse, and shared-navigation escape.
- Phase 4 handoff recovery copy is merged through PR #159: Notes, Media, RAG Search, and Web Search explain how to recover when the Chat handoff surface is unavailable.
- Phase 4 invalid-selection hardening is merged through PR #160: Notes and workspace source/artifact `Use in Chat` actions are disabled until a valid selection exists and expose tooltip recovery copy.
- Phase 5 command-palette label consistency is merged through PR #161: command-palette tab navigation uses the same `Library`, `Models`, `Speech`, and `Settings` labels as top-level navigation while preserving route IDs.
- Phase 5 main-navigation tooltip copy is merged through PR #162: compact top-level labels expose concise descriptions of each destination without changing route IDs.
- Phase 6 Speech/STTS capability-state copy is merged through PR #163: local Text-to-Speech and Speech Recognition dependency gaps are visible before failed actions.
- Phase 5/6 Web Search disabled-state copy is merged through PR #164: missing Web Search optional dependencies render as a real disabled action with recovery copy, not a clickable-looking no-op.
- Phase 5 Media Source disabled-action copy is merged through PR #165: disabled source sync/save/upload actions expose mode, selection, or archive-support recovery reasons.
- Phase 5 Study Quiz disabled-action copy is merged through PR #166: disabled quiz edit/start/load/submit actions expose scope or active-attempt recovery reasons.
- Phase 5 Study Flashcards disabled-action copy is merged through PR #167: disabled flashcard create/start/move/delete actions expose scope, selection, target deck, or server-mode recovery reasons.
- Phase 5 Media Viewer action-tooltip copy is merged through PR #168: Media Viewer `Use in Chat` and `Save for Later` actions expose selection/capability recovery reasons.
- Phase 5 Media Analysis action-tooltip copy is merged through PR #169: Media analysis save/note/edit/overwrite/delete actions expose generated-analysis and saved-version recovery reasons.
- Phase 5 Media Highlight action-tooltip copy is merged through PR #170: Media reading-highlight update/delete actions expose selected-highlight recovery reasons.
- Phase 5 Media Analysis navigation-tooltip copy is merged through PR #171: Media analysis previous/next version navigation explains empty, first-version, and last-version states.
- Phase 5 Search/RAG saved-search action recovery is merged through PR #172: saved-search Load/Delete actions explain selection requirements and selected searches repopulate the Search/RAG controls.
- Phase 5 Media list pagination-tooltip copy is merged through PR #173: Media result pagination explains single-page, first-page, middle-page, and last-page navigation states.
- Phase 5 Media multi-item review action-tooltip copy is merged through PR #174: batch Generate/Cancel analysis actions explain no-selection, selected, and in-progress states.
- Phase 4/7 handoff first-send smoke coverage is merged through PR #175: a mounted Textual test stages a handoff into a real Chat tab and verifies the first-send path applies staged context before marking the payload sent.
- Phase 4 invalid-selection smoke coverage is merged through PR #176: mounted Textual tests press empty Notes, workspace source/artifact, and Media handoff actions and verify they expose recovery copy without staging Chat.
- Phase 4 source-handoff replay smoke coverage is merged through PR #177: mounted Textual tests replay selected Notes, workspace details/note/source/artifact, Media, RAG Search, and Web Search handoffs into the app-owned Chat seam.
- Phase 4 backend-contract card copy is merged through PR #178: staged Chat context cards surface dry-run sync diagnostics and unsupported-action recovery messages without implying write sync.
- Phase 4 Notes/Workspace source capability-contract smoke coverage is merged through PR #179: Notes/Workspace handoff buttons consume runtime-policy denial state and expose recovery copy without staging Chat.
- Phase 4 Media handoff policy-smoke is merged through PR #180: Media `Use in Chat` consumes runtime-policy denial state and exposes recovery copy without staging Chat.
- Phase 4 RAG handoff policy-smoke is merged through PR #181: server-owned RAG result `Use in Chat` consumes runtime-policy denial state and exposes recovery copy without staging Chat.
- Phase 4 Web Search handoff policy-smoke is merged through PR #182: dedicated Web Search result `Use in Chat` consumes runtime-policy denial state and exposes recovery copy without staging Chat.
- Phase 4 source-policy handoff smoke closeout is merged through PR #183: mounted smoke coverage replays policy-blocked Media, RAG, and Web Search handoffs through real button events.
- Phase 6 Search/RAG dependency action-state is merged through PR #184: missing embeddings/RAG dependencies disable the primary Search action and expose install recovery copy.
- Phase 5 Study dashboard resume-tooltip is merged through PR #185: the disabled Resume action explains that no study session exists and points users to flashcards/quizzes.
- Phase 5 Study quiz start-tooltip is merged through PR #186: the shell-level Start quiz action explains no-quiz, select-quiz, unavailable-scope, active-attempt, and start-ready states.
- Phase 5 Study quiz Review in Chat is active in `codex/ux-quiz-review-chat-handoff`: the shell-level Review in chat action fails closed until a selected quiz and Chat handoff seam exist, then stages quiz context into Chat.
- Phase 6 still needs any remaining optional dependency gaps represented as user-facing capability states where relevant.
- Phase 7 still needs end-to-end audit replay on a clean home/config.

Planning consequence: remaining implementation should target broader live source-handoff replay cases, any non-handoff source actions where policy state can still block a visible action, any remaining compact-label/descriptive copy outside top navigation and command-palette tab navigation, disabled-state consistency beyond the already-covered Web Search, Media Source, Study Quiz, Study Flashcard, Study dashboard resume, Study quiz start, Study quiz Review in Chat, Media Viewer, Media Analysis, Media Highlight, Media Analysis navigation, Search/RAG saved-search actions, Media list pagination, and Media multi-item review actions, Phase 6 capability-state presentation beyond Speech/STTS, Web Search, and Search/RAG embeddings where user-relevant, and Phase 7 replay. Do not rebuild already-merged Chat handoff architecture.

## Issues Covered

- P0: Chatbooks is a navigation trap because it does not mount the shared global navigation shell.
- P0: Ingest could crash from an invalid ingestion source `Select` value. Current `dev` appears to fix the option/default mismatch; keep a direct regression.
- P0/P1: Study Quizzes can crash when the local quiz service returns an empty `{"items": [], "count": 0}` payload.
- P1: Chat first-run readiness is weak for providers that require API keys.
- P1: Notes manual save can leave dirty/title state untrustworthy.
- P1: Chat state-save can log a `log_selectors` unbound error when navigating away.
- P1: Search/RAG collection refresh can mutate Textual UI from a thread worker and log `active_app` context errors.
- P1: The chat-first `Use in Chat` handoff seam is implemented on current `dev`; remaining work is live smoke, clear/dismiss affordance, and disabled-state/recovery hardening.
- P1/P2: Search primary actions can be difficult to reach in the current viewport/layout.
- P2: First-run Chat is dense and underspecified for new users.
- P2: Empty and disabled states lack consistent reasons and recovery actions across Study, Media, Search, Notes, CCP/Personas, and Chatbooks.
- P2: Product labels still expose jargon such as `S/TT/S` and `LLM`; the main nav already uses `Library` for CCP.
- P2: Startup/log output reports optional or fallback conditions too noisily or inaccurately, including splash missing typing imports and misleading NLTK download logging.

## Non-Goals

- No broad visual redesign.
- No modal onboarding wizard.
- No multi-select handoffs.
- No automatic write sync, queued mutation replay, or local CRUD for remote-only domains.
- No server-only feature implementation unless needed to render honest unavailable states.
- No deprecation/removal of legacy route IDs; labels may change while route IDs stay stable.

## Design Constraints

- Chat remains the primary agentic programming/control surface.
- Notes, Workspaces, Media, Search, Study, personas, flashcards, quizzes, Chatbooks, and handoffs remain visible in the product model.
- Power users keep dense controls and shortcuts; beginner orientation is compact, skippable, and inline.
- Backend/source authority comes from `UX_Interop`, `runtime_policy`, and sync dry-run contracts, not from screen-local inference.
- Every phase uses TDD: failing regression first, smallest safe fix, focused verification.

## Remaining PR Sequence

1. `ux-smoke-shell-escape`
2. `runtime-layout-stability`
3. `chat-readiness-orientation`
4. `chat-handoff-closeout`
5. `empty-states-ia-language`
6. `startup-polish-audit-closeout`
7. `final-audit-replay`

Merged into current `dev`: `chat-handoff-core`, `source-handoff-actions`, and the parity-contract handoff documentation alignment.

Each PR should be independently shippable and should leave the app more usable than before.

## Phase 0: UX Smoke Harness And Shell Escape

Purpose: make the audit repeatable first, then fix the P0 navigation trap.

Branch state: completed in `codex/ux-audit-phase0-phase1`. `ChatbooksScreen` now uses `BaseAppScreen`, a routed-screen contract covers primary routes, and a Chatbooks smoke test catches missing shared navigation.

### Files

- Create: `Tests/UI/test_ux_audit_smoke.py`
- Modify: `Tests/UI/test_screen_navigation.py`
- Modify: `Tests/UI/test_chatbooks_screen_server_actions.py`
- Modify: `tldw_chatbook/UI/Screens/chatbooks_screen.py`
- Audit: all top-level screens in `tldw_chatbook/UI/Screens/`

### Steps

- [x] Add a Textual smoke test that proves Chatbooks keeps shared navigation mounted.
- [x] Make the smoke test fail on missing global nav after entering Chatbooks.
- [x] Add a routed-screen contract test: every primary route uses `BaseAppScreen` or has an explicit documented standalone exception.
- [x] Convert `ChatbooksScreen` from raw `Screen` to `BaseAppScreen`.
- [x] Move `ChatbooksWindowImproved` into `compose_content()` so global navigation remains mounted.
- [x] Add Chatbooks-specific regression: assert `#nav-chat` and `#nav-chatbooks` exist with the Chatbooks window.
- [x] Verify Chatbooks server/local action cards still render inside the shared shell.

### Acceptance Criteria

- [x] Chatbooks no longer traps mouse users.
- [x] All primary routes expose shared navigation through the routed-screen contract.
- [x] The UX smoke harness catches missing global nav before implementation work proceeds.
- [x] Focused tests pass: `Tests/UI/test_ux_audit_smoke.py`, `Tests/UI/test_screen_navigation.py`, `Tests/UI/test_chatbooks_screen_server_actions.py`.

## Phase 1: Runtime And Layout Stability

Purpose: remove runtime errors and unblock core surfaces before adding new UX behavior.

Branch state: completed in `codex/ux-audit-phase0-phase1`. Ingest has direct default coverage, Study normalizes empty mapping list responses, Chat save-state avoids the direct-log `log_selectors` bug, Search/RAG applies collection UI updates on the Textual thread, and Search primary-action reachability is covered.

### Files

- Modify: `tldw_chatbook/Widgets/Media/media_ingestion_source_panel.py`
- Modify: `Tests/UI/test_ingestion_ui_redesigned.py`
- Modify: `Tests/UI/test_media_ingestion_tab_integration.py`
- Modify: `tldw_chatbook/Study_Interop/quiz_scope_service.py`
- Modify: `Tests/Study_Interop/test_quiz_scope_service.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `Tests/UI/test_chat_screen_state.py`
- Modify: `tldw_chatbook/UI/Views/RAGSearch/search_rag_window.py`
- Modify: `Tests/UI/test_search_rag_window.py`

### Steps

- [x] Add a direct Ingest regression for server/local mount with the default source type.
- [x] Fix the source type options/default so `local_directory` is valid when selected.
- [x] Add a failing quiz-scope regression for empty mapping responses.
- [x] Normalize quiz list responses by explicit shape: `None`, mapping with `items`, or iterable records.
- [x] Add a failing Chat state-save regression where primary chat-log lookup succeeds and fallback selectors are not needed.
- [x] Define fallback log selectors before use and avoid the `log_selectors` unbound path.
- [x] Add a failing RAG collection refresh regression that asserts collection loading does not touch Textual widgets.
- [x] Keep `query_one`, `clear`, `append`, and `set_options` on the Textual message thread.
- [x] Add a separate Search layout/clickability regression for the primary Search button in the default viewport.

### Acceptance Criteria

- [x] Ingest mounts in local and server modes without `InvalidSelectValueError` in the existing focused test suite.
- [x] A direct source-panel regression asserts the default source `Select` value remains valid.
- [x] Study Quizzes handles empty local quiz lists without worker failure.
- [x] Chat navigation/state save no longer logs the `log_selectors` unbound error.
- [x] Search/RAG collection refresh no longer mutates UI from a worker thread.
- [x] Search primary action remains reachable/clickable in the standard audit viewport.

## Phase 2: Chat Destination Readiness And First-Run Orientation

Purpose: make Chat understandable and runnable before handoffs start landing users there.

Branch state: completed in `codex/ux-chat-readiness-orientation`.

### Files

- Create: `tldw_chatbook/Chat/provider_readiness.py`
- Create: `Tests/Chat/test_provider_readiness.py`
- Modify: `tldw_chatbook/UI/Chat_Window_Enhanced.py`
- Modify: `tldw_chatbook/Event_Handlers/Chat_Events/chat_events.py`
- Create or modify: `Tests/UI/test_chat_first_run_orientation.py`

### Steps

- [x] Add provider readiness unit tests for key-required and keyless/local providers.
- [x] Implement a side-effect-free `ProviderReadiness` helper that checks config and environment values.
- [x] Reuse the helper for send-time missing-key handling so pre-send and post-send messages stay consistent.
- [x] Add a compact Chat empty-state orientation strip: what Chat is for, current provider readiness, source/context entry points, and `Ctrl+P`.
- [x] Keep power-user controls available; collapse or summarize only the most advanced first-run-only noise.
- [x] Add a regression that Chat first-run exposes provider readiness before the first failed send.

### Acceptance Criteria

- [x] Chat shows whether the selected provider is ready before Send.
- [x] Missing API-key send-time error remains actionable.
- [x] Chat explains that Notes, Media, Search, Workspaces, Study, and personas can feed context into Chat.
- [x] No modal onboarding is introduced.

## Phase 3: Chat Handoff Core Closeout

Purpose: verify and harden the destination-side contract that has now landed in current `dev`.

Branch state: clear/dismiss behavior completed in `codex/ux-chat-handoff-clear-context`. Remaining work is live smoke replay and shared disabled/recovery language.

### Files

- Current: `tldw_chatbook/Chat/chat_handoff_models.py`
- Modify: `tldw_chatbook/app.py`
- Modify: `tldw_chatbook/Chat/chat_models.py`
- Modify: `tldw_chatbook/Chat/tabs/tab_state.py`
- Modify: `tldw_chatbook/Chat/tabs/chat_tab_container.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Current: `tldw_chatbook/Widgets/Chat_Widgets/chat_handoff_card.py`
- Extend: `Tests/UI/test_chat_first_handoffs.py`
- Extend: `Tests/UI/test_chat_screen_state.py`
- Extend: `Tests/UI/test_chat_tab_container.py`

### Steps

- [x] Add failing serialization tests for `ChatHandoffPayload`.
- [x] Include runtime backend, source owner, source selector state, active server profile, workspace ID, unsupported reports, sync dry-run diagnostics, and metadata.
- [x] Add persistence guardrails: JSON/TOML-safe normalization, secret redaction, body length caps, `body_truncated`, and `content_ref`.
- [x] Add app-owned `open_chat_with_handoff(payload)` and `pending_chat_handoff`.
- [x] Add failing tests that handoff opens a fresh Chat tab with `conversation_id=None`.
- [x] Chat consumes pending handoff after normal restore and clears it only after successful application.
- [x] Render a visible `Context staged` handoff card.
- [x] Prefill the draft prompt but do not auto-send.
- [x] Inject staged context into the first user send in an auditable way.
- [x] Add clear/undo behavior for staged context before send.
- [x] Fail closed when chat tabs are explicitly disabled.
- [x] Replay handoff creation and first send in the Phase 0/7 smoke harness.

### Acceptance Criteria

- [x] Handoff-created sessions always use a fresh Chat tab.
- [x] Context is visibly staged and not labeled as sent before Send.
- [x] The first user send includes staged source content in focused tests.
- [x] Large bodies are capped and truthfully marked truncated.
- [x] Local/server/workspace state remains visible and source-honest in payload/card tests.
- [x] Users can clear or dismiss staged context before Send.
- [x] The behavior passes live Textual smoke replay, not only unit/widget tests.

## Phase 4: Source Handoff Surfaces Closeout

Purpose: verify and harden the source-side handoff surfaces that have now landed in current `dev`.

Current-dev state: Notes, Workspace details/notes/sources/artifacts, Media, RAG Search, and dedicated Web Search handoffs are implemented and covered by focused tests. Notes, workspace source/artifact, and Media invalid-selection states now have shared-harness smoke coverage. Valid source handoffs from mounted source surfaces replay into the app-owned Chat seam. Backend-provided dry-run and unsupported-action messages are visible on staged Chat context cards. Runtime-policy-denied Notes/Workspace handoff smoke coverage is merged through PR #179. Media handoff policy-denial recovery is merged through PR #180. RAG handoff policy-denial recovery is merged through PR #181. Web Search handoff policy-denial recovery is merged through PR #182. The active branch closes mounted smoke coverage for policy-blocked Media, RAG, and Web Search `Use in Chat`. Remaining work is to close remaining source-action capability edge cases in the shared UX smoke pass.

### Files

- Modify: `tldw_chatbook/UI/Screens/notes_screen.py`
- Modify: `tldw_chatbook/UI/Notes_Window.py` or note sidebar/right panel modules as needed
- Modify: `tldw_chatbook/UI/MediaWindow_v2.py`
- Modify: `tldw_chatbook/Widgets/Media/media_viewer_panel.py` if present
- Modify: `tldw_chatbook/UI/Views/RAGSearch/search_result.py`
- Modify: `tldw_chatbook/UI/Views/RAGSearch/search_rag_window.py`
- Modify: `tldw_chatbook/UI/SearchWindow.py`
- Extend: `Tests/UI/test_notes_screen.py`
- Extend: `Tests/UI/test_media_handoffs.py`
- Extend: `Tests/UI/test_search_handoffs.py`

### Steps

- [x] Wire Notes local/server selected note `Use in Chat`.
- [x] Wire Workspace details, workspace note, source, and artifact `Use in Chat`.
- [x] Preserve dirty editor visible content in Notes handoff payloads.
- [x] Wire Media selected hydrated detail `Use in Chat`.
- [x] Wire RAG result card `Use in Chat` through a Textual message.
- [x] Cardify dedicated Web Search results enough to support `Use in Chat`.
- [x] Disable Notes and workspace item `Use in Chat` actions until a valid selected item exists, with tooltip recovery copy.
- [ ] Disable or explain source actions when server/auth/capability contracts block them.
- [ ] Consume backend-owned `UX_Interop` and `runtime_policy` contracts consistently; do not rebuild source authority from raw config.
- [ ] Keep dry-run sync reports diagnostic only; do not imply mirroring or write sync in source screens.
- [x] Replay each source handoff in the Phase 0/7 smoke harness.
- [x] Replay invalid-selection handoff states for Notes, workspace source/artifact, and Media in the Phase 0/7 smoke harness.
- [x] Surface backend dry-run sync and unsupported-action messages on staged Chat context cards without implying write sync.
- [x] Add mounted smoke coverage for runtime-policy-blocked Notes/Workspace handoff actions with recovery copy and no Chat staging.
- [x] Explain runtime-policy-blocked Media handoff actions with recovery copy and no Chat staging.
- [x] Explain runtime-policy-blocked server RAG result handoff actions with recovery copy and no Chat staging.
- [x] Explain runtime-policy-blocked Web Search result handoff actions with recovery copy and no Chat staging.
- [x] Replay runtime-policy-blocked Media, RAG, and Web Search handoff actions in mounted smoke coverage.

### Acceptance Criteria

- [x] Notes, Workspaces, Media, RAG Search, and Web Search can each stage one selected item into Chat in focused tests.
- [x] Notes and workspace item invalid-selection actions are disabled with recovery copy in mounted Textual tests.
- [x] No invalid selection navigates to Chat in live smoke coverage.
- [ ] Disabled handoff actions have a reason and a recovery path.
- [x] Workspace handoffs preserve `workspace_id` and isolation metadata in focused tests.
- [x] Web Search is in scope and not silently omitted.

## Phase 5: Empty-State, Disabled-State, And IA Language Cleanup

Purpose: make the app understandable without reducing expert efficiency.

Branch state: partially merged through PRs #152, #153, #154, #155, #156, #157, #158, #159, #160, #161, #162, #163, #164, #165, #166, #167, #168, #169, #170, #171, #172, #173, #174, #185, and #186. Top-level navigation labels now use `Library`, `Models`, and `Speech` while preserving route IDs, Media empty states now direct users to Ingest plus selected-item recovery actions, Study flashcard/quiz empty states distinguish no-content from unavailable runtime, Search/RAG empty states explain search modes, collections, and Chat handoffs, Notes empty states clarify local/server/workspace scope and creation/import routes, Library assets explain their Chat flow, Chatbooks clarifies portable context packs, blocked handoff recovery is clarified, invalid source-selection handoff controls are disabled with recovery copy, command-palette tab navigation aligns with the same IA names, compact main-navigation labels expose explanatory tooltips, Speech/STTS exposes local dependency capability states, Web Search missing dependencies render as disabled-state recovery copy, Media Source disabled actions expose recovery tooltips, Study Quiz disabled actions expose recovery tooltips, Study Flashcards disabled actions expose recovery tooltips, Study dashboard Resume action recovery is merged through PR #185, Study quiz Start action recovery is merged through PR #186, Study quiz Review in Chat recovery/handoff is active in `codex/ux-quiz-review-chat-handoff`, Media Viewer actions expose selection/capability tooltips, Media Analysis actions expose generated-analysis/saved-version tooltips, Media Highlight actions expose selected-highlight tooltips, Media Analysis navigation actions expose saved-version boundary tooltips, Search/RAG saved-search actions expose selection/reuse recovery, Media list pagination exposes result-page boundary tooltips, and Media multi-item review actions expose generation/cancellation state tooltips.

### Files

- Modify: `tldw_chatbook/UI/Navigation/main_navigation.py`
- Modify: `tldw_chatbook/UI/Chat_Window_Enhanced.py`
- Modify: `tldw_chatbook/UI/MediaWindow_v2.py`
- Modify: `tldw_chatbook/UI/Views/RAGSearch/search_rag_window.py`
- Modify: `tldw_chatbook/UI/Screens/notes_screen.py`
- Modify: `tldw_chatbook/UI/Screens/study_screen.py`
- Modify: `tldw_chatbook/UI/Study_Window.py`
- Modify: `tldw_chatbook/UI/Screens/ccp_screen.py`
- Extend: related UI tests for each screen touched

### Steps

- [ ] Standardize disabled primary actions: disabled reason, next action, and no silent no-op.
- [x] Study empty states distinguish no decks/quizzes from unavailable runtime.
- [x] Media empty states point to Ingest and explain when analysis/save/export need a selected item.
- [x] Search empty states explain plain search vs RAG, collections, and Chat handoff flow.
- [x] Notes empty states clarify local/server/workspace scope and creation/import routes.
- [x] CCP/Library empty states explain personas, characters, prompts, dictionaries, and how they relate to Chat.
- [x] Chatbooks empty state keeps portable knowledge-pack explanation and retains escape navigation.
- [x] Rename top-level navigation jargon while preserving route IDs: `CCP` -> `Library`, `LLM` -> `Models`, and `S/TT/S` -> `Speech`.
- [x] Align command-palette tab navigation with current top-level labels while preserving route IDs.
- [x] Add main-navigation tooltips that explain compact destination labels without changing route IDs.
- [x] Render missing Web Search dependencies as a disabled nav action with tooltip and pane recovery copy.
- [x] Add disabled-action recovery tooltips for Media Source sync/save/upload actions when server mode, selected source, or archive support is missing.
- [x] Add disabled-action recovery tooltips for Study Quiz editing and attempt actions when scope is unavailable or an attempt is active.
- [x] Add disabled-action recovery tooltips for Study Flashcards create/start/move/delete actions when scope, deck, card, target deck, or server-mode deletion support is missing.
- [x] Add Media Viewer action tooltips for `Use in Chat` and `Save for Later` when selection or read-it-later capability is missing.
- [x] Add Media Analysis action tooltips for save, note, edit, overwrite, and delete states when generated analysis or saved versions are missing.
- [x] Add Media Reading Highlight action tooltips for update/delete states when no highlight is selected.
- [x] Add Media Analysis navigation tooltips for previous/next saved-version boundary states.
- [x] Add Search/RAG saved-search action recovery for Load/Delete selection states and selected-config reuse.
- [x] Add Media list pagination tooltips for single-page and first/last result-page states.
- [x] Add Media multi-item review action tooltips for Generate/Cancel analysis states.
- [x] Add Study dashboard Resume action tooltip copy for no-session and resumable-session states.
- [x] Add Study quiz Start action tooltip copy for no-quiz, select-quiz, scope-unavailable, active-attempt, and ready states.
- [x] Add Study quiz Review in Chat recovery copy and selected-quiz handoff through the app-owned Chat seam.
- [ ] Add tooltips or short descriptions where compact labels remain necessary outside top navigation.

### Acceptance Criteria

- [ ] A first-time user can identify what Chat, Notes, Media, Search, Study, personas, and Chatbooks are for from visible text.
- [ ] A power user still has direct access to dense controls and shortcuts.
- [x] Route IDs and saved navigation contracts remain stable for command-palette tab navigation.
- [x] Tests assert command-palette labels and core top-level product surfaces remain visible.
- [x] Tests assert top-level navigation descriptions for compact labels.

## Phase 6: Startup Capability And Log Polish

Purpose: make first-run diagnostics truthful and actionable without hiding real failures.

Branch state: partially completed in `codex/ux-startup-log-polish`, with Speech/STTS capability-state merged through PR #163, Web Search disabled-state merged through PR #164, and Search/RAG embeddings dependency action-state merged through PR #184. Splash import regressions, optional OpenAI TTS mapping fallback logging, and NLTK download-result logging are covered by focused tests. Broader optional dependency capability-state presentation remains open.

### Files

- Create: `Tests/Utils/test_startup_polish_regressions.py`
- Modify: `tldw_chatbook/config.py`
- Modify: `tldw_chatbook/Chunking/Chunk_Lib.py`
- Modify: `tldw_chatbook/Utils/Splash_Screens/classic/scrolling_credits.py`
- Modify: `tldw_chatbook/Utils/Splash_Screens/tech/terminal_boot.py`
- Optionally modify optional dependency/capability presentation helpers

### Steps

- [x] Add import tests for splash effects that previously failed on missing `Dict`.
- [x] Fix missing typing imports.
- [x] Add tests that missing optional OpenAI TTS mapping uses defaults without error-level startup noise.
- [x] Make NLTK download logging truthful: no success message when download fails.
- [x] Expose local Speech/STTS TTS and STT dependency gaps as an inline capability state before failed actions.
- [x] Expose missing Web Search optional dependencies as disabled-state and pane recovery copy.
- [x] Expose missing Search/RAG embeddings dependencies as a disabled primary Search action with install recovery copy.
- [ ] Present optional dependency gaps as capability states when user-relevant, not stack traces.
- [ ] Keep real crashes/error states visible.

### Acceptance Criteria

- [x] Splash effects import without missing typing-name errors.
- [x] Missing optional TTS mapping falls back quietly.
- [x] Failed NLTK download is not logged as success.
- [x] Speech/STTS tells users when local Text-to-Speech or Speech Recognition dependencies are unavailable and how to recover.
- [x] Search tells users when Web Search dependencies are unavailable and how to recover.
- [x] Search/RAG tells users when embeddings dependencies are unavailable and disables the primary Search action before failure.
- [ ] Optional dependency gaps tell users what capability is unavailable and how to recover.

## Phase 7: End-To-End Audit Replay And Closeout

Purpose: prove the plan actually fixes the observed UX failures.

Current-dev state: partially started. A mounted handoff first-send smoke test now covers Chat staging and send-time context application; the broader replay artifacts still need to be regenerated after Phases 0, 1, 2, 5, and 6 land, then rerun against each source workflow.

### Files

- Create or keep temporary: `/private/tmp/tldw-chatbook-ux-remediation-verify/`
- Optional create: `Tests/UI/test_first_run_audit_replay.py` if stable enough for CI

### Steps

- [ ] Run focused tests from every phase.
- [ ] Run `git diff --check`.
- [ ] Launch with a clean home/config and verify first-run Chat.
- [ ] Replay audited workflows: Chat readiness/send, Notes create/save/handoff, Ingest mount, Media search/handoff, Search/RAG/handoff, Study flashcards/quizzes, CCP persona chat, Chatbooks escape.
- [ ] Capture probe JSON and screenshots/log excerpts under `/private/tmp/tldw-chatbook-ux-remediation-verify/`.
- [ ] Document any remaining uncertainty, especially live server/API paths not covered by local probes.

### Acceptance Criteria

- [ ] P0 blockers are fixed.
- [ ] P1 workflow trust failures are fixed or have explicit follow-up tasks.
- [ ] P2 orientation and empty-state issues have consistent patterns across core surfaces.
- [ ] The repeatable smoke harness prevents regression of the audited workflows.

## Verification Matrix

| Area | Minimum Verification |
| --- | --- |
| Shell/navigation | `Tests/UI/test_ux_audit_smoke.py`, `Tests/UI/test_screen_navigation.py`, Chatbooks nav escape probe |
| Ingest | `Tests/UI/test_ingestion_ui_redesigned.py`, `Tests/UI/test_media_ingestion_tab_integration.py` |
| Study | `Tests/Study_Interop/test_quiz_scope_service.py`, Study dashboard/UI tests |
| Chat readiness/state | `Tests/Chat/test_provider_readiness.py`, `Tests/UI/test_chat_first_run_orientation.py`, `Tests/UI/test_chat_screen_state.py` |
| Handoffs | Already passing on current `dev`: `Tests/UI/test_chat_first_handoffs.py`, `Tests/UI/test_search_handoffs.py`, `Tests/UI/test_media_handoffs.py`, selected Notes handoff tests, `Tests/UI/test_chat_screen_state.py`, `Tests/UI/test_chat_tab_container.py`; still add live smoke replay and clear/dismiss coverage |
| Search/RAG | `Tests/UI/test_search_rag_window.py`, `Tests/UI/test_search_handoffs.py` |
| Notes | `Tests/UI/test_notes_screen.py`, notes handoff tests |
| Media | Media window/viewer tests, media handoff tests |
| Library | `Tests/Widgets/test_ccp_widgets.py`, `Tests/UI/test_ccp_handlers.py`, `Tests/UI/test_ccp_screen.py` |
| Startup polish | `Tests/Utils/test_startup_polish_regressions.py` |
| Final audit | Clean-home Textual probe with saved artifacts |

## Implementation Rules

- Start each issue with a failing regression.
- Do not combine unrelated phases in one PR.
- Do not silently hide capability failures; convert them into user-facing unavailable states.
- Do not infer backend/source truth from current screen state when `UX_Interop` or `runtime_policy` has the authoritative contract.
- Do not break route IDs while changing labels.
- Do not remove power-user controls to create beginner clarity; use progressive disclosure, summaries, and better defaults.

## Next Step

After this Study quiz Review in Chat slice, remaining Phase 4 work should focus on broader live audit replay or any non-handoff source actions where policy state can still block a visible action. The remaining Phase 5 work is any compact-label descriptions outside top navigation plus broader disabled primary-action consistency. Phase 6 should only add more capability states where a missing optional dependency blocks a visible user workflow. Phase 7 live replay remains the higher-risk workflow-completion follow-up.
