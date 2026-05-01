# UX Audit Remediation Plan

Date: 2026-05-01
Status: Current-dev rebaseline after intern handoff merge
Branch context: current `dev` / `origin/dev` at `61cd555d`
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

Verified on current `dev` at `61cd555d`:

- The intern landed the main `Use in Chat` implementation path: `ChatHandoffPayload`, app-owned `pending_chat_handoff`, fresh ephemeral Chat tabs with `conversation_id=None`, staged context cards, draft prefill, first-send context injection, source-specific Notes/Workspace/Media/RAG/Web Search handoffs, and persistence coverage.
- Focused handoff verification passed: `47 passed, 1 warning` across Chat handoff core, source handoffs, state persistence, and tab container tests.
- Existing non-handoff focused tests passed: `49 passed, 8 warnings` across Chatbooks server actions, screen navigation, Ingest, Study scope service, and Search/RAG. These tests do not yet cover every UX-audit failure listed below.
- Ingest source creation now uses `local_directory`, `archive_snapshot`, and `git_repository`, matching `ALLOWED_SERVER_CREATE_SOURCE_TYPES`; keep a direct regression so this does not drift.

Still open by current source inspection:

- `ChatbooksScreen` still extends raw `Screen`, not `BaseAppScreen`, so the global-navigation trap remains open until a routed-shell regression proves otherwise.
- `Tests/UI/test_ux_audit_smoke.py` does not exist yet.
- `tldw_chatbook/Chat/provider_readiness.py` does not exist yet, so Chat first-run readiness remains open.
- `QuizScopeService.list_quizzes()` still uses `list((records or {}).get("items") or records or [])`, so an empty mapping payload can still fall through to mapping keys instead of an empty item list.
- `ChatScreen._extract_and_save_messages()` still defines `log_selectors` only inside the fallback path, so the unbound save-state bug remains open when the direct chat-log lookup succeeds.
- `SearchRAGWindow._refresh_collections_list()` is still a `thread=True` worker that queries and mutates Textual widgets from the worker.
- Visible IA labels still include `LLM` and `S/TT/S`.
- Splash modules still reference `Dict` without importing it, and NLTK download logging can still report success after a falsy download result.

Planning consequence: Phases 3 and most of 4 are no longer greenfield implementation work. They become current-dev verification, hardening, and UX closeout. Phases 0, 1, 2, 5, 6, and 7 remain active.

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

Current-dev state: still open. Handoffs now send users into Chat, but provider readiness and first-run orientation have not landed.

### Files

- Create: `tldw_chatbook/Chat/provider_readiness.py`
- Create: `Tests/Chat/test_provider_readiness.py`
- Modify: `tldw_chatbook/UI/Chat_Window_Enhanced.py`
- Modify: `tldw_chatbook/Event_Handlers/Chat_Events/chat_events.py`
- Create or modify: `Tests/UI/test_chat_first_run_orientation.py`

### Steps

- [ ] Add provider readiness unit tests for key-required and keyless/local providers.
- [ ] Implement a side-effect-free `ProviderReadiness` helper that checks config and environment values.
- [ ] Reuse the helper for send-time missing-key handling so pre-send and post-send messages stay consistent.
- [ ] Add a compact Chat empty-state orientation strip: what Chat is for, current provider readiness, source/context entry points, and `Ctrl+P`.
- [ ] Keep power-user controls available; collapse or summarize only the most advanced first-run-only noise.
- [ ] Add a regression that Chat first-run exposes provider readiness before the first failed send.

### Acceptance Criteria

- [ ] Chat shows whether the selected provider is ready before Send.
- [ ] Missing API-key send-time error remains actionable.
- [ ] Chat explains that Notes, Media, Search, Workspaces, Study, and personas can feed context into Chat.
- [ ] No modal onboarding is introduced.

## Phase 3: Chat Handoff Core Closeout

Purpose: verify and harden the destination-side contract that has now landed in current `dev`.

Current-dev state: mostly landed and covered by focused tests. Remaining work is UX closeout: add a visible clear/dismiss path before send, replay in the smoke harness, and make disabled/recovery states consistent with Phase 5.

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
- [ ] Add clear/undo behavior for staged context before send.
- [x] Fail closed when chat tabs are explicitly disabled.
- [ ] Replay handoff creation and first send in the Phase 0/7 smoke harness.

### Acceptance Criteria

- [x] Handoff-created sessions always use a fresh Chat tab.
- [x] Context is visibly staged and not labeled as sent before Send.
- [x] The first user send includes staged source content in focused tests.
- [x] Large bodies are capped and truthfully marked truncated.
- [x] Local/server/workspace state remains visible and source-honest in payload/card tests.
- [ ] Users can clear or dismiss staged context before Send.
- [ ] The behavior passes live Textual smoke replay, not only unit/widget tests.

## Phase 4: Source Handoff Surfaces Closeout

Purpose: verify and harden the source-side handoff surfaces that have now landed in current `dev`.

Current-dev state: Notes, Workspace details/notes/sources/artifacts, Media, RAG Search, and dedicated Web Search handoffs are implemented and covered by focused tests. Remaining work is to close disabled-state, invalid-selection, and source-authority edge cases in the shared UX smoke pass.

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
- [ ] Disable or explain source actions when server/auth/capability contracts block them.
- [ ] Consume backend-owned `UX_Interop` and `runtime_policy` contracts consistently; do not rebuild source authority from raw config.
- [ ] Keep dry-run sync reports diagnostic only; do not imply mirroring or write sync in source screens.
- [ ] Replay each source handoff in the Phase 0/7 smoke harness.

### Acceptance Criteria

- [x] Notes, Workspaces, Media, RAG Search, and Web Search can each stage one selected item into Chat in focused tests.
- [ ] No invalid selection navigates to Chat in live smoke coverage.
- [ ] Disabled handoff actions have a reason and a recovery path.
- [x] Workspace handoffs preserve `workspace_id` and isolation metadata in focused tests.
- [x] Web Search is in scope and not silently omitted.

## Phase 5: Empty-State, Disabled-State, And IA Language Cleanup

Purpose: make the app understandable without reducing expert efficiency.

Current-dev state: still open. The main navigation now says `Library` for CCP, but visible `LLM` and `S/TT/S` jargon remain and the cross-surface disabled/empty-state pattern is not yet standardized.

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
- [ ] Study empty states distinguish no decks/quizzes from unavailable runtime.
- [ ] Media empty states point to Ingest and explain when analysis/save/export need a selected item.
- [ ] Search empty states explain plain search vs RAG, collections, and Chat handoff flow.
- [ ] Notes empty states clarify local/server/workspace scope and creation/import routes.
- [ ] CCP/Library empty states explain personas, characters, prompts, dictionaries, and how they relate to Chat.
- [ ] Chatbooks empty state keeps portable knowledge-pack explanation and retains escape navigation.
- [ ] Rename visible jargon while preserving route IDs: `S/TT/S` -> `Speech`; consider `LLM` -> `Models`; keep `Library` for CCP/persona surfaces.
- [ ] Add tooltips or short descriptions where compact labels remain necessary.

### Acceptance Criteria

- [ ] A first-time user can identify what Chat, Notes, Media, Search, Study, personas, and Chatbooks are for from visible text.
- [ ] A power user still has direct access to dense controls and shortcuts.
- [ ] Route IDs and saved navigation contracts remain stable.
- [ ] Tests assert the key user-facing labels and core product surfaces remain visible.

## Phase 6: Startup Capability And Log Polish

Purpose: make first-run diagnostics truthful and actionable without hiding real failures.

Branch state: partially completed in `codex/ux-startup-log-polish`. Splash import regressions, optional OpenAI TTS mapping fallback logging, and NLTK download-result logging are covered by focused tests. Broader optional dependency capability-state presentation remains open.

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
- [ ] Present optional dependency gaps as capability states when user-relevant, not stack traces.
- [ ] Keep real crashes/error states visible.

### Acceptance Criteria

- [x] Splash effects import without missing typing-name errors.
- [x] Missing optional TTS mapping falls back quietly.
- [x] Failed NLTK download is not logged as success.
- [ ] Optional dependency gaps tell users what capability is unavailable and how to recover.

## Phase 7: End-To-End Audit Replay And Closeout

Purpose: prove the plan actually fixes the observed UX failures.

Current-dev state: not started. The smoke/replay artifacts need to be regenerated after Phases 0, 1, 2, 5, and 6 land, then rerun against the already-merged handoff flows.

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

Start with Phase 0, then Phase 1. The intern work reduced the handoff backlog, but it did not remove the highest-impact blocker: losing global navigation after entering Chatbooks. After the shell and runtime fixes land, close out the already-merged handoff work with clear/dismiss behavior and live smoke replay instead of rebuilding the handoff architecture.
