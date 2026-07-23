---
id: TASK-341
title: Fix silent loss of conversation rename made via the tab rename modal
status: Done
assignee:
  - '@claude'
created_date: '2026-07-20 14:21'
updated_date: '2026-07-21 03:34'
labels:
  - console
  - ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resumed 'Websocket reconnect strategy', clicked its active tab to open 'Rename Chat Tab', typed 'Reconnect deep dive', pressed Enter. Tab strip and the transcript header — the same header that identifies the conversation by title ('Transcript / Event Stream | Reconnect deep dive') — updated immediately, signalling the conversation was renamed. On next app start the rail lists 'Websocket reconne...' again; 'Reconnect deep' appears nowhere. The rename applied only to the ephemeral tab label, and tabs do not survive restarts, so the user's rename evaporates without any warning or scope hint beyond the modal title.

**Repro:** Open a saved conversation from the rail, click its (active) tab label once or twice until 'Rename Chat Tab' opens, type a new name, Enter. Observe tab + header update. Restart the app: rail still shows the old title.

**Verifier note:** Code-verified: ConsoleChatStore.rename_session (console_chat_store.py:253) only mutates the in-memory session title; _apply_rename (chat_screen.py:1080-1100) never writes the persisted conversation title, yet the transcript header (set_session_title from session.title) confirms the rename as if conversation-level. Modal is labeled 'Rename Chat Tab' but the tab IS the conversation under resume-opens-new-tab semantics, and the rename is silently discarded on restart. No ledger item or backlog task covers persisting renames; silent loss of confirmed user input justifies P1.

**Source:** Console UX expert review 2026-07-20 (finding j2-rename-tab-only-silently-lost; P1, verdict NEW, high confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J2 returning power user journey. Evidence: `j2-51-rename-modal.png`, `j2-52-rename-typed.png`, `j2-53-after-rename-enter.png`, `j2-55-final-boot-rail.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Renaming the tab of a saved conversation should rename (or offer to rename) the conversation, or the modal must state that the change is tab-only and temporary
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add update_conversation_title to ConsoleChatPersistence protocol + ChatPersistenceService (mirror update_conversation_system_prompt: version-checked db.update_conversation)
2. rename_session persists when the session has a persisted_conversation_id, returns (session, persisted) like the system-prompt seam; in-memory title always applied
3. _apply_rename warns when persistence fails
4. TDD store-level (fake persistence recorder) + UI-level modal flow
<!-- SECTION:PLAN:END -->

## Implementation Notes

Renaming now persists: `ConsoleChatStore.rename_session` returns
`(session, persisted)` and, for sessions with a `persisted_conversation_id`,
calls a new `update_conversation_title` persistence seam (protocol method +
`ChatPersistenceService` implementation mirroring
`update_conversation_system_prompt`: version-checked `db.update_conversation`).
The in-memory rename is always applied; a persistence failure logs and
returns `persisted=False`, and `_apply_rename` warns the user that the
stored conversation kept its old name. Unsaved sessions are unchanged
(title flows into `create_conversation` at first persist). The controller's
auto-title path picks up persistence for free.

Verified: 3 store-level tests (recorder/exploding/unsaved fakes) + a
UI-level modal-flow test (`Tests/UI/test_console_rename_persistence.py`),
reproduced RED first; live served-app check — renamed via the real tab
modal, restarted the app, rail shows the new title. Files:
`Chat/console_chat_store.py`, `Chat/chat_persistence_service.py`,
`UI/Screens/chat_screen.py`.
