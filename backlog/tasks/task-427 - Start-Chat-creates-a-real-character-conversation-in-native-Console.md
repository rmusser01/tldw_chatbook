---
id: TASK-427
title: Start Chat creates a real character conversation in native Console
status: Done
assignee: []
created_date: '2026-07-21 09:38'
updated_date: '2026-07-22 02:14'
labels:
  - roleplay
  - console
  - ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the RP/character-card UX review (Docs/superpowers/qa/rp-ux-review-2026-07-21/report.md). P0. Start Chat / Open in Console currently degrade to invisible "staged live work" (chat_screen.py:8490-8497): blank transcript, no greeting, no character identity on the session (no character_id; chat_screen.py:5795-5853), composer prefilled with a meta-instruction. Observed live: sending the prefilled text routed through the agent harness, spawned a sub-agent, and the model replied "Please provide the previous conversation and the character details you'd like me to use" - the staged context never reached it. The RP core loop is a dead end. Start Chat must produce a conversation that (a) shows the character greeting as the first assistant message, (b) is identified with the character (title + session identity), (c) applies the card system prompt/definition on the send path, and (d) sends via the plain provider path, not the agent harness, by default.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Start Chat on a character opens Console with the character greeting visible as the first assistant message
- [x] #2 The created conversation is titled/labelled with the character name and retains that identity across app restarts
- [x] #3 The first user send replies in character (card system prompt and definition applied) without the user re-describing the character
- [x] #4 Character sends do not route through the agent/sub-agent harness unless the user has explicitly enabled an agent profile
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Start Chat from the Roleplay workbench now opens a REAL character conversation in the native Console (was: invisible staged-context dead-end where the agent sub-agent asked the user to re-describe the character). No ChaChaNotes migration (conversations.character_id already existed at schema v22).

Delivered across 5 TDD tasks + review fixes:
1. character_id/character_name on ConsoleChatSession; persist_session_if_needed passes assistant_kind='character'+character_id when bound (console_chat_store.py).
2. Greeting is display-only to the provider: _provider_message_payloads drops leading pre-first-user assistant turns (avoids Anthropic/Gemini assistant-first 400), matching the preview; regenerate/continue guarded against a lone-greeting payload (console_chat_controller.py).
3. Plain-provider gate keyed on the message-OWNING session's character_id (race-free via session_id_for_message), so character sends never hit the agent harness; closed-session KeyError guard added (console_chat_controller.py).
4. Native handoff branch _start_character_console_session: inline asyncio.to_thread card fetch (get_character_card_by_id) with guarded fallback to staged-context, builds the effective system prompt like the preview (system_prompt+personality+description+scenario), creates a dedicated GLOBAL-workspace session titled 'Chat with {name}', seeds the greeting as the first ASSISTANT message, then _sync_native_console_chat_ui() so the card prompt reaches the FIRST send; post-seed sync guarded so a failure can't duplicate the conversation (chat_screen.py).
5. Restore character_id on resume so a resumed character conversation stays on the plain path (chat_screen.py).

Deferred (out of scope, per design §Scope): character-scoped dictionaries/world-info on native send; preview 'Open in Console' transcript continuation; per-session agent opt-in UI.

Design: Docs/superpowers/specs/2026-07-21-start-chat-character-conversation-design.md (rev 2, adversarial-reviewed). Plan: Docs/superpowers/plans/2026-07-21-start-chat-character-conversation.md.

Tests: store/controller/handoff suites green (227 in the 3 affected files); whole-branch opus review = ready to merge, no Critical/Important. LIVE-VERIFIED in the real TUI (all 4 ACs incl. across an app RESTART): greeting as first assistant message, 'Chat with {name}' title persisted, in-character reply with system prompt applied ([sys:on]), plain provider path with no sub-agent dead-end.

Files: tldw_chatbook/Chat/console_chat_store.py, tldw_chatbook/Chat/console_chat_controller.py, tldw_chatbook/UI/Screens/chat_screen.py + their tests.
<!-- SECTION:NOTES:END -->
