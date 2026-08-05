---
id: TASK-504
title: >-
  Fix #chat-right-sidebar QueryError silently aborting conversation load in Chat
  tab
status: Done
assignee: []
created_date: '2026-07-24 01:29'
updated_date: '2026-07-25 03:30'
labels:
  - tech-debt
  - dead-code
  - chat
  - bug
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
display_conversation_in_chat_tab_ui() (Event_Handlers/Chat_Events/chat_events.py) queries app.query_one("#chat-right-sidebar") unconditionally while populating a loaded conversation into the Chat tab. That id has not existed in the live compose tree since ChatWindowEnhanced replaced the legacy ChatWindow (right sidebar functionality moved into settings_sidebar) -- discovered during task-412's chat_right_sidebar.py deletion audit. The query is wrapped by a broad try/except QueryError that swallows the exception and shows a generic 'Error updating UI for loaded chat.' notification, but because the query sits partway through the population logic, everything after it (conversation title, keywords, UUID display, and the full message log) never gets populated when it fires. Users loading a saved conversation in the live Chat tab may be seeing an empty/stale chat log today.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Loading a saved conversation through display_conversation_in_chat_tab_ui populates the live sidebar title (#chat-chat-title), the conversation id display (#chat-chat-id), and the full message log without hitting the #chat-right-sidebar QueryError path (keywords dropped: no live keywords field exists -- user-approved amendment 2026-07-24)
- [x] #2 The character-detail-edit fields this function tries to populate (chat-character-name-edit etc.) are either restored somewhere reachable in the live UI, or the dead write attempt is removed without regressing conversation loading
- [x] #3 A regression test loads a conversation through display_conversation_in_chat_tab_ui (or its tab-aware wrapper) against a live-shaped widget tree (no #chat-right-sidebar) and asserts the message log and title actually populate
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Amend AC #1 to name the live sidebar ids (#chat-chat-title / #chat-chat-id) and drop the keywords requirement (user-approved, no live keywords field exists).
2. Write three failing regression tests in Tests/Event_Handlers/Chat_Events/test_chat_events.py against display_conversation_in_chat_tab_ui: (a) live sidebar + log population, (b) dead #chat-right-sidebar/#chat-conversation-*/#chat-character-*-edit families never queried, (c) a missing sidebar field does not abort the message-log mount.
3. Live-shape Tests/fixtures/event_handler_mocks.py: remove the #chat-right-sidebar bare mock and its wiring block, the three #chat-conversation-* entries, and the six #chat-character-*-edit entries; add #chat-chat-title / #chat-chat-id Input mocks.
4. Run the new tests to confirm they fail against the unmodified production code (RED), and capture the blast radius against the rest of the shared-fixture consumer suites.
5. Repair display_conversation_in_chat_tab_ui: delete the #chat-right-sidebar block and its six character-edit writes (no live equivalents); repoint conversation-detail writes to #chat-chat-title / #chat-chat-id behind a scoped try/except QueryError (keywords write dropped); scope-guard the #chat-system-prompt writes the same way; repoint the not-found branch to the same live ids; move the closing info log inside the try success path so it no longer fires when the except path ran.
6. Clean the twin handle_chat_clear_active_character_button_pressed: remove the dead #chat-right-sidebar query and its six field writes; keep the live #chat-system-prompt reset and make the success notify unconditional.
7. Update Chat/tabs/tab_context.py GLOBAL_WIDGETS and the chat_events_tabs.py wrapper's #chat-conversation-title-input read to the new live ids; update the matching test_chat_events_tabs.py pins.
8. Run the full affected suites (Tests/Event_Handlers/Chat_Events/, plus every consumer of Tests/fixtures/event_handler_mocks.py) and apply the blast-radius rule: restore any dead id whose removal broke a pre-existing test of an out-of-scope legacy handler, with a DEAD-ID comment.
9. Update this task file (AC checkboxes, plan, notes, status) and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Repaired display_conversation_in_chat_tab_ui() and its twin handle_chat_clear_active_character_button_pressed() in Event_Handlers/Chat_Events/chat_events.py, which both unconditionally queried the dead #chat-right-sidebar container (removed from the live compose tree since ChatWindowEnhanced replaced the legacy ChatWindow, task-412) followed by six #chat-character-*-edit fields with no live equivalents. Because the query sat mid-population inside one broad try/except QueryError, everything downstream (title, keywords, UUID, and the full message log) silently never populated.

Approach:
- Deleted the #chat-right-sidebar block and its six character-edit writes outright (no live equivalents to relocate to, per user-approved scope).
- Repointed the conversation-detail writes to the live EnhancedSettingsSidebar ids #chat-chat-title and #chat-chat-id (id_prefix="chat"), wrapped in their own scoped try/except QueryError so a missing sidebar field warns and continues instead of aborting the message-log mount. The keywords write was dropped entirely -- no live keywords field exists (AC #1 amended, user-approved).
- Wrapped the three #chat-system-prompt writes (character-branch, character-load-failure branch, no-character branch) in a small `_safe_set_system_prompt` local helper using the same scoped-guard pattern.
- Repointed the not-found branch's dead-id writes to the same #chat-chat-title/#chat-chat-id pair behind the same guard.
- Moved the closing "Displayed conversation ..." info log inside the try's success path so it no longer logs success when the outer except ran.
- Cleaned handle_chat_clear_active_character_button_pressed the same way: removed the dead #chat-right-sidebar query and six field writes; the "Active character cleared." notify is now unconditional after the #chat-system-prompt reset (its own try/except is untouched).
- Repointed Chat/tabs/tab_context.py's GLOBAL_WIDGETS and the chat_events_tabs.py wrapper's post-load title read from the dead #chat-conversation-title-input to #chat-chat-title; updated the corresponding test_chat_events_tabs.py pins (including the skipped edge-case tests, for consistency).
- Live-shaped Tests/fixtures/event_handler_mocks.py: removed the bare #chat-right-sidebar mock and its query_one wiring block, the three #chat-conversation-* entries, and the six #chat-character-*-edit entries; added #chat-chat-title / #chat-chat-id Input mocks.

Blast radius (spec's documented rule applied): removing the six #chat-character-*-edit ids broke a pre-existing test of a different, out-of-scope legacy handler -- test_handle_chat_load_character_with_greeting, which exercises handle_chat_load_character_button_pressed (not one of this task's two target functions; queries the same six ids at chat_events.py:4645-4686). Restored those six ids in the fixture with a `# DEAD-ID: not in live tree; kept for legacy-handler test -- see follow-up task` comment. The #chat-right-sidebar and #chat-conversation-* removals had no other consumers and needed no restoration. The two pre-existing task-442 T4 twins (test_display_conversation_substitutes_active_profile_name / test_display_conversation_keeps_users_name_without_active_profile) failed transiently once the fixture was live-shaped (old code still queried the now-gone #chat-right-sidebar) and self-healed once the production repair landed -- no test edits were needed for them.

Added three new regression tests next to the existing T4 tests: population against the live-shaped tree, absence of any #chat-right-sidebar/#chat-conversation-*/#chat-character-*-edit query, and a forced #chat-chat-title QueryError that must not block the message-log mount. Verified RED (all three failing) against the unmodified production code with the live-shaped fixture before implementing the repair.

Modified files: tldw_chatbook/Event_Handlers/Chat_Events/chat_events.py, tldw_chatbook/Event_Handlers/Chat_Events/chat_events_tabs.py, tldw_chatbook/Chat/tabs/tab_context.py, Tests/fixtures/event_handler_mocks.py, Tests/Event_Handlers/Chat_Events/test_chat_events.py, Tests/Event_Handlers/Chat_Events/test_chat_events_tabs.py.
<!-- SECTION:NOTES:END -->
