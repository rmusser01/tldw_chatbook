# task-562 — Retire the Chat-tab conversation-entry chain — Design

**Date:** 2026-07-25
**Branch:** `claude/task-562-chat-tab-conversation-entry` off dev `0ef4ee638` (#861 merge).
**Status:** design approved by user (decision: "Retire, task scope"); spec pending user review.

## The decision (AC #1)

The Chat tab has been the **native Console** since commit `8ea71071f` (2026-05-06) removed `yield self._ensure_chat_window()` from `ChatScreen.compose_content`. Scout-verified (2026-07-25):

- `ChatWindowEnhanced` is instantiated only in `_ensure_chat_window()` (`chat_screen.py:2162-2164`), which has **zero callers**; `self.chat_window` is `None` for the process lifetime. `#chat-window`, `#chat-log`, `#chat-input`, and the whole `EnhancedSettingsSidebar` never exist in the live tree.
- The Console's conversation browser covers all ChaChaNotes conversations (global + workspace scopes, not character-gated — filter at `chat_screen.py:5395-5460` never sets `character_id`), resumes into native sessions, and auto-persists (no manual save concept). Every cross-screen "open in chat" flow targets the Console via `app.open_chat_with_handoff`.
- The `use_enhanced_window` config flag selects nothing; its Settings checkbox is a no-op for surface choice.

**Decision: retire the dead conversation-entry chain in favor of the Console.** Restoring would mean reversing the May migration AND duplicating the Console browser — rejected. A decision doc in `backlog/decisions/` (naming per existing files there) records this with the provenance above.

## Deletion set — gated units

**The gate (every unit, before deletion):** (a) every trigger id is composed nowhere outside retired/deleted modules; (b) zero direct Python callers outside the deleted set — actions, keybindings, routers, other handlers, not just button ids; (c) tests that call it are themselves in the deletion set. A unit failing its gate is NOT deleted — it is recorded in the follow-up task instead.

**Shared-helper rule:** only functions whose sole callers are inside the deleted set may go. Known keeps: `_update_console_session_title` (live Console callers), the reactives `current_chat_conversation_id` / `current_chat_is_ephemeral` / `current_chat_active_character_data` (read by live streaming/worker paths — only the dead *watcher methods* go).

**Same-commit reference stripping:** `Chat_Window_Enhanced.py` STAYS this cycle but references deleted symbols (core_handlers fallback map `:646-656`, sidebar-resize imports/usages `:670, 709-710, 1009-1011, 1015-1017`, right-sidebar toggle buttons `:577/:673`). Every unit's commit strips CWE's references to that unit so the branch never has dangling imports.

### Unit 1 — save/clone/load-selected handlers + display fn + tabs wrapper
- `chat_events.py`: `handle_chat_save_current_chat_button_pressed` (~:3183), `handle_chat_clone_current_chat_button_pressed` (~:3460), `handle_chat_load_selected_button_pressed` (~:3762), `display_conversation_in_chat_tab_ui` (~:4148), their `CHAT_BUTTON_HANDLERS` entries, their now-unused imports.
- `chat_events_tabs.py`: `display_conversation_in_chat_tab_ui_with_tabs` region (~:295-403 — verify neighbors first: the other wrappers writing `current_chat_conversation_id` at :145/:274 belong to different functions and are assessed by their own gates or kept).
- Honest accounting: this deletes the function repaired in #861 (task-504) and task-551's site (c). The repair was correct while the decision was open; the resolver and sites (a)/(b) keep their own tests untouched.

### Unit 2 — new-conversation + save-details handlers
- `chat_events.py`: `handle_chat_new_conversation_button_pressed` (~:3070), `handle_chat_save_details_button_pressed`-family (~:3612), their map entries. Gate especially for direct callers (an `action_new_chat`/keybinding calling the handler directly fails the gate → unit deferred).

### Unit 3 — conversation-search stack
- `chat_events.py` ~:3822-4147 (`handle_chat_conversation_search_bar_changed` and companions) + the `app.py` arms routing to them: `on_input_changed` conversation-search arms (~:9297-9317), `on_list_view_selected` arm (~:9444), `on_checkbox_changed` arm (~:9463), `on_select_changed` arm (~:9601). Gate: the Console browser (`_refresh_console_conversation_browser_search`) shares nothing with this stack.

### Unit 4 — character-load-into-sidebar family (transitively in scope via the fixture item)
- `handle_chat_load_character_button_pressed` + the character search/name-edit handlers behind dead ids, their `app.py` Input.Changed arms (~:9350/:9354), their tests (incl. `test_handle_chat_load_character_with_greeting`), and the fixture's `# DEAD-ID` character-edit mocks. This unit completes AC #3's fixture item honestly. If its gate fails (a live caller exists), the unit is deferred and the fixture mocks stay with an updated comment.
- Gated candidate: `handle_chat_clear_active_character_button_pressed` (the task-504-cleaned twin) — same dead surface if its trigger button is composed only in retired modules; gate decides, defer if ambiguous (it touches `#chat-system-prompt`, which borders the send-path surfaces kept this cycle).

### Unit 5 — whole-file retirements
- `Widgets/settings_sidebar.py` (1,314 LOC, zero importers), `Widgets/settings_sidebar_optimized.py` (367, zero importers), `Event_Handlers/Chat_Events/chat_events_sidebar_resize.py` (89, sole consumers in CWE — strip same-commit).

### Unit 6 — app.py dead watchers/branches
- `watch_current_chat_is_ephemeral` (~:6916-6960, guaranteed-QueryError body), `watch_chat_right_sidebar_collapsed`/`_width` (~:8605-8630), the `#chat-window` checkbox branch (~:9486-9496). Watcher METHODS deleted; reactives stay.

### Unit 7 — orphan CSS
- `#chat-right-sidebar` + dead-id rules in SOURCE files only: `Constants.py` CSS strings, `css/layout/_sidebars.tcss`, `css/features/_chat.tcss`, `css/components/_buttons.tcss`. **Never hand-edit `css/tldw_cli_modular.tcss` (generated bundle) — regenerate via `build_css.sh` and commit the rebuilt bundle.** The live `toggle-chat-right-sidebar` *button-id* CSS family is checked against whether the buttons themselves were removed from CWE (Unit 5 strips them); rules for still-composed ids stay.

## Retirement guards + tests

- Extend `Tests/UI/test_legacy_entrypoints_retired.py` (task-412 pattern): add the three retired module paths to `RETIRED_MODULES`/`RETIRED_FILES`; add symbol-absence pins (deleted handler names absent from `chat_events`, ids absent from `CHAT_BUTTON_HANDLERS`).
- **Test-casualty enumeration is done by grep, not by list**: the plan greps ALL of `Tests/` for every deleted symbol and id; the files named here are the known majority, not the closed set (known additional suspect: `test_chat_events_integration.py:278,528` queries `#chat-save-current-chat-button` on a real app).
- Delete test regions guarding deleted code: `test_chat_events.py` ~:277-1028 affected regions (save-chat test, T4 site-(c) region, task-504 region), `test_chat_events_tabs.py` display-wrapper tests (~:535-700 affected, ~:836), Unit-4 tests.
- **Deletion order rule:** callers before callees — a shared or internal helper is deleted only in or after the commit that removes its last caller, so no intermediate commit has dangling references.
- Prune `Tests/fixtures/event_handler_mocks.py`: `#chat-chat-title`/`#chat-chat-id` (their consumers go), DEAD-ID character-edit mocks (Unit 4), any id whose last consumer is deleted.
- Full affected suites after each unit's commit; `test_legacy_entrypoints_retired.py` green at the end; app boot smoke (`Tests/test_smoke.py`) unaffected.

## Kept this cycle (explicit boundary) → follow-up task

`Chat_Window_Enhanced.py`, `enhanced_settings_sidebar.py`, `UI/Chat_Modules/`, ten enhanced-window test suites, the `use_enhanced_window` flag + Settings checkbox, the chat_events send-path (`:792` family) liveness audit, and any unit whose gate fails. Filed as: **"Retire Chat_Window_Enhanced + enhanced_settings_sidebar (unmounted since 8ea71071f)"** — ID assigned via the collision protocol at file time, re-verified at merge.

## Process constraints (carried)

- Per-unit commits (concurrent sessions are active in `chat_events.py`/`app.py`; small commits keep rebases tractable). Grep-gates re-run at rebase time.
- Test env prefix + Tests/UI isolation + foreground-only pytest + explicit staging, as in the B5 plan.
- Backlog: task-562 ACs #1/#3 completed (#2 is the not-chosen branch — checked N/A via the decision note); status Done in the final commit.
- PR to `dev`; Qodo adjudication; STOP for explicit user merge-go.
