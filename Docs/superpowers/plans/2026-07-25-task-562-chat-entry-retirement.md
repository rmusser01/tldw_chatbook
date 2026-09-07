# task-562 Chat-Entry Retirement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retire the Chat tab's dead conversation-entry chain (task-562, decision: retire in favor of the Console) — ~3,200 production LOC + associated tests — behind per-unit grep-gates, with retirement-guard pins.

**Architecture:** Pure deletion campaign. Every unit deletes only after its gate proves it dead: (a) trigger ids composed nowhere outside retired/deleted modules, (b) zero direct Python callers outside the deleted set, (c) calling tests are themselves casualties. Gate failure ⇒ the unit is DEFERRED to the follow-up task, never forced. Callers delete before callees; `Chat_Window_Enhanced.py` stays but gets its references to each deleted unit stripped in that unit's own commit.

**Tech Stack:** Python 3.11, Textual 8.x, pytest.

**Spec:** `Docs/superpowers/specs/2026-07-25-task-562-chat-entry-retirement-design.md` (committed 96a2dd545) — binding, incl. the gate definition, shared-helper rule, and kept-this-cycle boundary.

## Global Constraints

- Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/personas-redesign`, branch `claude/task-562-chat-tab-conversation-entry`. Subagent shells start in the MAIN checkout and a hook strips a LEADING `cd` — prepend `true; cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/personas-redesign; ` to EVERY Bash command.
- Test prefix: `HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <paths> -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread`. Never mix `Tests/UI/` with other dirs in one invocation. Foreground only. Stage only your files; never `.superpowers/`.
- **Gate command shape (run per symbol/id, expect empty or only-deleted-set hits):**
  - ids: `grep -rn "<id>" tldw_chatbook/ --include='*.py'` (compose sites must all be in retired/deleted modules) and `grep -rn "<id>" Tests/`
  - callers: `grep -rn "<function_name>" tldw_chatbook/ Tests/` (every hit must be the def, the deleted set, or a casualty test)
  - Paste gate outputs into your report. A non-empty unexpected hit ⇒ STOP that unit, mark DEFERRED, continue with the rest, list it in your report.
- After every commit: run the affected suites (listed per task) and `python -c "import tldw_chatbook.app"` as an import smoke.
- Line numbers are from dev `0ef4ee638` + 2 docs commits; locate by symbol.
- Decision doc + follow-up task are CONTROLLER work (already/handled outside these tasks) — do not create them.

---

### Task 1: Unit 1 — the conversation load/save/clone core

**Files:** Modify `tldw_chatbook/Event_Handlers/Chat_Events/chat_events.py`, `chat_events_tabs.py`, `tldw_chatbook/UI/Chat_Window_Enhanced.py`, `Tests/Event_Handlers/Chat_Events/test_chat_events.py`, `test_chat_events_tabs.py`, `Tests/fixtures/event_handler_mocks.py`, plus any Tests/ file the casualty grep surfaces (known suspect: `test_chat_events_integration.py:278,528`).

**Deletion set (each behind its gate):**
- `handle_chat_save_current_chat_button_pressed` (~:3183), `handle_chat_clone_current_chat_button_pressed` (~:3460), `handle_chat_load_selected_button_pressed` (~:3762), `display_conversation_in_chat_tab_ui` (~:4148).
- `chat_events_tabs.py`: `handle_chat_conversation_search_changed_with_tabs` (:295) + `display_conversation_in_chat_tab_ui_with_tabs` (:319); strip their entries from `setup_tab_aware_handlers` (:404) if referenced. Do NOT touch the send/stop/respond wrappers (:99-294).
- `CHAT_BUTTON_HANDLERS` entries: `chat-save-current-chat-button`, `chat-clone-current-chat-button`, `chat-conversation-load-selected-button`.
- `Chat_Window_Enhanced.py` core_handlers map (~:646-656): remove the same three entries + their imports.

**Pre-gate extras (report the answers):**
- Verify `load_branched_conversation_history_ui` (~:4407) does NOT call `display_conversation_in_chat_tab_ui` and is not called BY the deleted set (if either fails: STOP, report — Unit 1's premise changes).
- Shared-helper rule: `_update_console_session_title`, `load_character_and_image`, `resolve_active_user_profile_name_async` etc. have live callers — they STAY; only delete imports that become unused in the edited files (`pyflakes` the touched files).

**Steps:**
- [ ] Run every gate; paste outputs. Expected: all compose sites in `Widgets/settings_sidebar*.py` only; callers = CHAT_BUTTON_HANDLERS/CWE map/tests only.
- [ ] Casualty grep across Tests/ for the four function names + three button ids + `#chat-conversation-title-input`/`-keywords-input`/`-uuid-display`; enumerate the casualty list in your report.
- [ ] Delete test casualties first (they reference symbols about to vanish): `test_chat_events.py` save-chat test (~:302) and the display-fn regions (~:742-1028: T4 site-(c) region incl. `_T4ProfileService`/`_t4_*`/`_T4ServerScopeService`, task-504 region incl. `_t504_conversation_fixture`); `test_chat_events_tabs.py` `TestChatEventsTabsStateSynchronization` display tests (~:575, ~:657) + `test_widget_id_mapping_with_special_selectors` (~:836) if it pins deleted selectors; integration-test hits.
- [ ] Delete the production symbols + map entries + CWE references + now-unused imports (pyflakes-verify).
- [ ] Prune fixture: `#chat-chat-title`/`#chat-chat-id` entries (last consumers deleted) — re-grep Tests/ first to confirm no survivor uses them.
- [ ] Run: `Tests/Event_Handlers/Chat_Events/` full + import smoke. Expected: all green.
- [ ] Commit: `refactor(chat): task-562 U1 — retire load/save/clone handlers + display fn + tabs wrapper (dead since 8ea71071f)`

### Task 2: Units 2+3 — remaining sidebar handlers + conversation-search stack + app.py arms

**Files:** `chat_events.py`, `tldw_chatbook/app.py`, `Chat_Window_Enhanced.py`, affected Tests/ files.

**Deletion set (each behind its gate):**
- Unit 2: `handle_chat_new_conversation_button_pressed` (~:3070), `handle_chat_save_details_button_pressed` (~:3612); gated candidate `handle_chat_convert_to_note_button_pressed` (~:3347) — same dead-button family, delete only if its gate is clean (watch for live "convert to note" callers from Notes/Console integrations). Matching `CHAT_BUTTON_HANDLERS`/CWE-map entries (`chat-new-conversation-button`, `chat-save-conversation-details-button`, `chat-convert-to-note-button`). Gate hard for DIRECT callers (an `action_*`/keybinding/Console command calling any of these fails that symbol's gate).
- Unit 3: `perform_chat_conversation_search` (~:3822), `handle_chat_conversation_search_bar_changed` (~:4108), `handle_chat_search_checkbox_changed` (~:4123); helper `is_general_history_conversation` (~:4118) ONLY if its callers are all in the deleted set (it may serve the Console browser — gate it).
- `app.py`: `on_input_changed` conversation-search arms (~:9297-9317), `on_list_view_selected` `chat-conversation-search-results-list` arm (~:9444-9452), `on_checkbox_changed` conversation-search arm (~:9463), `on_select_changed` character-filter arm (~:9601). Remove the arms; keep the methods and all other arms intact.

**Steps:**
- [ ] Gates + casualty grep (function names, ids: `chat-conversation-search-bar`, `chat-conversation-keyword-search-bar`, `chat-conversation-tags-search-bar`, `chat-conversation-search-results-list`, `chat-conversation-search-character-filter-select`, the three button ids above). Paste outputs; DEFER any symbol with a live hit.
- [ ] Delete casualties in Tests/, then production symbols/arms/map entries/unused imports (pyflakes).
- [ ] Run: `Tests/Event_Handlers/Chat_Events/` + any touched suite + import smoke. Green.
- [ ] Commit: `refactor(chat): task-562 U2+U3 — retire remaining sidebar handlers + conversation-search stack + dead app.py arms`

### Task 3: Units 4+6 — character-sidebar family + dead app.py watchers/branches

**Files:** `chat_events.py`, `app.py`, `Chat_Window_Enhanced.py` (reference strip), `Tests/fixtures/event_handler_mocks.py`, affected Tests/ files.

**Deletion set (each behind its gate):**
- Unit 4: `handle_chat_load_character_button_pressed` + the character search/name-edit handler family behind dead `chat-character-*` ids; their `app.py` `on_input_changed` arms (~:9350/:9354) and any `chat-character-search-results-list` list-view arm; their `CHAT_BUTTON_HANDLERS`/CWE entries (`chat-load-character-button`); their tests (incl. `test_handle_chat_load_character_with_greeting`); the fixture's six `# DEAD-ID` `#chat-character-*-edit` mocks + `#chat-character-search-results-list` if its last consumer goes.
- Gated candidate: `handle_chat_clear_active_character_button_pressed` + `chat-clear-active-character-button` map entries — defer if ambiguous (borders `#chat-system-prompt` send-path surfaces).
- Unit 6 (`app.py`): delete METHODS `watch_current_chat_is_ephemeral` (~:6916-6960), `watch_chat_right_sidebar_collapsed` (~:8605-8617), `watch_chat_right_sidebar_width` (~:8619-8630); delete the `#chat-window` branch in the checkbox handler (~:9486-9496). The reactive ATTRIBUTES stay (live readers in streaming/worker paths — verify `current_chat_is_ephemeral` writers still exist or that remaining writers don't depend on the watcher; report).
- Fixture: remove entries whose last consumers died in Tasks 1-3 (re-grep before each removal).

**Steps:**
- [ ] Gates + casualty grep (ids: `chat-load-character-button`, `chat-clear-active-character-button`, `chat-character-search-input`, `chat-character-name-edit`, `chat-character-search-results-list`; the watcher names). Paste outputs; DEFER on live hits.
- [ ] Delete casualties, then production symbols/arms/watchers/branches/map entries/unused imports (pyflakes).
- [ ] Run: `Tests/Event_Handlers/Chat_Events/` + `Tests/LLM_Management/test_llm_management_events.py` (fixture consumer) + import smoke. Green.
- [ ] Commit: `refactor(chat): task-562 U4+U6 — retire character-sidebar family + dead app.py watchers/branches`

### Task 4: Units 5+7 — whole-file retirements, CSS, retirement guards, task hygiene

**Files:** Delete `tldw_chatbook/Widgets/settings_sidebar.py`, `settings_sidebar_optimized.py`, `tldw_chatbook/Event_Handlers/Chat_Events/chat_events_sidebar_resize.py`. Modify `Chat_Window_Enhanced.py`, `tldw_chatbook/Constants.py`, `css/layout/_sidebars.tcss`, `css/features/_chat.tcss`, `css/components/_buttons.tcss`, regenerate `css/tldw_cli_modular.tcss` via `./build_css.sh`, `Tests/UI/test_legacy_entrypoints_retired.py`, `backlog/tasks/task-562 - ....md`.

**Steps:**
- [ ] Gates: `grep -rn "settings_sidebar\b\|settings_sidebar_optimized\|create_settings_sidebar\|get_pipeline_description\|chat_events_sidebar_resize\|CHAT_SIDEBAR_RESIZE_HANDLERS" tldw_chatbook/ Tests/` — expected hits only in the three files themselves, CWE (to strip), and retirement-guard/test casualties. Paste output.
- [ ] Strip CWE references: sidebar-resize imports/usages (~:670, :709-710, :1009-1017), the `toggle-chat-right-sidebar` buttons (~:577/:673) and their `CHAT_BUTTON_HANDLERS`/core_handlers entries IF `handle_chat_tab_sidebar_toggle` has no other live trigger — gate it; the left-sidebar toggle stays.
- [ ] `git rm` the three files; pyflakes CWE.
- [ ] CSS: remove `#chat-right-sidebar` + deleted-id rules from `Constants.py` and the three SOURCE tcss files; run `./build_css.sh`; commit the regenerated bundle. NEVER hand-edit `css/tldw_cli_modular.tcss`.
- [ ] Extend `Tests/UI/test_legacy_entrypoints_retired.py`: add the three paths to `RETIRED_FILES`/`RETIRED_MODULES` (match the file's existing structure) AND add:
```python
def test_task_562_conversation_entry_chain_retired():
    """task-562: the dead Chat-tab conversation-entry chain must not return."""
    from tldw_chatbook.Event_Handlers.Chat_Events import chat_events

    for name in (
        "display_conversation_in_chat_tab_ui",
        "handle_chat_save_current_chat_button_pressed",
        "handle_chat_clone_current_chat_button_pressed",
        "handle_chat_load_selected_button_pressed",
        "perform_chat_conversation_search",
    ):
        assert not hasattr(chat_events, name), f"{name} was retired in task-562"
    for button_id in (
        "chat-save-current-chat-button",
        "chat-clone-current-chat-button",
        "chat-conversation-load-selected-button",
    ):
        assert button_id not in chat_events.CHAT_BUTTON_HANDLERS
```
  (Adjust the symbol list to what actually got deleted vs deferred — the pin must match reality, not the plan's hope.)
- [ ] Run: `Tests/UI/test_legacy_entrypoints_retired.py` alone; then `Tests/Event_Handlers/Chat_Events/`; then `Tests/test_smoke.py`; import smoke. All green.
- [ ] Update `backlog/tasks/task-562 - ....md`: AC #1 [x] (decision doc reference), AC #2 noted N/A (retire chosen), AC #3 [x] with the deferred-units list if any; Implementation Plan + Implementation Notes; `status: Done`.
- [ ] Commit: `refactor(chat): task-562 U5+U7 — retire settings_sidebar modules, sweep orphan CSS, add retirement guards`

---

### Controller-level (not subagent tasks)

- Decision doc in `backlog/decisions/` (naming per existing files) + follow-up task "Retire Chat_Window_Enhanced + enhanced_settings_sidebar (unmounted since 8ea71071f)" (ID via collision protocol; include every unit DEFERRED by a gate) — committed before Task 1 dispatch.
- Final whole-branch review (sonnet; opus limit until Jul 30): re-run all gates as the rebase/coherence verifier, confirm the retirement-guard pin matches the actual deletion set.
- PR to `dev`; Qodo adjudication; STOP for user merge-go.
