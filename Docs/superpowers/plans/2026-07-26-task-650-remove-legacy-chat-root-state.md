# Legacy Chat Root State Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove all legacy Chat domain/view reactives and singleton worker/widget fields from `TldwCli` after the dormant composition is gone.

**Architecture:** Native Console remains owned by `ConsoleChatStore`, `ConsoleChatSession`, and `ChatScreen` rail/snapshot fields. Legacy app handlers, debounce timers, sidebar caches, and streaming-worker bridges disappear atomically with their root names; no compatibility mirror replaces them.

**Tech Stack:** Python 3.11+, Textual production app, Console store/session models, AST ownership tests, pytest/pytest-asyncio, Ruff.

**Backlog:** [TASK-650](../../../backlog/tasks/task-650%20-%20Remove-legacy-Chat-root-reactive-and-worker-state.md)

**Specification:** [TldwCli Reactive State Decomposition Design](../specs/2026-07-26-tldwcli-reactive-state-decomposition-design.md)

**Depends on:** TASK-648, TASK-649

**ADR required:** yes

**ADR path:** `backlog/decisions/026-application-session-state-ownership.md`; `backlog/decisions/011-chatbook-workbench-ui-system.md`

**Reason:** Existing ADRs assign Chat session/run/view ownership to native Console and forbid root mirrors.

---

## Execution and Test Boundary

Mounted coverage goes in
`Tests/ProductionApp/test_chat_root_state_removal.py` using the normal
`TldwCli` and registered `ChatScreen`. Direct tests may exercise
`ConsoleChatStore`/session functions. Do not run or adapt legacy Chat widget
test applications.

## Exact Removal Set

Remove these root reactives and every initializer, writer, watcher, direct
reader, dynamic access, and snapshot dependency:

```text
rag_expansion_provider_value
chat_sidebar_collapsed
chat_right_sidebar_collapsed
chat_right_sidebar_width
chat_sidebar_selected_prompt_id
chat_sidebar_selected_prompt_system
chat_sidebar_selected_prompt_user
current_chat_is_ephemeral
current_chat_conversation_id
current_chat_active_character_data
active_chat_tab_id
chat_sessions
chat_sidebar_loaded_prompt_id
chat_sidebar_loaded_prompt_title_text
chat_sidebar_loaded_prompt_system_text
chat_sidebar_loaded_prompt_user_text
chat_sidebar_loaded_prompt_keywords_text
chat_sidebar_prompt_display_visible
chat_settings_mode
chat_settings_search_query
```

Also remove `_chat_state_lock`, `current_ai_message_widget`,
`current_chat_worker`, `current_chat_is_streaming` and its accessors,
`current_chat_note_id`, `current_chat_note_version`,
`_conversation_search_timer`, `_chat_sidebar_prompt_search_timer`,
`_media_sidebar_search_timer`, and legacy sidebar media pagination/selection
fields `media_search_current_page`, `media_search_total_pages`, and
`current_sidebar_media_item`, proven exclusive by TASK-649's manifest.

## File Structure

- Modify `tldw_chatbook/app.py`: delete the exact root set and legacy app event methods.
- Modify/delete root-wired modules under `tldw_chatbook/Event_Handlers/Chat_Events/` according to live imports.
- Modify `tldw_chatbook/Event_Handlers/sidebar_events.py`, `worker_events.py`, and `worker_handlers/chat_worker_handler.py`: remove legacy singleton/run bridges while preserving live non-Chat consumers.
- Modify `tldw_chatbook/UI/Screens/chat_screen.py` and `chat_screen_state.py`: native-only snapshots and rail state.
- Modify shared widgets/helpers only when they still reference a removed root name.
- Create `Tests/ProductionApp/test_chat_root_state_removal.py`.
- Modify `Tests/test_application_state_ownership.py`.

## Task 1: Start TASK-650 and Add the Full Removal Guard

- [ ] Move the task In Progress and add the exact removal set to its local plan:

```bash
backlog task edit 650 -s "In Progress"
backlog task edit 650 --plan $'ADR required: yes\nADR path: backlog/decisions/026-application-session-state-ownership.md; backlog/decisions/011-chatbook-workbench-ui-system.md\nReason: Existing ADRs make native Console the only Chat session and run owner.\n\n1. Add exact removed-name AST and mounted guards.\n2. Remove root descriptors and companion singleton fields.\n3. Delete root-wired handlers and timers.\n4. Verify native Console snapshots, runs, and cancellation.'
```

- [ ] Extend the AST collector to reject each exact name as a `TldwCli`
  descriptor/assignment, `app.<name>` access, constant
  `getattr`/`setattr`/`delattr`, and string-key access. A same-named field on a
  genuine destination owner remains allowed.
- [ ] Add a mounted test that reaches native Console, opens/closes actual rails,
  creates/switches sessions, saves/restores the real screen snapshot, and
  verifies no removed root name appears on the app or in the snapshot.
- [ ] Run:

```bash
pytest Tests/ProductionApp/test_chat_root_state_removal.py Tests/test_application_state_ownership.py -q
```

Expected: FAIL against current root state.

## Task 2: Delete Root Reactives and Companion Fields Atomically

- [ ] Delete the exact reactive descriptors and `__init__` assignments.
- [ ] Delete watchers and root event methods whose only purpose is legacy
  sidebar, prompt, character, conversation, model-select, or streaming state.
- [ ] Remove legacy timers from creation, cancellation, and shutdown cleanup.
  Do not move them into a new aggregate object.
- [ ] Remove singleton worker/widget accessors and ensure native Console
  continues to use its existing controller/store cancellation path.
- [ ] Delete obsolete imports/constants after each code path is removed.

## Task 3: Make ChatScreen Snapshot and Runtime Paths Native-Only

- [ ] Remove all save/restore reads or writes of root sidebar fields.
  Serialize only the actual Console-owned rail/session primitives already
  allowed by the snapshot contract.
- [ ] Remove fallback lookup of legacy tab containers, legacy chat workers,
  legacy prompt caches, and legacy message widgets.
- [ ] Keep native transcript, active session ID, per-session settings,
  cancellation, and composer behavior unchanged.
- [ ] Add a production test that starts a native Console run with a narrow
  injected provider collaborator on the real screen, cancels through the
  public Console operation, and proves no legacy app worker field participates.

## Task 4: Remove Root-Wired Legacy Modules and Tests

- [ ] Re-run:

```bash
rg -n "rag_expansion_provider_value|chat_sidebar_collapsed|current_chat_is_ephemeral|current_chat_conversation_id|current_ai_message_widget|current_chat_worker|current_chat_is_streaming|chat_sidebar_loaded_prompt" tldw_chatbook Tests
```

- [ ] Delete an event/helper module only when no registered production route or
  retained direct function imports it. Prune legacy branches from shared
  modules and record the disposition in TASK-649's reachability document.
- [ ] Remove or rewrite tests that use mock/test applications to exercise the
  deleted root contract. Do not replace them with another simplified app.
- [ ] Run:

```bash
pytest Tests/ProductionApp/test_chat_composition_retirement.py Tests/ProductionApp/test_chat_root_state_removal.py Tests/ProductionApp/test_provider_selection_ownership.py Tests/test_application_state_ownership.py -q
```

Expected: PASS.

## Task 5: Verify and Close TASK-650

- [ ] Run:

```bash
python -m compileall -q tldw_chatbook/app.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/UI/Screens/chat_screen_state.py tldw_chatbook/Event_Handlers/Chat_Events tldw_chatbook/Event_Handlers/sidebar_events.py tldw_chatbook/Event_Handlers/worker_events.py tldw_chatbook/Event_Handlers/worker_handlers/chat_worker_handler.py
python -m ruff check tldw_chatbook/app.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/UI/Screens/chat_screen_state.py tldw_chatbook/Event_Handlers/Chat_Events tldw_chatbook/Event_Handlers/sidebar_events.py tldw_chatbook/Event_Handlers/worker_events.py tldw_chatbook/Event_Handlers/worker_handlers/chat_worker_handler.py Tests/ProductionApp/test_chat_root_state_removal.py Tests/test_application_state_ownership.py
python -m ruff format --check tldw_chatbook/UI/Screens/chat_screen_state.py tldw_chatbook/Event_Handlers/sidebar_events.py tldw_chatbook/Event_Handlers/worker_events.py tldw_chatbook/Event_Handlers/worker_handlers/chat_worker_handler.py Tests/ProductionApp/test_chat_root_state_removal.py Tests/test_application_state_ownership.py
git diff --check
```

- Do not mass-format the verified pre-task `app.py`, `chat_screen.py`, or
  `Chat_Events/chat_events.py` baseline exceptions.

- [ ] Commit implementation:

```bash
git add tldw_chatbook/app.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/UI/Screens/chat_screen_state.py tldw_chatbook/Event_Handlers/sidebar_events.py tldw_chatbook/Event_Handlers/worker_events.py tldw_chatbook/Event_Handlers/worker_handlers/chat_worker_handler.py Tests/ProductionApp/test_chat_root_state_removal.py Tests/test_application_state_ownership.py Docs/superpowers/reviews/2026-07-26-task-649-legacy-chat-reachability.md
git commit -m "refactor(chat): remove legacy root state (task-650)"
```

- Use `git status --short` and stage each modified/deleted
  `Event_Handlers/Chat_Events` path approved by TASK-649's manifest explicitly.
  Do not stage the whole `Event_Handlers` tree.

- [ ] Re-read TASK-650, add Implementation Notes containing actual commands,
  counts, durations, modified/deleted files, ADRs, and deviations, check all
  acceptance criteria, then mark Done and commit its task file:

```bash
backlog task 650 --plain
backlog task edit 650 --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 -s Done
git add 'backlog/tasks/task-650 - Remove-legacy-Chat-root-reactive-and-worker-state.md'
git commit -m "docs(backlog): close legacy Chat root state (task-650)"
```
