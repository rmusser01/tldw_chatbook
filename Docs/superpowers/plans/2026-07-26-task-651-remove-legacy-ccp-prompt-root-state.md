# Legacy CCP and Prompt Root State Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove retired CCP/prompt root state and stale legacy import callbacks so Personas and Library remain the only production owners.

**Architecture:** Canonical `PersonasScreen` owns characters/personas and its workers; `LibraryScreen` owns prompts and prompt import. Legacy `TAB_CCP` app handlers, prompt-body caches, old widget refresh callbacks, and their timers are deleted without a replacement root cache.

**Tech Stack:** Python 3.11+, Textual production screens/workers, direct import parsers, pytest/pytest-asyncio, AST privacy guards.

**Backlog:** [TASK-651](../../../backlog/tasks/task-651%20-%20Remove-legacy-CCP-and-prompt-root-state.md)

**Specification:** [TldwCli Reactive State Decomposition Design](../specs/2026-07-26-tldwcli-reactive-state-decomposition-design.md)

**Depends on:** TASK-647

**ADR required:** yes

**ADR path:** `backlog/decisions/026-application-session-state-ownership.md`; `backlog/decisions/011-chatbook-workbench-ui-system.md`

**Reason:** Existing ADRs make Personas and Library the production owners and prohibit prompt/domain mirrors on the application root.

---

## Execution and Test Boundary

Mounted tests go in
`Tests/ProductionApp/test_personas_library_root_state.py` and exercise the
registered `PersonasScreen` and `LibraryScreen`. Import parsing may be tested
directly. Do not construct a CCP/prompt test screen or use a mock application.

## Exact Removal Set

Remove:

```text
ccp_active_view
ccp_api_provider_value
current_editing_character_id
current_editing_character_data
conv_char_sidebar_left_collapsed
conv_char_sidebar_right_collapsed
current_conv_char_tab_conversation_id
current_ccp_character_details
current_prompt_id
current_prompt_uuid
current_prompt_name
current_prompt_author
current_prompt_details
current_prompt_system
current_prompt_user
current_prompt_keywords_str
current_prompt_version
```

Also remove `current_ccp_character_image`, `_conv_char_search_timer`,
`_ccp_conversation_search_generation`, obsolete CCP initializer/handlers, and
legacy character/prompt import staging fields when their retired import
surface is deleted.

## File Structure

- Modify `tldw_chatbook/app.py`: remove descriptors, fields, watchers, `TAB_CCP` event branches, and dead imports.
- Modify/delete `tldw_chatbook/Event_Handlers/conv_char_events.py` and relevant `sidebar_events.py` branches after reachability proof.
- Delete or prune `character_ingest_events.py`, `prompt_ingest_events.py`, and the `ingest_events.py` compatibility exports if no registered destination imports them.
- Modify `tldw_chatbook/Event_Handlers/tab_initializers/misc_tab_initializers.py` and `__init__.py`: remove `CCPTabInitializer`.
- Preserve and, only if required, narrow hooks on `tldw_chatbook/UI/Screens/personas_screen.py` and `library_screen.py`.
- Create `Tests/ProductionApp/test_personas_library_root_state.py`.
- Modify `Tests/test_application_state_ownership.py`.

## Task 1: Start TASK-651 and Add Structural/Privacy Failures

- [ ] Move the task In Progress and add its task-local plan:

```bash
backlog task edit 651 -s "In Progress"
backlog task edit 651 --plan $'ADR required: yes\nADR path: backlog/decisions/026-application-session-state-ownership.md; backlog/decisions/011-chatbook-workbench-ui-system.md\nReason: Existing ADRs assign character/persona and prompt state to Personas and Library.\n\n1. Guard the exact removed root set.\n2. Delete legacy CCP/prompt app paths.\n3. Remove stale import refresh callbacks.\n4. Verify canonical production imports and privacy.'
```

- [ ] Add AST guards for the exact root and companion names, including dynamic
  access and prompt-body assignments.
- [ ] Add a source guard that rejects production callbacks querying retired
  `#ccp-*`, `#conv-char-*`, or `#prompt-import-*` widgets outside the live
  Personas implementation.
- [ ] Add a privacy sentinel proving prompt system/user bodies are not stored
  on or rendered through `TldwCli`.
- [ ] Run:

```bash
pytest Tests/ProductionApp/test_personas_library_root_state.py Tests/test_application_state_ownership.py -q
```

Expected: FAIL while root fields and handlers remain.

## Task 2: Delete Legacy CCP and Prompt App Paths

- [ ] Delete the exact descriptors, initializers, writer/watcher methods,
  sidebar toggle/search timers, conversation generation field, prompt editor
  handlers, and `TAB_CCP`-guarded list/input/select/collapsible branches.
- [ ] Remove the dead CCP handler map/imports left after TASK-647.
- [ ] Remove the old initializer class and exports.
- [ ] Delete old modules only after `rg` proves no registered screen imports
  them:

```bash
rg -n "conv_char_events|character_ingest_events|prompt_ingest_events|CCPTabInitializer" tldw_chatbook
```

- [ ] Do not alter the canonical route alias `ccp -> personas`.

## Task 3: Keep Import Completion with the Real Owner

- [ ] Verify `PersonasScreen._import_character_from_path()` refreshes its own
  `CCPCharacterHandler` and exact selected record. If a live external import
  producer remains, give it a narrow mounted-Personas refresh callback; when
  Personas is absent, rely on the next fresh load.
- [ ] Verify `LibraryScreen._run_library_prompts_import()` updates its own
  prompt list/detail state. Remove callbacks to
  `populate_ccp_prompts_list_view()` and legacy Chat prompt search.
- [ ] Add mounted tests using temporary valid import files and the actual
  screens. Completion after navigating away may settle durable import work but
  must not query retired widgets or create root caches.
- [ ] Capture logs with unique character/prompt sentinels and assert bodies are
  absent from bounded failure diagnostics.
- [ ] Run:

```bash
pytest Tests/ProductionApp/test_personas_library_root_state.py -q
```

Expected: PASS.

## Task 4: Verify and Close TASK-651

- [ ] Run:

```bash
pytest Tests/ProductionApp/test_personas_library_root_state.py Tests/ProductionApp/test_chat_composition_retirement.py Tests/test_application_state_ownership.py -q
python -m compileall -q tldw_chatbook/app.py tldw_chatbook/Event_Handlers tldw_chatbook/UI/Screens/personas_screen.py tldw_chatbook/UI/Screens/library_screen.py
python -m ruff check tldw_chatbook/app.py tldw_chatbook/Event_Handlers/conv_char_events.py tldw_chatbook/Event_Handlers/sidebar_events.py tldw_chatbook/Event_Handlers/ingest_events.py tldw_chatbook/Event_Handlers/tab_initializers tldw_chatbook/UI/Screens/personas_screen.py tldw_chatbook/UI/Screens/library_screen.py Tests/ProductionApp/test_personas_library_root_state.py Tests/test_application_state_ownership.py
python -m ruff format --check tldw_chatbook/Event_Handlers/sidebar_events.py tldw_chatbook/Event_Handlers/ingest_events.py tldw_chatbook/Event_Handlers/tab_initializers tldw_chatbook/UI/Screens/personas_screen.py tldw_chatbook/UI/Screens/library_screen.py Tests/ProductionApp/test_personas_library_root_state.py Tests/test_application_state_ownership.py
git diff --check
```

If an optional file above was deleted, omit it from Ruff/format by replacing
the command with the exact surviving changed-file list; do not restore a dead
file merely to satisfy a command. Do not mass-format the verified pre-task
`app.py` or `conv_char_events.py` baseline exceptions.

- [ ] Commit the exact changed/deleted file set:

```bash
git add tldw_chatbook/app.py tldw_chatbook/Event_Handlers/conv_char_events.py tldw_chatbook/Event_Handlers/sidebar_events.py tldw_chatbook/Event_Handlers/ingest_events.py tldw_chatbook/Event_Handlers/tab_initializers/misc_tab_initializers.py tldw_chatbook/Event_Handlers/tab_initializers/__init__.py tldw_chatbook/UI/Screens/personas_screen.py tldw_chatbook/UI/Screens/library_screen.py Tests/ProductionApp/test_personas_library_root_state.py Tests/test_application_state_ownership.py
git commit -m "refactor(personas): remove legacy CCP prompt root state (task-651)"
```

- Stage any manifest-proven deletion such as `character_ingest_events.py` or
  `prompt_ingest_events.py` explicitly with `git add -u -- <path>` after
  checking `git status --short`; do not stage the whole `Event_Handlers` tree.

- [ ] Re-read TASK-651, add Implementation Notes containing actual commands,
  counts, durations, privacy evidence, modified/deleted files, ADRs, and
  deviations, check all acceptance criteria, then mark Done and commit its
  task file:

```bash
backlog task 651 --plain
backlog task edit 651 --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 -s Done
git add 'backlog/tasks/task-651 - Remove-legacy-CCP-and-prompt-root-state.md'
git commit -m "docs(backlog): close CCP prompt root state (task-651)"
```
