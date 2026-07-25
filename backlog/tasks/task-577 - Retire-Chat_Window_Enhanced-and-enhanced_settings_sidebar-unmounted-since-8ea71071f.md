---
id: TASK-577
title: >-
  Retire Chat_Window_Enhanced and enhanced_settings_sidebar (unmounted since
  8ea71071f)
status: In Progress
assignee:
  - '@claude'
created_date: '2026-07-25 15:10'
updated_date: '2026-07-25 19:07'
labels:
  - chat
  - dead-code
  - tech-debt
dependencies:
  - task-562
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up to task-562 / ADR-026. The task-562 scout established that the entire
`ChatWindowEnhanced` surface has been unmounted since commit `8ea71071f`
("Move Console transcript and composer to native surface", 2026-05-06):
`_ensure_chat_window()` (chat_screen.py) has zero callers, `self.chat_window`
stays `None` for the process lifetime, and `#chat-window` / `#chat-log` /
`EnhancedSettingsSidebar` never exist in the live tree. task-562 retired the
conversation-entry chain but deliberately KEPT the window family to bound its
blast radius.

Remaining retirement audit (~2,600+ production LOC + ten test suites):
`UI/Chat_Window_Enhanced.py` (~1,163), `Widgets/enhanced_settings_sidebar.py`
(~1,429), `UI/Chat_Modules/`, the `use_enhanced_window` config flag +
Tools/Settings checkbox (a no-op toggle today), the `#chat-window` dead-end
consumers (`app.py`, `worker_events.py`, `chat_events.py`, `chat_events_tabs.py`),
the chat_events send-path liveness question (the `use_enhanced_window` reads at
chat_events.py ~:792/:1067/:1125/:1266/:1661/:2760 and the tab wrappers in
`chat_events_tabs.py` :99-294 serve surfaces that may all be unreachable), the
chat-tabs subsystem (`ChatTabContainer`/`chat_session.py` — composed only inside
the unmounted window), plus any unit task-562's gates DEFERRED (recorded in
task-562's Implementation Notes). Same method as task-562: per-unit grep-gates
(ids composed nowhere live + zero direct callers), retirement-guard pins in
`test_legacy_entrypoints_retired.py`, defer on gate failure.

Additional audit items surfaced during task-562's execution (all verified inert,
left in place for scope discipline):
- `UI/Chat_Modules/chat_sidebar_handler.py` `handle_character_buttons` — dict
  referencing task-562-deleted handler names inside a method with zero callers.
- `app.py` `_build_handler_map()`/`self.button_handler_map` — built once at
  init, never read (write-only dead).
- Dead `toggle-chat-right-sidebar` map entries: `app.py` ~:5256
  (`chat_handlers_map`), `chat_events.py` CHAT_BUTTON_HANDLERS (~:331/:4808) —
  keys that can never match (id composed nowhere); handler
  `handle_chat_tab_sidebar_toggle` stays for the live left toggle.
- `Event_Handlers/sidebar_events.py` — whole module dead
  (`SIDEBAR_BUTTON_HANDLERS` never imported).
- Orphaned CSS *class* selectors (`.save-chat-button`, `.sidebar-resize-button`)
  in source tcss (id rules were swept in task-562; class rules kept).
- `chat_right_sidebar_width` reactive (`app.py` ~:2856) — zero readers/writers
  since `chat_events_sidebar_resize.py` was retired.
- `_populate_chat_character_search_list` (chat_events.py ~:3376) — targets the
  deleted sidebar's `#chat-character-search-results-list`; invoked on LIVE hot
  paths (app.py watch_current_tab ~:8373; tab_initializers/chat_tab_initializer.py:52)
  and fails its first query_one and logs an ERROR on every Chat-tab
  switch/init — the retirement should remove the calls AND the helper
  (live-path edit, deferred from task-562 for scope discipline).
- `populate_chat_conversation_character_filter_select` (chat_events.py ~:4614)
  — same family; one live caller (character_ingest_events.py:451) + one dead
  caller (app.py ~:9130).
- Two dead `@on(Collapsible.Toggled)` decorators: app.py ~:9089
  (`#chat-active-character-info-collapsible`) and ~:9108 (`#chat-conversations`)
  — ids composed only in the deleted settings_sidebar.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every unit above is either deleted behind a passing grep-gate or explicitly recorded as live/deferred with the gate evidence
- [ ] #2 Retirement-guard pins cover the deleted modules and symbols (test_legacy_entrypoints_retired.py pattern)
- [ ] #3 No live behavior regresses: full test suite green, app boots, Console chat and all cross-screen handoffs unaffected
<!-- AC:END -->

## PR1 progress

PR1 (branch `claude/task-577-enhanced-window-retirement`, spec
`Docs/superpowers/specs/2026-07-25-task-577-enhanced-window-retirement-design.md`)
is code-complete across four sequential tasks. Status stays **In Progress**
— PR2 (the dead-pipeline phase: `chat_events.py`, the streaming/worker
family, the app.py seams) has not started.

### Units delivered (PR1)

- **T1** `fa00fbb40` — stripped every `self.chat_window`/`ChatWindowEnhanced`/
  `ChatTabContainer` seam out of the live `chat_screen.py` (the 64 refs) and
  the dead CWE-ancestor-walk branch in `compact_model_bar.py`'s sidebar
  toggle. Per-block DELETE-vs-PRESERVE-FALLTHROUGH classification, gated by
  grep before each removal.
- **T2** `542269648` — whole-file/package `git rm`: `UI/Chat_Window_Enhanced.py`,
  `Widgets/enhanced_settings_sidebar.py`, `Widgets/minimal_settings_sidebar.py`,
  `UI/Chat_Modules/` (7 files + package), `Widgets/Chat_Widgets/chat_tab_container.py`,
  `chat_session.py`, `chat_tab_bar.py`, `Chat/tabs/` (package),
  `Event_Handlers/Chat_Events/chat_events_tabs.py`,
  `Event_Handlers/tab_initializers/` (whole package),
  `Event_Handlers/sidebar_events.py`. Ten test suites retired alongside as
  casualties (module-level CWE/tabs imports). Discovered and recorded a new
  inter-PR dangling beyond the spec's declared one (below).
- **T3** `302c71c3a` — removed the `use_enhanced_window` and
  `enable_tabs`/`max_tabs` config keys + their four Settings UI rows
  (checkboxes/input, save, reset, restart notices) in
  `Tools_Settings_Window.py`; the now-empty "Interface Options" settings
  group removed. Spec-approved deliberate behavior change:
  `open_chat_with_handoff` no longer refuses a handoff on the retired
  `enable_tabs` flag.
- **T4** `78a413acc` (this task) — U6 residual sweeps + U7 window-family CSS
  retirement + retirement-guard pins:
  - `app.py`: dropped the dead `"chat-window"` entry from `ALL_MAIN_WINDOW_IDS`,
    the `screen.chat_window` streaming-button poke (T1 already removed the
    field it guarded), the dead `toggle-chat-right-sidebar` entry from
    `chat_handlers_map` (`button_handler_map` has zero readers), and the
    `chat_right_sidebar_width` reactive (zero readers/writers since
    task-562 retired `chat_events_sidebar_resize.py`).
  - `chat_right_sidebar_collapsed` **KEPT** — Ambiguous Gate 3 evidence:
    `chat_screen.py`'s `save_state()` (invoked live on every screen switch
    via `app.py:5767`, `current_screen.save_state()`) reads it with
    `getattr(self.app_instance, "chat_right_sidebar_collapsed", False)` to
    persist the flag into `ChatScreenState.right_sidebar_collapsed`. A live
    reader, not a dead guard; documented in place with a code comment.
  - `Utils/chat_diagnostics.py` retired outright (`git rm`): zero importers
    anywhere in `tldw_chatbook/` or `Tests/`.
  - Carried items closed: `_restore_input_text`/`_restore_scroll_positions`/
    `_extract_and_save_messages` (chat_screen.py) deleted — zero production
    callers (the first two were only *mentioned*, not called, in a stale
    docstring; the third's sole test caller was a casualty, deleted with
    its `EmptyChatLog` fixture); `_save_scroll_positions` **KEPT** — still
    called from live `save_state()`. `Docs/CHAT_TABS_GUIDE.md` removed
    (zero referencers). Repo-root `config.toml`'s stray
    `enable_tabs`/`max_tabs` lines removed (two lines only).
  - CSS sweep (SOURCE tcss + `Constants.py`'s legacy, zero-importer
    `css_content` string; regenerated via `build_css.sh`): whole-file
    removal of `features/_chat_tabs.tcss` (+ its `build_css.py` manifest
    entry) and dead window-family id/class rules from `_chat.tcss`,
    `_sidebars.tcss` (`.preset-bar`/`.preset-button`,
    `.sidebar-resize-button`), `_buttons.tcss` (`.mic-button`,
    `#toggle-chat-left-sidebar`), and their `Constants.py` mirrors.
    `components/_agentic_terminal.tcss` deliberately left untouched — it
    scopes `.chat-log`/`.chat-empty-state`/`.chat-input-area`/`.chat-session`
    under `#console-session-surface`/`#console-chat-tabs`, the Console-twin
    case the design doc flagged to leave alone. Bundle diff verified to be
    exactly the swept selectors + the regen timestamp.
  - Retirement guards: `RETIRED_MODULES`/`RETIRED_FILES` extended with
    every T2 whole-file/package deletion + this task's `chat_diagnostics.py`;
    added `test_task_577_pr1_window_family_retired` pinning module
    unimportability, `ChatScreen` having neither `_ensure_chat_window` nor a
    `chat_window` attribute, and the three retired config keys being absent
    from `config.py`'s `CONFIG_TOML_CONTENT`.

### Deferred (inter-PR dangling, resolved in PR2)

1. **Declared by the spec**: `chat_events.py` keeps a DEFERRED,
   function-local, never-executed `ChatWindowEnhanced` import inside its
   dead send path. The whole file is deleted in PR2 (P1); module import and
   app boot are unaffected today.
2. **Discovered by T2's gate**: `Event_Handlers/Chat_Events/chat_events_sidebar.py`
   is NOT retired — `chat_events.py` has an eager, module-level
   `from ... import chat_events_sidebar` and builds `CHAT_BUTTON_HANDLERS`
   from its `CHAT_SIDEBAR_BUTTON_HANDLERS` at module scope; deleting the
   file today would break `chat_events.py`'s import and app boot. It
   retires alongside `chat_events.py` itself in PR2 P1.

### Still open (out of T4's scope, left for a later pass)

The task-577 file's "Additional audit items" list still has three
unaddressed entries that were not named in the T4 brief:
`_populate_chat_character_search_list` (chat_events.py, live hot-path
caller in `app.py` `watch_current_tab` + a `tab_initializers` caller —
needs a live-path edit, not a pure deletion), the sibling
`populate_chat_conversation_character_filter_select` (one live caller in
`character_ingest_events.py:451`), and the two dead
`@on(Collapsible.Toggled)` decorators in `app.py` (`~:9089`/`~:9108`).
These belong to `chat_events.py`'s own retirement and are folded into PR2
P1/P3.

### Verification (T4)

`Tests/UI/test_legacy_entrypoints_retired.py` alone (6 passed);
`Tests/Event_Handlers/Chat_Events/ Tests/test_smoke.py` (71 passed, 2
skipped); `Tests/UI/test_css_bundle_sync_guard.py
Tests/UI/test_css_build_integrity.py` (10 passed);
`Tests/UI/test_console_native_chat_flow.py Tests/UI/test_chat_first_handoffs.py`
(232 passed); pyflakes clean on every touched production/test file;
`python -c "import tldw_chatbook.app"` clean.

### LOC delta (T4 commit `78a413acc`)

14 files changed, 135 insertions(+), 1,440 deletions(-). Whole-file
deletions: `tldw_chatbook/Utils/chat_diagnostics.py` (367 lines),
`tldw_chatbook/Docs/CHAT_TABS_GUIDE.md` (96 lines),
`tldw_chatbook/css/features/_chat_tabs.tcss` (175 lines).
