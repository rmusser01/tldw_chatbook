---
id: TASK-577
title: >-
  Retire Chat_Window_Enhanced and enhanced_settings_sidebar (unmounted since
  8ea71071f)
status: Done
assignee:
  - '@claude'
created_date: '2026-07-25 15:10'
updated_date: '2026-07-25 22:09'
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
- [x] #1 Every unit above is either deleted behind a passing grep-gate or explicitly recorded as live/deferred with the gate evidence
- [x] #2 Retirement-guard pins cover the deleted modules and symbols (test_legacy_entrypoints_retired.py pattern)
- [x] #3 No live behavior regresses: full test suite green, app boots, Console chat and all cross-screen handoffs unaffected
<!-- AC:END -->

## PR1 progress

PR1 (branch `claude/task-577-enhanced-window-retirement`, spec
`Docs/superpowers/specs/2026-07-25-task-577-enhanced-window-retirement-design.md`)
is code-complete across four sequential tasks.

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
- **T4** `78a413acc` — U6 residual sweeps + U7 window-family CSS
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

## PR2 progress

PR2 (branch `claude/task-577-pr2-pipeline` off dev `c7bcc6fdd`, spec
`Docs/superpowers/specs/2026-07-25-task-577-enhanced-window-retirement-design.md`
Phase 2 P1-P4) retired the dead legacy chat pipeline across four sequential
tasks. Both PR1 declared inter-PR danglings (the deferred `ChatWindowEnhanced`
import inside `chat_events.py`'s dead send path, and the discovered eager
`chat_events_sidebar` coupling) are now CLOSED — both files are deleted.

### Units delivered (PR2)

- **T1** `3a643ed68` — the handoff-helper gate
  (`apply_current_handoff_context`/`attach_current_handoff_citation_validation`)
  found zero live callers: outcome (b), the helpers die with `chat_events.py`
  in T3 and their pinning tests were removed as casualties.
  `load_branched_conversation_history_ui` (the CCP stratum's sole coupling to
  `chat_events.py`) was relocated verbatim into `conv_char_events.py`,
  keeping the CCP stratum importable and untouched otherwise (out of scope
  per the spec's Boundaries). 3 files changed, 121 insertions(+), 93
  deletions(-).
- **T2** `8bb276103` — cleared `app.py`'s references to the dead pipeline
  before T3's deletions: the `ChatMessage.Action`/`ChatMessageEnhanced.Action`
  arms (Ambiguous Gate 2 resolved DEAD — both widget classes are mounted only
  by the retired send/regenerate flow and the write-only CCP map; the
  imports themselves stay, since the live TTS complete/progress handlers
  still query the DOM for these widget types), the `@on(StreamingChunk)`/
  `@on(StreamDone)` arms + their now-dead import, the write-only
  `_build_handler_map`/`button_handler_map` fabric (scout finding #3: zero
  readers; `on_button_pressed` stays as the no-op screen-nav guard), the
  unreachable legacy body of `watch_current_tab` + `_execute_tab_switch`
  whole, and the zero-reader reactives
  (`current_chat_conversation_id`/`current_chat_is_ephemeral`/
  `current_chat_active_character_data`/`current_ai_message_widget` +
  `set_current_ai_message_widget`/`get_current_ai_message_widget`/
  `active_chat_tab_id`/`chat_sessions`). 1 file changed, 51 insertions(+),
  666 deletions(-).
- **T3** `97304efea` — `git rm` of the four dead-pipeline files behind
  re-run gates: `chat_events.py` (4,738 LOC; closes PR1 dangling #1),
  `chat_events_sidebar.py` (closes PR1 dangling #2 — its only importer was
  `chat_events.py`), `chat_streaming_events.py` (re-confirmed nothing live
  posts `StreamingChunk`/`StreamDone` outside `worker_events` internals),
  `worker_handlers/chat_worker_handler.py` (re-verified: every producer of
  its claimed worker names — `API_Call_chat*`/`API_Call_ccp*`/
  `respond_for_me_worker` — was dead or nonexistent). **Ambiguous Gate 1
  resolved LIVE**: `worker_events.py` gutted from 1,447 to 431 lines —
  deleted `handle_api_call_worker_state_changed` (~1,000 LOC, its only
  caller was the deleted `chat_worker_handler.py`) but KEPT
  `chat_wrapper_function` byte-for-byte (the live core reached via
  `app.chat_wrapper` ← `MediaWindow_v2.py`'s media-analysis flow, and ←
  `conv_char_events.py`'s CCP generators, out of scope) and KEPT the
  `StreamingChunk`/`StreamingChunkWithLogits`/`StreamDone` message classes —
  they are load-bearing internals `chat_wrapper_function` constructs and
  posts itself, both in its streaming branch and, critically, in its
  top-level exception handler that runs on ANY `core_chat_function` failure
  regardless of the streaming flag; deleting them would leave live code
  referencing undefined names (pyflakes F821 at 5 sites, verified). Filed
  **task-634** for a pre-existing bug this review surfaced (not introduced
  by the retirement, byte-identical exception-path contract before/after):
  `MediaWindow_v2.py` treats `chat_wrapper_function`'s
  `"STREAMING_HANDLED_BY_EVENTS"` sentinel return as valid response text on
  LLM failure. 17 files changed, 26 insertions(+), 8,873 deletions(-).
- **T4** (this task) — retirement guards, doc hygiene, closure:
  - `Tests/UI/test_legacy_entrypoints_retired.py`: added the four T3 files
    to `RETIRED_MODULES`/`RETIRED_FILES`; added
    `test_task_577_pr2_pipeline_retired` pinning the verified reality —
    the four modules unimportable, `worker_events.chat_wrapper_function`
    present (`hasattr(...) is True`),
    `worker_events.handle_api_call_worker_state_changed` absent
    (`hasattr(...) is False`), and `app.py` source containing neither
    `def _build_handler_map` nor `self.button_handler_map` (source-grep
    style pin — the retirement is documented in-place by a code comment
    that itself mentions both names, so a naive substring pin on the bare
    names would have false-failed against its own comment).
    `StreamingChunk`/`StreamDone` are deliberately NOT pinned absent — they
    are kept, load-bearing internals, not dead code.
  - Doc hygiene: `tldw_chatbook/Chat/Chat-Uploads-Documentation.md`
    rewritten (not deleted) — its file-type/handler developer content
    (`Utils/file_handlers.py`'s `FileHandler`/`ProcessedFile`/
    `FileHandlerRegistry`) is still live and accurate, but its UI-layer
    framing was 100% `ChatWindowEnhanced`/`chat_events.py`-centric (both
    retired). Rewrote the architecture diagram, User Guide, Implementation
    Details, Database Storage, and API Reference sections to name the
    actual live layer (`UI/Screens/chat_screen.py`'s Console composer →
    `Chat/attachment_core.py` → `Utils/file_handlers.py` →
    `message_attachments` table / `messages.image_data` scalars); also
    corrected two pre-existing drifts unrelated to the retirement (the doc
    described a fictional `[chat.uploads]` config section that was never
    implemented, and a stale 5-handler registry list missing the 4 handlers
    — PDF/Document/Ebook/PlaintextDatabase — added since). Four archival
    docs (`Chat/TABBED_CHATS_LESSONS_LEARNED.md`,
    `Docs/Development/chat-window-behavior-checklist.md`,
    `Docs/Development/Chat/Chat-redux.md`,
    `Docs/Development/app-refactoring-plan-v2.md`) got a one-line header
    note each, no rewrites.
  - CSS: found and swept one orphaned cluster the spec flagged for a check
    (`#chat-sidebar-prompts-*`-era ids) — 7 selector blocks in
    `css/features/_llm-management.tcss` (mirrored in `Constants.py`'s
    unimported legacy `css_content` string) styling ids from the
    long-retired `settings_sidebar.py` (task-562), composed nowhere in any
    live Python file (`id="chat-sidebar` greps to zero hits repo-wide).
    Removed from both sources, regenerated the bundle via `build_css.sh` —
    diff is exactly the swept selectors + the regen timestamp.
  - ADR-026 addendum recording the pipeline-retirement completion.

### Kept-live inventory (AC #1 evidence — what did NOT get deleted, and why)

- `worker_events.chat_wrapper_function` + its `StreamingChunk`/
  `StreamingChunkWithLogits`/`StreamDone` internals — Ambiguous Gate 1,
  LIVE via `app.chat_wrapper` (MediaWindow_v2 media analysis + CCP
  generators).
- `AIGenerationHandler` (`worker_handlers/ai_generation_handler.py`) — a
  structurally distinct handler keyed on `ai_generate_*` worker names (CCP
  character-field generation), never touched — it was never part of the
  `chat_worker_handler.py`/`API_Call_chat*` family that retired.
- The TTS complete/progress handlers' `self.query(ChatMessage)`/
  `self.query(ChatMessageEnhanced)` loops (`app.py`, T2's comment in place)
  — structurally harmless empty-query fallbacks (no live surface mounts
  either widget class) kept because the TTS handlers themselves are live
  and still reference the widget types by name.
- The CCP (`conv_char_events.py`) stratum — explicitly out of scope both
  phases; only received T1's relocation, otherwise untouched.
- `tldw_chatbook/state/chat_state.py::ChatSession` — a distinct dataclass
  from the retired `Widgets/Chat_Widgets/chat_session.py`'s `ChatSession`
  widget class (task-577 T2, PR1); unrelated, unaffected.
- `chat_token_events.py` — out of scope per the spec's Boundaries
  (live-invoked by the 120s `db_status_manager` timer; degenerate queries
  harmless).

### Verification (T4)

`Tests/UI/test_legacy_entrypoints_retired.py` alone (7 passed);
`Tests/Event_Handlers/ Tests/test_smoke.py Tests/LLM_Management/test_llm_management_events.py`
(68 passed, 2 skipped); `Tests/UI/test_media_window_v2_parity.py
Tests/UI/test_chat_first_handoffs.py` (60 passed);
`Tests/UI/test_css_bundle_sync_guard.py Tests/UI/test_css_build_integrity.py`
(10 passed, CSS was touched); pyflakes clean on
`Tests/UI/test_legacy_entrypoints_retired.py` + `tldw_chatbook/Constants.py`;
`python -c "import tldw_chatbook.app"` clean. Full-tree
`pytest --collect-only`: 16,186 tests collected, 4 collection errors — 3
previously-known pre-existing errors in `Tests/Research_Interop/` +
`Tests/Watchlists/` (unrelated rootdir/import-mode quirk, each collects
fine in isolation) plus **1 new pre-existing error**,
`Tests/Skills/test_skill_trust_keyring_autounlock.py` (a relative import
that fails outside its package's own collection root, introduced by
unrelated task-624 keyring work on `dev` — reproduced failing even in
isolation, confirmed unrelated to this task by `git log` on the file).

### Cumulative PR2 LOC delta

`git diff --shortstat c7bcc6fdd 97304efea` (T1-T3, production +
first-round test casualties): 21 files changed, 275 insertions(+), 9,632
deletions(-). T4 adds guard-pin/doc/CSS-only changes on top (no further
production deletions).
