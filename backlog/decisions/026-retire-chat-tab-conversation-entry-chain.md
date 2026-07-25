# 026 — Retire the Chat-tab conversation-entry chain in favor of the Console

**Status:** Accepted (2026-07-25, task-562)
**Deciders:** project owner (explicit decision, task-562 AC #1)

## Context

`display_conversation_in_chat_tab_ui()` and its caller handlers (save / clone /
load-selected) were repaired in task-504 (PR #861) but remained unreachable: their
trigger buttons are composed only in `Widgets/settings_sidebar.py`, whose
`create_settings_sidebar()` lost its last caller when task-412 retired the legacy
ChatWindow. A task-562 scout then established the deeper fact: **the entire
`ChatWindowEnhanced` surface has been unmounted since commit `8ea71071f`
("Move Console transcript and composer to native surface", 2026-05-06)** —
`_ensure_chat_window()` has zero callers, `self.chat_window` stays `None`, and
`#chat-window`/`#chat-log`/the enhanced sidebar never exist in the live tree. The
`use_enhanced_window` config flag selects nothing.

Meanwhile the native Console already provides the user-facing capability:
- its conversation browser searches all ChaChaNotes conversations (global +
  workspace scopes, not character-gated, not console-only — filter at
  `chat_screen._persisted_console_browser_rows`),
- resume loads full conversation trees into native Console sessions,
- persistence is automatic (no manual "save chat" concept),
- every cross-screen "open in chat" flow (`app.open_chat_with_handoff`) targets
  the Console.

## Decision

**Retire the dead conversation-entry chain rather than restore it.** Restoring
would require reversing the May native-Console migration *and* duplicating the
Console's conversation browser — rejected.

Scope (task-562): the load/save/clone/new/save-details handler family, the
conversation-search stack, `display_conversation_in_chat_tab_ui` (+ tabs
wrapper), the retired sidebar modules (`settings_sidebar.py`,
`settings_sidebar_optimized.py`, `chat_events_sidebar_resize.py`), the dead
`app.py` router arms/watchers, and the orphan CSS — each unit deleted only
after a grep-gate proves it dead (ids composed nowhere live + zero direct
Python callers); gate failures are deferred, not forced.

Explicitly **kept this cycle**: `Chat_Window_Enhanced.py`,
`enhanced_settings_sidebar.py`, `UI/Chat_Modules/`, their test suites, the
`use_enhanced_window` flag + Settings checkbox, and the chat_events send-path
liveness question — deferred to the follow-up task "Retire Chat_Window_Enhanced +
enhanced_settings_sidebar (unmounted since 8ea71071f)".

## Consequences

- ~3,200 production LOC + ~490 test LOC of dead code removed; retirement-guard
  pins in `Tests/UI/test_legacy_entrypoints_retired.py` keep the chain from
  silently returning.
- The function repaired in PR #861 (task-504) is deleted two days after its
  repair — the repair was correct while the restore-vs-retire decision was
  open; the retirement guard replaces its regression tests.
- task-551's third substitution site (the display path) is removed by deletion,
  not regression; the async resolver and its other two live sites keep their
  own coverage.
- Chat-tab conversation loading is officially a Console capability; any future
  "load a conversation in the Chat tab" feature request routes to the Console
  browser, not to a revived enhanced-window sidebar.
- Post-implementation addendum (2026-07-25): final campaign total ~5,150 lines
  removed once Unit 5+7 whole-file/CSS/reference reductions landed (the ~3,200
  figure above was the pre-implementation Units 1-4/6 estimate).
