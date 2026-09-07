---
id: TASK-1190
title: 'Legacy conversation-list height shares the empty-copy wrap undercount'
status: Done
assignee: []
created_date: '2026-07-27 21:30'
labels: [console, ui, layout]
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-1142 fixed the grouped browser's tray-height undercount (empty_copy Statics wrap to 2 lines at ~100-190-column widths, clipping later headers out of the hit-testable bounds). The transitional legacy path `_legacy_conversation_list_height` (taken when state.conversation_browser is None) still uses the flat `_CONVERSATION_BROWSER_EMPTY_COPY_HEIGHT = 1` constant — the same undercount class can clip the "New conversation" button composed after it. Reviewer-confirmed real; different code path and symptom from 1142 so filed separately.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Legacy-path empty-copy heights account for wrapping (reuse 1142's estimator) or the legacy path is retired.
- [x] #2 A width-parameterized test pins the fix (or the retirement removes the surface).
<!-- AC:END -->

## Implementation Notes

**Decision: retired the legacy path** (not patched). Reachability sweep before
touching any code, grepping every constructor/caller that could produce a
`ConsoleWorkspaceContextState` reaching `ConsoleWorkspaceContextTray.compose()`:

- The sole production builder, `ChatScreen._build_console_workspace_context_state`
  (`tldw_chatbook/UI/Screens/chat_screen.py`), always ends by calling
  `self._with_console_conversation_browser_state(state, ...)`.
- That method always does `browser = build_console_conversation_browser_state(...)`
  and returns `replace(state, conversation_browser=browser, ...)`.
- `build_console_conversation_browser_state`
  (`tldw_chatbook/Workspaces/conversation_browser_state.py`) has exactly one
  `return` statement, constructing a `ConsoleConversationBrowserState` — it can
  never return `None`.
- `build_console_workspace_state` (`tldw_chatbook/Workspaces/display_state.py`,
  the dataclass factory `_build_console_workspace_context_state` starts from)
  has exactly one production caller — `_build_console_workspace_context_state`
  itself — and never sets `conversation_browser`, so the field's `None` default
  is always overwritten by the wrapper before the state reaches any widget.
- Every `.sync_state(...)` call on a mounted `ConsoleWorkspaceContextTray` in
  production (`_sync_console_workspace_context`, the tray construction at
  `chat_screen.py:9822`) is fed by `_build_console_workspace_context_state()`.
  The one defensive branch that reads `state.conversation_browser is None`
  (`chat_screen.py` around the `console-workspace-conversations-toggle`
  button handler) is itself unreachable in production, since that literal
  button id was only ever composed by the now-removed legacy path.

Only test fixtures ever constructed a `ConsoleWorkspaceContextState` with
`conversation_browser=None` and fed it straight to `sync_state()` — no
production code path does. The legacy path was confirmed dead code.

Retired in `tldw_chatbook/Widgets/Console/console_workspace_context.py`:
`_compose_legacy_conversation_section`, `_legacy_conversation_list_height`,
`_conversation_section` (only caller was the legacy compose method),
`_conversation_count_title` (same), `_legacy_title_budget` (same),
`_LEGACY_ROW_CHROME_WIDTH` (only used by `_legacy_title_budget`), and the
now-unused `ConsoleWorkspaceConversationSectionState` import. `compose()`
still guards `if browser is not None:` (renders nothing rather than crash)
as defense in depth rather than an assertion, since the field's declared
type remains `Optional`. `_CONVERSATION_BROWSER_EMPTY_COPY_HEIGHT` was kept —
it is still used by the grouped-browser height helpers
(`_empty_copy_line_count`, `_conversation_browser_list_height`), which is
also why AC#1's "reuse 1142's estimator" is already satisfied for the one
real path.

Tests: removed the 5 tests in `Tests/UI/test_console_workspace_context_rail.py`
that pinned the legacy compose path by calling `tray.sync_state(...)` with a
hand-built `conversation_browser=None` state (render-bounded-expanded-section,
collapsed-shows-selected-summary-only, the legacy-toggle test, the
fallback-disables-unowned-controls test, and the clear-requires-enabled-search
test) — none of them exercised a reachable production path.
`test_console_workspace_context_renders_grouped_conversation_browser`
(same file) already covers "the grouped browser renders when
`conversation_browser` is present" — the one real path — so no new test was
needed for that. Fixed `_workspace_state()` in
`Tests/UI/test_console_rail_sections.py`, which fed `ConsoleWorkspaceContextTray`
via `sync_state`/construction without a `conversation_browser` (previously
silently exercising the legacy fallback); it now includes a real empty
`build_console_conversation_browser_state(rows=(), active_workspace_id=None)`
to match the one production shape (this was needed to keep
`test_context_tray_without_heading_omits_status_rows` passing, since it
asserts `#console-workspace-selected-conversation` renders).

Verified: `pytest Tests/UI/test_console_workspace_context_rail.py
Tests/Workspaces/test_console_conversation_browser_state.py
Tests/UI/test_console_rail_sections.py` — 155 passed, 1 pre-existing failure
(`test_console_workspace_context_syncs_active_conversation_marker`, a
`_sync_console_workspace_context()` positional-arg `TypeError` unrelated to
this change — this file was not touched).

Modified files:
- `tldw_chatbook/Widgets/Console/console_workspace_context.py`
- `Tests/UI/test_console_workspace_context_rail.py`
- `Tests/UI/test_console_rail_sections.py`
