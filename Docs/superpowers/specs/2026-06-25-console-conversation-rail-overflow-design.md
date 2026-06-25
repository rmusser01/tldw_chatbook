# Console Conversation Rail Overflow Design

Date: 2026-06-25
Status: Approved by user through brainstorming, pending written-spec review
Primary Repo: `tldw_chatbook`
Scope: Console left Context rail, workspace conversation list, search, collapse state, and layout QA

## Summary

Users can create enough Console conversations that the left Context rail fills with conversation rows. The lower workspace status, server readiness, and handoff content then becomes hard to reach. The selected direction is to keep conversations in the Console workspace rail, but make the Conversations subsection bounded, collapsible, and searchable across all saved conversations in the active workspace.

This is a Console presentation and interaction fix. It must preserve the existing local-first workspace contract, workspace-scoped new chat creation, saved conversation resume behavior, and server/sync/handoff unavailable copy.

## Current Context

The relevant Console implementation is already split into useful seams:

- `tldw_chatbook/UI/Screens/chat_screen.py` builds Console workspace display state, owns native Console session state, handles workspace row selection, and resumes persisted workspace conversations.
- `tldw_chatbook/Widgets/Console/console_workspace_context.py` renders the left rail workspace tray, including the current inline `Conversations` list.
- `tldw_chatbook/Workspaces/display_state.py` defines `ConsoleWorkspaceContextState` and `ConsoleWorkspaceConversationRow`.
- `tldw_chatbook/Workspaces/registry_service.py` lists workspace conversation memberships.
- `tldw_chatbook/Chat/chat_conversation_service.py` can list/search persisted conversations with workspace scope.

The current rail renders the conversation list inline with a height derived from `len(rows) * 3`. That makes a large workspace conversation set consume the rail before users can reach content below it.

ADR-005 remains the governing workspace boundary: Console workspace switching is local-first, no background sync or server hydration is implied, and the built-in Default workspace remains local-only with file tools disabled.

## Goals

- Bound the Console workspace Conversations subsection so many conversations cannot permanently push lower rail content out of reach.
- Make the subsection adaptive to the available left-rail height rather than using one fixed row count.
- Let users collapse Conversations per active workspace.
- In collapsed mode, show the header/count plus the selected conversation summary.
- Keep workspace-scoped `New conversation` available in expanded mode.
- Add transient search across all saved persisted conversations in the active workspace.
- Keep search active after selecting a result so users can continue browsing matching conversations.
- Reset search text on workspace switch or screen recreation.
- Keep selected conversation context visible even when the selected row is not in the current scroll position or result page.
- Preserve open native Console sessions, saved conversation resume, workspace isolation, Default workspace policy, and existing server/sync/handoff copy.
- Verify with mounted Console tests and rendered Textual-web/CDP evidence.

## Non-Goals

- Do not change workspace storage schema, chat storage schema, sync policy, server handoff policy, or workspace ownership.
- Do not move saved conversation browsing out of Console into Library for this fix.
- Do not remove the existing workspace rail `New conversation` behavior; it remains available when the subsection is expanded.
- Do not implement full conversation management, deletion, bulk actions, or advanced filters.
- Do not make search persistent. Search is a transient screen interaction.
- Do not implement infinite scrolling in the first slice. A capped result set with explicit count copy is enough.

## Approved Direction

Use a bounded adaptive Conversations subsection plus active-workspace search.

Rejected alternatives:

- Recent-only list with manual `More`: simpler, but it still needs bounding and makes older conversations harder to find.
- Move conversation browsing to Library only: cleaner rail, but it breaks the recently added Console workspace resume workflow and adds navigation friction.

The chosen approach preserves Console as the work hub while fixing the overflow at the source.

## UX Model

### Expanded Conversations

The expanded subsection contains:

1. Header with `Conversations` and active-workspace count.
2. Collapse toggle.
3. Selected conversation summary.
4. Compact search input and clear affordance.
5. Bounded scrollable row list.
6. Compact `New conversation` action.

The bounded row list uses adaptive height. Target 45% of the mounted left-rail body height, clamp to 4-12 visible rows, and account for the existing two-line row plus margin geometry. If the mounted terminal is too short to satisfy both the list and lower status sections, preserve at least 4 visible rows and let the outer left rail scroll to the status/handoff rows.

### Collapsed Conversations

Collapsed mode shows:

1. Header with count.
2. Selected conversation summary, or an explicit no-active-conversation summary.

Collapsed mode does not show `New conversation`, because the Console tab strip already offers a primary new-chat path. The workspace rail action remains available when Conversations is expanded to preserve the existing TASK-126 behavior.

Collapse state persists per active workspace. It is UI-only Console config, not workspace registry data.

### Search

Search text is transient. It resets when:

- The active workspace changes.
- The Console screen is recreated.

When the query is non-empty, the row list shows capped search results from saved persisted conversations in the active workspace, merged with matching open native Console sessions. Selecting a result resumes or switches to that conversation and keeps the search query active.

Users can clear search through a compact clear affordance or by clearing the input text. Clearing search returns to the default recent/open list.

## Architecture

`ChatScreen` should own interaction state and service calls. `ConsoleWorkspaceContextTray` should render state and emit normal Textual events.

### New Presentation State

Add a small presentation model for the Conversations subsection in `tldw_chatbook/Workspaces/display_state.py`, beside the existing `ConsoleWorkspaceContextState` and `ConsoleWorkspaceConversationRow`.

Suggested shape:

```python
@dataclass(frozen=True)
class ConsoleWorkspaceConversationSectionState:
    workspace_id: str
    collapsed: bool
    query: str
    selected_summary: str
    rows: tuple[ConsoleWorkspaceConversationRow, ...]
    total_count: int | None = None
    result_limit: int = 50
    status_copy: str = ""
    empty_copy: str = ""
    search_enabled: bool = True
    new_conversation_enabled: bool = True
    error_copy: str = ""
```

Exact naming can adapt to existing conventions. The important contract is that the tray can render the subsection without querying services or mutating app state.

### Screen-Owned State

`ChatScreen` owns:

- `conversation_search_query`
- Pending debounce timer or equivalent input delay
- Per-workspace collapsed preference
- Current conversation-section state

`ChatScreen` applies search results only when the result workspace id and query still match the current screen state. This prevents stale search results from rendering after workspace switches.

### Service Boundaries

Default rows come from:

- Open native Console sessions for the active workspace.
- Recent workspace conversation memberships from the existing workspace registry.

Search rows come from:

- Matching open native Console sessions.
- `app.chat_conversation_scope_service.list_conversations(mode="local", query=..., scope_type="workspace", workspace_id=active_workspace_id, limit=50)`, which routes to the existing local `ChatConversationService.list_conversations(...)`.

Rows are deduped by persisted conversation id when available. Native sessions without persisted conversation ids keep `native:<session-id>` identifiers so they remain switchable before first durable message persistence.

Search must not bypass workspace scope. Results from other workspaces must never appear in the active workspace rail.

### Widget Rendering

`ConsoleWorkspaceContextTray` changes the Conversations region from a growing inline `Vertical` into:

- Header/summary controls.
- Search controls when expanded.
- A bounded `VerticalScroll` row list when expanded.
- Compact `New conversation` action when expanded.

The tray should not own app state. It emits `Input.Changed` and `Button.Pressed`; `ChatScreen` updates state, calls services, and syncs the tray.

Because the current tray uses `refresh(recompose=True)`, implementation must preserve search input value and focus during search updates. A smaller list-only refresh is preferred if practical; otherwise `ChatScreen` must restore the input value and focus after recompose.

## Data Flow

### Mount Or Workspace Switch

1. `ChatScreen` reads the active workspace id.
2. It resets transient search text.
3. It loads the per-workspace Conversations collapse preference.
4. It builds the default section from open native sessions and recent workspace conversation memberships.
5. It syncs the tray.

### Search Input

1. User edits the search input.
2. `ChatScreen` stores the query and starts a debounced search.
3. The search captures workspace id and query.
4. Results are applied only if captured workspace id and query still match current state.
5. The tray renders capped results and count copy.

### Row Selection

1. User selects a row.
2. Existing native-session switch or persisted-conversation resume path runs.
3. Selected marker and selected summary update.
4. Active search remains active.
5. Composer focus returns as it does today.

### Clear Search

1. User clears the input or activates the clear control.
2. Query becomes empty.
3. `ChatScreen` rebuilds default recent/open rows for the active workspace.

## Error Handling And Edge Cases

- If the workspace registry cannot be read, keep existing workspace recovery copy and disable conversation search.
- If persisted conversation search fails, keep selected summary and open/recent rows visible, then show a scoped warning inside the Conversations subsection.
- Empty search results show `No matches in this workspace.`
- Capped results show explicit copy such as `Showing 50 of 143`.
- If a row points to a persisted conversation that no longer exists, use the existing resume failure path and refresh the current workspace/query state.
- Membership-only rows still render in the default list even if persisted conversation search cannot find them.
- Open native sessions remain reachable even before a persisted conversation id exists.
- The selected summary remains visible outside the bounded scroll list.
- With many rows, both the bounded list and the outer left rail must still allow keyboard and scroll access to lower workspace status, server readiness, and handoff rows.

## Accessibility And Keyboard Behavior

- Header toggle, search input, clear affordance, rows, selected summary, `New conversation`, and lower workspace status/handoff controls remain keyboard reachable.
- Search input keeps focus while results refresh.
- Collapsed state can be toggled without requiring mouse interaction.
- Row labels remain rail-safe and must not render markup from conversation titles.
- Search status, empty state, and error copy are visible text inside the subsection, not notification-only state.

## Testing And Acceptance

Add focused mounted tests around the Console workspace rail:

- Many active-workspace conversations do not grow `#console-workspace-conversations` beyond its adaptive bound.
- Lower workspace status, server readiness, and handoff rows remain reachable with many conversations.
- Collapsing Conversations persists per workspace and leaves selected conversation summary visible.
- Switching workspace clears search text and loads that workspace's collapse preference.
- Searching queries all saved persisted conversations in the active workspace.
- Searching does not leak another workspace's matching conversations.
- Matching open native sessions appear in search results.
- Selecting a search result resumes or switches the conversation while keeping the search query active.
- Search cap, empty state, and error state render explicit copy.
- Expanded Conversations exposes the existing workspace-scoped `New conversation` action.
- Collapsed Conversations does not show the `New conversation` action.
- Search refresh guards stale workspace/query results.

Verification should include:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_console_workspace_context_rail.py Tests/UI/test_console_native_chat_flow.py -k "workspace_conversation or workspace_rail or workspace_switch or default_workspace" --tb=short
git diff --check
```

Because this is a visible layout fix, final implementation should also capture Textual-web/CDP evidence showing many conversations in the left rail with lower workspace status/handoff content still reachable.

## ADR Check

ADR required: no
ADR path: N/A
Reason: this changes Console presentation and UI preference state only. It does not change workspace ownership, storage/schema, sync policy, provider/runtime boundary, server handoff contract, or data ownership.
