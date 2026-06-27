# Console Grouped Conversation Browser Design

Date: 2026-06-27
Status: Approved by user through brainstorming, pending written-spec review
Primary Repo: `tldw_chatbook`
Scope: Console left Context rail, grouped conversation browsing, workspace resume behavior, and local-only conversation stars

## Summary

Console should let users browse and resume conversations across all workspaces at once while preserving the active workspace as the authority for new chats, staging, tool eligibility, sync/server readiness, and handoff state.

The current Console `Convos & Workspaces` rail subsection lists conversations only for the active workspace. This design replaces that subsection with a grouped browser:

- `Starred`
- `Workspaces`
- `Chats`

`Starred` provides quick access to locally marked conversations. `Workspaces` groups workspace-scoped conversations beneath collapsible workspace headers. `Chats` groups true global conversations and built-in Default workspace conversations for user-facing unscoped browsing while keeping their internal storage scopes distinct.

Conversation stars are durable local-only organization metadata. They are intentionally excluded from Sync v2, server payloads, and chat metadata mirror reports. This storage and sync boundary is recorded in [ADR-010](../../../backlog/decisions/010-console-conversation-local-marks.md).

## Current Context

Relevant existing seams:

- `tldw_chatbook/UI/Screens/chat_screen.py` owns Console session state, workspace switching, search state, row selection, and persisted conversation resume.
- `tldw_chatbook/Widgets/Console/console_workspace_context.py` renders the current `Convos & Workspaces` tray.
- `tldw_chatbook/Workspaces/display_state.py` defines `ConsoleWorkspaceContextState`, `ConsoleWorkspaceConversationRow`, and the current active-workspace conversation-section state.
- `tldw_chatbook/Workspaces/registry_service.py` lists workspaces and workspace conversation memberships.
- `tldw_chatbook/Chat/chat_conversation_service.py` lists and searches persisted conversations by scope.
- `tldw_chatbook/Chat/chat_conversation_scope_service.py` routes local/server conversation calls behind policy enforcement.
- `tldw_chatbook/Chat/console_chat_store.py` tracks open native Console sessions before or after durable persistence.

The recent Console conversation rail overflow work already made the active-workspace conversation list bounded, searchable, and collapsible. This design must preserve that safety while expanding the browser from one active workspace to all workspace and unscoped groups.

ADR-005 remains the active workspace boundary: Console workspace switching is local-first, no background sync or server hydration is implied, and the built-in Default workspace remains safe, local-only, and file-tools disabled.

## Goals

- Replace the active-workspace-only `Conversations` subsection with a grouped all-workspaces browser inside `Convos & Workspaces`.
- Show `Starred` above workspace and unscoped groups for quick access.
- Show workspace-scoped conversations beneath collapsible workspace headers.
- Show conversations with no associated user workspace under a `Chats` or `Unscoped Chats` category.
- Treat global conversations and built-in Default workspace conversations as one user-facing unscoped category while keeping their internal scopes distinct.
- Keep active workspace status, sync/server readiness, runtime, and handoff rows below the grouped browser.
- Opening a workspace-scoped conversation switches the active workspace to that conversation's workspace before resume.
- Opening a true global conversation does not change active workspace.
- Opening a built-in Default workspace conversation preserves Default workspace safety semantics.
- Provide one search input that filters all groups while preserving group labels.
- Remember group expanded/collapsed state locally.
- Store stars as durable local-only marks that survive restarts but are not synced.
- Keep the grouped browser bounded so lower rail status/readiness/handoff rows remain reachable.
- Preserve native open-session rows before durable conversation persistence.
- Verify with pure tests, service tests, mounted Console tests, sync exclusion tests, and screenshot/CDP evidence.

## Non-Goals

- Do not add conversation stars to the `conversations` table.
- Do not sync conversation stars.
- Do not send conversation stars to server APIs.
- Do not include conversation stars in Sync v2 chat metadata outbox records or mirror reports.
- Do not implement full conversation management, deletion, bulk actions, drag/drop reordering, or workspace reassignment.
- Do not change workspace membership semantics.
- Do not change Default workspace file-tool or runtime restrictions.
- Do not make Library Conversations the primary quick-resume surface. Library remains the broader source inspection and handoff browser.
- Do not implement server-backed workspace hydration or remote-only workspace browsing in this slice.

## ADR Check

ADR required: yes
ADR path: `backlog/decisions/010-console-conversation-local-marks.md`
Reason: This design chooses storage ownership and sync exclusion for durable conversation stars, which affects schema, data ownership, and future sync/server boundaries.

## Approved Direction

Use a grouped Console browser with a dedicated local-only marks table.

Rejected alternatives:

- Store stars in app config: simple, but too fragile and not truly attached to a conversation.
- Add `starred` directly to `conversations`: durable, but it risks accidental inclusion in sync, mirror reports, or server payloads.
- Defer stars and only group by workspace: lower risk, but it misses the quick-access requirement and would force a second pass through the same rail.

The selected approach keeps stars durable and local-only, while preserving explicit workspace scope and Console's active-workspace contract.

## UX Model

The `Convos & Workspaces` section keeps its existing workspace selector and recovery/status rows. The conversation browser within it changes from one active-workspace list to grouped sections.

Suggested layout:

```text
Convos & Workspaces
Workspace: Research

Search conversations

Starred
  * Resume design review        Research       2h
  * Provider investigation      Chats          1d

Workspaces
  - Research
      Current workspace chat                  2h
      Notes import follow-up                  1w
  + Personal
  - tldw_chatbook
      Review UX and improve it                1mo

Chats
  Help me fix the mermaid diagram             1w
  Plan focaccia dinner                        1mo

Storage: local
Sync: not configured
File tools: disabled
Server handoff: unavailable
```

Textual controls should use compact symbols for star/unstar and collapse/expand, with tooltips for clarity. The star control must stay narrow enough for the Console rail. Row labels must truncate safely without rendering markup from conversation titles.

### Group Defaults

- `Starred` is expanded by default.
- The active workspace group is expanded by default.
- Workspaces with recent activity may be expanded when rail space allows.
- Other workspace groups are collapsed by default.
- `Chats` is expanded by default when it has recent unscoped activity, otherwise it can follow remembered local preference.
- User toggles persist as Console UI preferences, not workspace registry data.

### Search

One search input filters `Starred`, `Workspaces`, and `Chats` together. Group labels remain visible in filtered results so users can see where each match belongs.

Search remains transient. It should reset on Console recreation. It should not change group collapse preferences permanently.

When search is active, matching groups are temporarily shown with their matching rows even if the group is normally collapsed. Clearing search restores the remembered collapse state. Groups with no matches may remain hidden or render an explicit no-match empty state, but matching rows must not be hidden behind a collapsed group during search.

### Row Selection

Selecting a row uses one resolver:

1. Resolve the row's scope before opening anything.
2. If the row belongs to the built-in Default workspace, switch active workspace to Default and preserve Default workspace semantics without granting file/runtime authority.
3. If the row is scoped to any other workspace, switch active workspace to the row's workspace first. This applies to both open native sessions and persisted conversations.
4. If the row is true global, keep the current active workspace unchanged.
5. If the row maps to an open native Console session, switch to that session.
6. Otherwise, resume the persisted conversation through the existing local conversation service path.

Rows should make the scope visible enough to avoid surprise. The selected row or inspector/status copy should indicate when a resume changed the active workspace.

### Star/Unstar

Star and unstar are available from each conversation row where a durable conversation id exists. Native sessions that have not yet persisted can show the star control disabled with a tooltip such as `Send or save this conversation before starring.`

Starred conversations still appear in their normal `Workspaces` or `Chats` group. The row state must remain consistent in both places. Deduplication applies while merging source records into one canonical row per group, not across final rendered sections; the same starred conversation can intentionally render once in `Starred` and once in its normal group.

### Ordering And Bounds

The grouped browser must remain bounded independently of the outer rail. Within that bound, the selected/open conversation should remain easy to find, then rows should favor recent activity. `Starred` should favor recently starred or recently active conversations. Workspace groups should put the active workspace first, then recently active workspaces, then stable workspace-name order. Exact row caps can be tuned during implementation, but capped groups need explicit copy when more rows are available.

## Architecture

### Presentation State

Add a pure grouped browser display-state model. It can live beside the existing workspace display state or in a new Console display-state module if that keeps boundaries clearer.

Suggested shape:

```python
@dataclass(frozen=True)
class ConsoleConversationBrowserRow:
    conversation_id: str
    title: str
    scope_type: str
    workspace_id: str | None
    workspace_label: str
    status: str
    updated_label: str
    selected: bool = False
    starred: bool = False
    star_enabled: bool = True
    source_kind: str = "persisted"  # persisted | native

@dataclass(frozen=True)
class ConsoleConversationBrowserGroup:
    group_id: str
    label: str
    collapsed: bool
    rows: tuple[ConsoleConversationBrowserRow, ...]
    count: int
    empty_copy: str = ""

@dataclass(frozen=True)
class ConsoleConversationBrowserState:
    query: str
    groups: tuple[ConsoleConversationBrowserGroup, ...]
    selected_summary: str
    status_copy: str = ""
    error_copy: str = ""
    marks_available: bool = True
```

Exact names can change during implementation. The contract matters: the widget renders state and does not query services or mutate app state.

### Local Marks Service

Add `ConversationLocalMarksService` as a local-only service. It should be separate from normalized conversation metadata and sync surfaces.

Suggested table:

```sql
CREATE TABLE IF NOT EXISTS conversation_local_marks (
  conversation_id TEXT NOT NULL,
  mark_type TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (conversation_id, mark_type)
);
CREATE INDEX IF NOT EXISTS idx_conversation_local_marks_type
  ON conversation_local_marks(mark_type, updated_at DESC, conversation_id);
```

Initial supported mark:

- `starred`

Service responsibilities:

- `star_conversation(conversation_id)`
- `unstar_conversation(conversation_id)`
- `is_starred(conversation_id)`
- `list_marked_conversation_ids(mark_type="starred", limit=...)`
- tolerate missing/deleted conversations
- optionally expose cleanup for orphan marks

It should not update `conversations`, workspace memberships, sync outbox rows, mirror reports, or server schemas.

### Screen Ownership

`ChatScreen` owns:

- browser query text
- async/debounced search token
- group collapse preferences
- row selection and resume behavior
- active workspace switching before workspace-scoped resume
- star/unstar commands
- service calls
- tray state synchronization

`ConsoleWorkspaceContextTray` owns:

- rendering grouped state
- emitting Textual button/input events
- preserving accessible focus and compact controls
- not directly mutating state

This matches the current Console rail pattern and keeps state mutations in the screen/controller layer.

## Data Flow

### Mount Or Refresh

1. `ChatScreen` reads current active workspace context.
2. It reads workspaces and conversation memberships from `workspace_registry_service`.
3. It reads recent/searchable persisted conversations from `chat_conversation_scope_service` in local mode.
4. It reads starred conversation ids from `ConversationLocalMarksService`.
5. It merges open native sessions from `ConsoleChatStore`.
6. It dedupes rows by persisted conversation id when available, otherwise by `native:<session-id>`.
7. It builds `Starred`, `Workspaces`, and `Chats` groups.
8. It applies collapse preferences and syncs the tray.

### Search

1. User edits the search field.
2. `ChatScreen` stores query and increments a search token.
3. It filters fast local/native/membership rows immediately when possible.
4. It requests persisted results through existing local conversation services.
5. Results apply only when token and query still match current state.
6. Partial failures keep available groups visible and show scoped copy.

### Star/Unstar

1. User activates the row star control.
2. `ChatScreen` calls `ConversationLocalMarksService`.
3. On success, rebuild grouped state.
4. On failure, preserve browsing and show scoped warning copy; disable mark controls if storage remains unavailable.

### Row Selection

1. User selects a conversation row.
2. The resolver determines whether the row is true global, built-in Default workspace, or another workspace-scoped conversation.
3. Built-in Default workspace rows switch active workspace to Default while preserving Default safety restrictions.
4. Other workspace-scoped rows switch active workspace first, including open native sessions and persisted conversations.
5. True global rows keep the current active workspace unchanged.
6. The resolver switches to an open native session when present; otherwise it resumes persisted messages into a native Console session.
7. The browser, active workspace status rows, transcript, and composer update together.

## Error Handling And Edge Cases

- Marks storage unavailable: browsing still works; star controls disable; scoped warning appears.
- Workspace registry unavailable: show available `Starred` and `Chats` rows when possible, plus existing workspace recovery copy.
- Conversation search unavailable: show native/open and membership rows, plus partial-result copy.
- Starred conversation deleted or missing: omit from `Starred`; do not fail browser render.
- Starred conversation also appears in a group: keep star state consistent in both rows.
- Native session not persisted: render in its proper group but disable star until a durable conversation id exists.
- Large workspace sets: keep the grouped browser bounded and let its own scroll area handle long lists.
- Long titles: truncate safely, no markup interpretation.
- Default workspace: display under `Chats`, but keep internal `workspace-default` scope and safety policy.
- Remote-only/server workspaces: do not show as resumable unless existing local services can truthfully resolve them.

## Accessibility And Keyboard Behavior

- Search input, clear search, group toggles, row selection, and star/unstar controls must be keyboard reachable.
- Star controls need clear tooltips and accessible labels.
- Focus should remain stable when search results refresh.
- Row selection should return focus to the composer when appropriate, matching current Console behavior.
- Group toggles should not trap focus or resize controls unpredictably.
- Empty/error/status copy must render inside the rail, not only as notifications.

## Testing

### Pure Tests

- grouping of starred, workspace-scoped, global, and Default workspace conversations
- dedupe while merging source rows within each rendered group, while still allowing a starred conversation to appear in `Starred` and its normal group
- default group expansion rules
- search filtering across all groups
- search temporarily exposes matching rows from collapsed groups without changing saved collapse preferences
- global versus Default workspace display grouping with distinct internal scope
- row label truncation and no-markup safety

### Service Tests

- local marks star/unstar CRUD
- idempotent star/unstar
- `list_marked_conversation_ids` ordering and limit behavior
- orphan/missing conversation tolerance
- no mutation of conversation rows

### Mounted Console Tests

- grouped browser renders in `Convos & Workspaces`
- active workspace status/readiness/handoff rows remain below and reachable
- star/unstar updates `Starred`
- selecting workspace-scoped row switches active workspace and resumes transcript
- selecting global row does not switch active workspace
- Default workspace row remains safe and does not enable file tools
- search filters all groups and ignores stale async results
- collapse preferences persist locally
- keyboard focus and controls remain usable
- long lists do not recreate rail overflow

### Sync Exclusion Tests

- local marks are not included in Sync v2 chat outbox records
- local marks are not included in chat metadata mirror reports
- local marks are not sent through server conversation payloads

### Visual Evidence

Capture Textual-web/CDP evidence for:

- default grouped browser
- expanded active workspace group
- search results across groups
- starred quick-access section
- long grouped list with lower readiness rows still reachable

## Open Questions

No unresolved product questions remain from brainstorming.

Implementation can tune exact labels, row density, and module placement as long as it preserves the storage, sync, active workspace, and grouping contracts above.
