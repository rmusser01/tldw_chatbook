# Use In Chat Handoffs Design

Date: 2026-04-21
Last Updated: 2026-04-29
Status: Re-baselined against current `dev`
Branch Context: Current `dev` after the runtime/server-parity, shell-context, workspace, media-runtime, and search modularization work
Scope: Single-item chat-first handoffs from Notes, Workspaces, Media, and Search into a new Chat session

## Summary

This slice adds explicit `Use in Chat` handoffs for one selected source item at a time. A handoff opens a brand-new Chat session, stages the selected source as visible context, and prefills the composer with a draft prompt without auto-sending anything.

The updated design reflects the current `dev` branch rather than the older shell-cleanup branch. Chat now has a richer session contract, a visible shell bar, optional tabbed sessions, source-aware runtime state, and server/local separation. The handoff design should use those seams directly instead of introducing parallel state.

## User-Validated Decisions

The design is anchored to the following explicit user decisions:

- `Use in Chat` always opens a new Chat session.
- Chat shows a visible handoff card and prefills a draft prompt in the composer.
- The draft is not auto-sent.
- The first pass only supports single-item handoffs.
- The first pass wires `Notes`, `Media`, and `Search`.
- Search support includes both RAG results and Web Search results.
- Workspaces must be handled as part of the current Notes/learning workflow, not treated as an unrelated future destination.

## Current Dev Snapshot

The current branch has materially changed since the original spec:

- `TldwCli` uses `NavigateToScreen` as the canonical screen-router message and persists screen state before switching.
- `TldwCli.open_study_screen()` and `TldwCli.open_notes_workspace()` already use app-owned pending context before navigation.
- There is no current `open_chat_with_handoff()` or `pending_chat_handoff` implementation in code.
- `ChatWindowEnhanced` mounts `ChatShellBar` and `ChatTaskCards` above the chat content.
- `ChatSessionData` and `TabState` already preserve `runtime_backend`, `discovery_owner`, `discovery_entity_id`, `assistant_kind`, `assistant_id`, `scope_type`, and `workspace_id`.
- Neither `ChatSessionData` nor `TabState` currently has a handoff payload, staged context, or draft-prompt field.
- Chat tabs are still optional via `chat_defaults.enable_tabs`, and the default config currently sets `enable_tabs = false`.
- `ChatTabContainer.create_new_tab()` reuses tabs only when `conversation_id` is present and matches `(runtime_backend, conversation_id)`.
- `ChatScreen` restores saved tabs/input/messages after mount on a short timer, so handoff consumption must run after normal restore.
- `NotesScreen` now owns a scope-aware `NotesScreenState` with `local_note`, `server_note`, and `workspace` modes plus workspace subviews for `notes`, `details`, `sources`, and `artifacts`.
- `NotesScreen` keeps a cached `_workspace_context_payload` for workspace details, notes, sources, and artifacts.
- `MediaScreen` wraps `MediaWindow_v2`, which uses shared `MediaRuntimeState` for `runtime_backend`, active browse subview, selected record ID, detail cache, reading progress, and ingestion-source caches.
- `MediaViewerPanel` owns the visible selected media data and has an existing metadata `Actions` area.
- `SearchWindow` routes RAG Q&A through the compatibility export `UI/SearchRAGWindow.py`, backed by `UI/Views/RAGSearch`.
- `SearchResult` cards already have per-result actions for `View`, `Copy`, `Note`, and `Export`, but not `Use in Chat`.
- RAG Q&A can include web results when web search is enabled, while the dedicated Web Search subview still renders Markdown output rather than reusable result cards.

## Goals

- Let users move one selected source item into Chat with one obvious action.
- Preserve the current source's runtime backend, discovery owner, scope, and workspace identity in the new Chat session.
- Make staged context visible before anything is sent to a model.
- Keep handoff behavior consistent across Notes, Workspaces, Media, RAG Search, and Web Search.
- Reuse app-owned pending navigation state instead of coupling source widgets directly to mounted Chat widgets.
- Keep the first implementation small enough to test and review without blocking broader server-parity work.

## Non-Goals

- Multi-select, package, or chatbook-style grouped handoffs.
- Auto-sending prompts after handoff.
- Reusing the currently active Chat session.
- Silent fallback to legacy single-session Chat when tabbed Chat is unavailable.
- Rebuilding the whole Notes, Media, or Search information architecture.
- Replacing existing CCP/persona launch flows.
- Implementing Study, Flashcards, or Quizzes handoffs in this slice.

## UX Principles

This slice should follow Nielsen Norman Group usability heuristics:

- Visibility of system status: Chat must show that context is staged, which source it came from, and whether it has been sent.
- Match between system and real world: labels should say `Use in Chat`, `Context staged`, `Workspace source`, or `Web result`, not internal route IDs.
- User control and freedom: users can edit or clear the draft before sending; nothing is sent automatically.
- Error prevention: unavailable handoffs should be disabled or warned before navigation, especially when tabs are off or no valid item is selected.
- Consistency and standards: Notes, Workspaces, Media, RAG, and Web Search should all use the same payload and destination behavior.
- Recognition rather than recall: the Chat shell and handoff card should expose backend, scope, title, source type, and key metadata.

## Recommended Architecture

### App-Owned Handoff Entry Point

Add one app-owned helper:

```python
def open_chat_with_handoff(self, payload: ChatHandoffPayload) -> None:
    self.pending_chat_handoff = payload
    self.post_message(NavigateToScreen(TAB_CHAT))
```

This mirrors the existing `open_study_screen()` and `open_notes_workspace()` pattern. Source surfaces should not query for a mounted Chat widget or mutate a Chat tab directly. The app owns navigation; Chat owns session creation and payload consumption.

Because the user selected `always open a new Chat session`, this helper must fail closed when tabbed Chat is unavailable. Legacy single-session Chat cannot safely satisfy the requirement. The source action should either be disabled with a clear reason or notify: `Use in Chat requires chat tabs to be enabled.`

The current default `enable_tabs = false` is a product risk. If this feature is shipped while tabs remain off by default, most users will experience `Use in Chat` as broken. The implementation should either enable tabs as part of this UX slice or make the unavailable state explicit wherever the action appears.

### Shared Payload Contract

Create a shared handoff model rather than passing ad hoc dictionaries:

```python
@dataclass
class ChatHandoffPayload:
    source: str
    item_type: str
    title: str
    body: str
    source_id: str | None = None
    display_summary: str = ""
    suggested_prompt: str = ""
    runtime_backend: str = "local"
    discovery_owner: str = "general_chat"
    discovery_entity_id: str | None = None
    scope_type: str | None = None
    workspace_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

Recommended `source` values:

- `notes`
- `workspace`
- `media`
- `search-rag`
- `search-web`

Recommended `item_type` values:

- `note`
- `workspace`
- `workspace-source`
- `workspace-artifact`
- `media`
- `rag-result`
- `web-result`

The payload should be serializable through `to_dict()` / `from_dict()` helpers because it must live in `ChatSessionData`, `TabState`, and saved screen state.

### Runtime And Scope Contract

Every payload must populate the fields already used by the current Chat shell:

- `runtime_backend`: `local` or `server`
- `discovery_owner`: stable source owner such as `notes`, `workspace`, `media`, `rag_search`, or `web_search`
- `discovery_entity_id`: selected source item ID when available
- `scope_type`: `workspace` when the source is workspace-bound, otherwise `global` or source-specific neutral value
- `workspace_id`: selected workspace ID for workspace notes, workspace details, workspace sources, and workspace artifacts

The destination `ChatSessionData` should copy these fields from the payload so `ChatShellBar` immediately shows the correct backend and scope.

### Staged Context Contract

The handoff card is not enough by itself. The staged source content must also be available to the first user send after handoff.

The required behavior is:

1. `Use in Chat` creates a new chat tab/session with a session-scoped handoff payload.
2. The chat log renders a visible `Context staged` card for that payload.
3. The composer is prefilled with the suggested draft prompt.
4. No request is sent until the user presses Send.
5. On Send, the chat request builder includes the staged handoff context in the outgoing request in an auditable way.
6. The UI must never label the context as sent before Send is pressed.

This avoids a misleading UX where Chat says a note or media item is being used, but the model only receives a generic prompt with no source content.

## Surface Adapters

### Notes And Workspaces

`NotesScreen` should expose `Use in Chat` for the current valid single selection.

For note-editor contexts, build from:

- `state.selected_note_id`
- `state.selected_note_version`
- `state.selected_note_title`
- current editor text, not stale saved content
- `_selected_note_keywords`
- `state.scope_type`

For workspace contexts, build from:

- `state.selected_workspace_id`
- `state.workspace_subview`
- `state.selected_workspace_*` IDs and versions
- `_workspace_context_payload["workspace"]`
- selected workspace note/source/artifact records when applicable
- current panel field values when the user is editing details, sources, or artifacts

Recommended mapping:

- Local note: `source=notes`, `item_type=note`, `runtime_backend=local`
- Server note: `source=notes`, `item_type=note`, `runtime_backend=server`
- Workspace note: `source=workspace`, `item_type=note`, `runtime_backend=server`, `scope_type=workspace`
- Workspace details: `source=workspace`, `item_type=workspace`, `runtime_backend=server`, `scope_type=workspace`
- Workspace source: `source=workspace`, `item_type=workspace-source`, `runtime_backend=server`, `scope_type=workspace`
- Workspace artifact: `source=workspace`, `item_type=workspace-artifact`, `runtime_backend=server`, `scope_type=workspace`

If the source has unsaved editor or panel changes, the handoff should use the visible current text and metadata, mark `metadata["unsaved_changes"] = True`, and avoid silently reloading older saved content.

Recommended placement:

- Add `Use in Chat` to `NotesSidebarRight` near `Save All Changes` for note editor contexts.
- Add `Use in Chat` to `WorkspaceContextPanel` near `Open Study` for workspace details.
- Add `Use in Chat` to the source/artifact action rows for workspace source and artifact subviews.

Recommended draft prompts:

- Note: `Use this note as context and help me work with it.`
- Workspace: `Use this workspace context and help me plan the next step.`
- Workspace source: `Use this workspace source as context and help me reason from it.`
- Workspace artifact: `Use this workspace artifact as context and help me improve it.`

### Media

`MediaWindow_v2` should expose `Use in Chat` for the currently selected hydrated media item.

Build from:

- `runtime_state.runtime_backend`
- `runtime_state.selected_record_id`
- `runtime_state.detail_by_record_id[selected_record_id]`
- `viewer_panel.media_data` as the visible fallback source of truth
- media title, content, summary, URL, author, media type, keywords, reading progress, and highlights

The adapter should prefer hydrated detail over list-row data. If hydrated content is missing, it should still create a payload from title, URL, metadata, and any available summary, but the card must indicate that full content was not available.

Recommended mapping:

- `source=media`
- `item_type=media`
- `runtime_backend=media_runtime_state.runtime_backend`
- `discovery_owner=media`
- `discovery_entity_id=record_id`

Recommended placement:

- Add `Use in Chat` inside `MediaViewerPanel`'s metadata `Actions` collapsible near `Edit`, `Save for Later`, and `Delete`.
- Disable it when `media_data` is empty.

Recommended draft prompt:

- `Use this media item as context and help me analyze or summarize it.`

### Search

Search should use per-result `Use in Chat` actions. A screen-level selected-result model is not needed for the first pass.

For RAG Q&A, add the action to `SearchResult` cards. `SearchResult` should emit a dedicated Textual message such as `UseInChatRequested(index, result)`; it should not call `app.open_chat_with_handoff()` directly. `SearchRAGWindow` should normalize the result and forward the payload to the app helper.

For Web Search, the current branch has two paths:

- Web results included in RAG Q&A already become `SearchResult` cards and can share the same action.
- The dedicated Web Search subview currently renders Markdown, so it must either be converted to reusable result cards in this slice or be explicitly marked unavailable until it is cardified.

Because the user validated Web Search support, the preferred implementation is to cardify dedicated Web Search results using the same `SearchResult` component rather than limiting support to the RAG Q&A include-web checkbox.

Recommended mapping:

- RAG result: `source=search-rag`, `item_type=rag-result`, `discovery_owner=rag_search`
- Web result: `source=search-web`, `item_type=web-result`, `discovery_owner=web_search`

Payload metadata should include:

- score
- source kind
- citations
- result metadata
- source URL and display URL for web results
- search mode and query when available

Recommended draft prompts:

- RAG: `Use this retrieved result as context and answer or reason from it carefully.`
- Web: `Use this web result as source context and preserve attribution in your answer.`

## Chat Destination Behavior

### New Session Rule

Every handoff creates a fresh ephemeral Chat session.

Destination session construction should:

- set `conversation_id=None`
- set `is_ephemeral=True`
- copy runtime/scope/discovery fields from the payload
- choose a useful title such as `Note: <title>`, `Media: <title>`, or `Search: <title>`
- not set any conversation reuse key

This works with current `ChatTabContainer` behavior because reuse only happens when `conversation_id` is present.

### Pending Handoff Lifecycle

The app stores the pending payload during navigation. `ChatScreen` consumes it only after normal state restoration.

Required lifecycle:

1. Source surface validates selection and builds `ChatHandoffPayload`.
2. Source calls `app_instance.open_chat_with_handoff(payload)`.
3. App stores `pending_chat_handoff`.
4. App navigates to `TAB_CHAT`.
5. `ChatScreen` mounts and finishes normal saved-state restore.
6. `ChatScreen` consumes `pending_chat_handoff` in a post-restore phase.
7. `ChatScreen` creates a fresh tab through `ChatTabContainer.create_new_tab(session_data=...)`.
8. `ChatScreen` mounts the visible handoff card and prefills the new session's input.
9. `ChatScreen` clears `pending_chat_handoff` only after successful tab creation and UI application.

If tab creation fails because of max tabs, missing tab container, or widget errors, the pending payload should remain available for debugging and the user should see a warning.

### Session State

The payload must become session-owned after consumption.

Add a session-scoped field such as `handoff_payload` to:

- `ChatSessionData`
- `TabState`

The field must serialize to saved Chat state. The handoff card and draft prompt should survive:

- tab switches
- navigation away from Chat
- screen restore
- app state save/restore within the existing screen-state boundary

### Visible Handoff Card

The handoff card should be visually distinct from user and assistant messages. It should not be stored as a fake assistant, user, or system message.

The card should show:

- status: `Context staged`
- source label such as `Notes`, `Workspace`, `Media`, `RAG Search`, or `Web Search`
- item type
- title
- short summary or excerpt
- backend and workspace chips where applicable
- source metadata such as URL, score, version, media type, or updated time
- hint: `Review the draft below and send when ready.`

The card may live inside the chat log for locality, but persistence should come from `TabState.handoff_payload`, not from message extraction.

### Draft Prompt

The destination composer should be prefilled from `payload.suggested_prompt`.

Fallback draft:

```text
Help me use this context.
```

Nothing is sent automatically. The user can edit or replace the draft before Send.

### Send-Time Context

The first send from a handoff session must include the staged context in the LLM request. The implementation should keep this explicit and auditable.

Recommended first-pass behavior:

- Attach the handoff body and metadata to the outgoing prompt/context assembly for that session.
- Mark the session payload as sent or used after the first successful send.
- Keep the card visible after send, with status changed from `Context staged` to `Context sent`.
- Do not resend large handoff context on every later message unless the user explicitly asks for persistent context in a later slice.

If the existing chat event handlers cannot support this cleanly in the first implementation pass, the draft prompt must visibly include the context text that will be sent. A hidden context that is never sent is not acceptable.

## Error Handling

- No valid Notes, Workspace, or Media selection: disable `Use in Chat` or show a short warning without navigating.
- Unsaved Notes/Workspace changes: use visible current content and mark the payload as unsaved, or ask the user to save if the adapter cannot safely read the visible state.
- Tabs disabled: show an explicit unavailable state; do not inject into legacy single-session Chat.
- Max tabs reached: notify and keep pending handoff uncleared.
- Source deleted or stale by the time the action is pressed: notify and do not navigate.
- Sparse source body: create the card with a metadata-only summary and use the neutral prompt.
- Web search dependency unavailable: do not show Web `Use in Chat` actions.
- Runtime/backend mismatch: preserve the source backend in the payload and destination session instead of silently switching to the currently active global backend.

## Testing Strategy

Create focused tests for the shared contract and each source adapter.

Minimum required coverage:

- `ChatHandoffPayload` serializes and deserializes without losing runtime, scope, workspace, source ID, or metadata.
- `ChatSessionData.to_dict()` / `from_dict()` preserves `handoff_payload`.
- `TabState.to_dict()` / `from_dict()` preserves `handoff_payload`.
- `open_chat_with_handoff()` stores pending payload and navigates to `TAB_CHAT`.
- `ChatScreen` consumes pending handoff only after restore.
- Handoff-created sessions always pass `conversation_id=None`.
- Existing `ChatTabContainer` reuse behavior does not reuse a prior persisted conversation for handoff sessions.
- Tabs-disabled mode refuses the handoff cleanly and does not mutate the legacy single-session conversation.
- Notes payloads cover local note, server note, workspace note, workspace details, workspace source, and workspace artifact.
- Notes dirty-state handoff uses visible current content or blocks with a clear warning.
- Media payloads use hydrated detail from `MediaRuntimeState` / `MediaViewerPanel`.
- RAG result cards emit a window-level handoff event and normalize a payload.
- Web results from both include-web RAG results and the dedicated Web Search view can produce payloads, or the dedicated view shows an explicit unavailable state.
- The handoff card renders as a distinct card, not as a user/assistant message.
- The destination draft is prefilled and no send handler is called during handoff.
- The first actual send includes staged context in the outgoing request.
- The pending app-level handoff clears after successful consumption and does not replay on later Chat visits.
- Handoff payload and draft survive tab switching and screen restore.

Recommended test files:

- create `Tests/UI/test_chat_first_handoffs.py`
- extend `Tests/UI/test_chat_screen_state.py`
- extend `Tests/UI/test_chat_tab_container.py`
- extend `Tests/UI/test_chat_window_enhanced.py`
- extend `Tests/UI/test_notes_screen.py`
- extend or create Media window/viewer tests near existing Media coverage
- extend RAG Search component tests or create `Tests/UI/test_search_handoffs.py`

## File Impact

Recommended implementation touch points:

- `tldw_chatbook/Chat/chat_handoff_models.py`
  new shared payload contract and serialization helpers
- `tldw_chatbook/app.py`
  add app-owned pending handoff state and `open_chat_with_handoff()`
- `tldw_chatbook/Chat/chat_models.py`
  add session-scoped handoff payload to `ChatSessionData`
- `tldw_chatbook/UI/Screens/chat_screen_state.py`
  add persisted handoff payload to `TabState`
- `tldw_chatbook/UI/Screens/chat_screen.py`
  consume pending handoff after restore, create a fresh session, mount card, prefill input, and clear pending state
- `tldw_chatbook/Widgets/Chat_Widgets/chat_handoff_card.py`
  render consistent staged/sent handoff context
- `tldw_chatbook/Widgets/Chat_Widgets/chat_session.py`
  provide a clean seam for mounting the card and setting the per-tab draft input
- `tldw_chatbook/Event_Handlers/Chat_Events/chat_events_tabs.py`
  include staged context in the first outgoing tabbed-chat request
- `tldw_chatbook/UI/Screens/notes_screen.py`
  build Notes/Workspace payloads and handle the action
- `tldw_chatbook/Widgets/Note_Widgets/notes_sidebar_right.py`
  add note-context `Use in Chat` placement
- `tldw_chatbook/Widgets/Note_Widgets/workspace_context_panel.py`
  add workspace/source/artifact `Use in Chat` placements
- `tldw_chatbook/UI/Screens/media_screen.py`
  preserve wrapper-level access to shared media runtime state if needed by tests
- `tldw_chatbook/UI/MediaWindow_v2.py`
  build Media payloads and handle the viewer action
- `tldw_chatbook/Widgets/Media/media_viewer_panel.py`
  add viewer-level `Use in Chat` action event
- `tldw_chatbook/UI/Views/RAGSearch/search_result.py`
  add per-result `Use in Chat` action and event
- `tldw_chatbook/UI/Views/RAGSearch/search_rag_window.py`
  normalize RAG/Web result payloads and forward them through the app helper
- `tldw_chatbook/UI/SearchWindow.py`
  cardify dedicated Web Search results or show explicit unavailable state

## Implementation Order

1. Add the shared payload contract and Chat session persistence fields.
2. Add app-owned pending handoff navigation and Chat destination consumption.
3. Add the handoff card and draft-prefill behavior.
4. Add send-time context injection for the first user send.
5. Wire Notes and Workspace source adapters.
6. Wire Media source adapter.
7. Wire RAG Search and Web Search adapters.
8. Add focused tests and regression coverage.

## Recommendation

Implement `Use in Chat` as one app-owned pending-context seam with source-specific payload adapters and session-scoped staged context.

This is still the smallest coherent design, but the current `dev` branch requires stricter runtime and persistence handling than the original spec captured. The key correction is that a handoff must be both visible to the user and actually available to the model on first send; otherwise the UX would look successful while failing the core task.
