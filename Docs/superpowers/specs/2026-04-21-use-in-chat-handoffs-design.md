# Use In Chat Handoffs Design

Date: 2026-04-21
Status: Proposed
Branch Context: Extends `codex/chat-first-shell-label-cleanup`
Scope: Single-item chat-first handoffs from Notes, Media, and Search into a new Chat session

## Summary

This slice adds explicit `Use in Chat` handoffs for the currently selected Note, the currently selected Media item, and individual Search results. Each handoff opens a brand-new Chat session, shows a visible handoff card inside Chat, and prefills the composer with a draft prompt without auto-sending.

The goal is to make Notes, Media, and Search behave like structured context providers for the chat-first agentic workflow rather than detached destinations.

## User-Validated Decisions

The design is anchored to the following explicit user decisions:

- `Use in Chat` always opens a new Chat session.
- Chat shows a visible handoff card and prefills a draft prompt in the composer.
- The draft is not auto-sent.
- The first pass only supports single-item handoffs.
- The first pass wires `Notes`, `Media`, and `Search`.
- Search support includes both RAG results and Web Search results.

## Goals

- Let users move a single selected item from Notes, Media, or Search into Chat with one visible action.
- Keep the handoff UX consistent across all three surfaces.
- Preserve enough source metadata that the Chat shell and handoff card explain what moved into Chat.
- Reuse existing app-owned cross-screen handoff patterns where possible instead of adding surface-to-surface coupling.

## Non-Goals

- Multi-select or package handoffs.
- Auto-send after handoff.
- Reusing the currently active Chat session.
- A full `Library` destination refactor; this slice adapts the existing `Notes`, `Media`, and `Search` surfaces directly.
- `Study -> Chat`, `Characters -> Chat`, or `Chat -> Library` flows.

## Current Seam Reality

The current branch already has reusable seams that make a thin first pass practical:

- `NotesScreen` maintains current selection state and mirrors key fields like `current_selected_note_title` and `current_selected_note_content` back onto the app instance.
- `MediaWindow_v2` maintains a current selected record and hydrates media detail before showing the viewer/metadata surface.
- `SearchRAGWindow` renders each result as a `SearchResult` card with per-result action buttons, but it does not expose a durable selected-result contract yet.
- `TldwCli` already owns app-level handoff helpers for other destinations, such as `open_study_screen()` and `open_notes_workspace()`, by storing pending context on the app and navigating to the target screen.

Because of that, the lowest-friction design is app-owned pending handoff state plus thin per-surface payload adapters.

## Recommended Architecture

### App-Owned Handoff Entry Point

Introduce one app-owned helper, conceptually:

```python
def open_chat_with_handoff(self, payload: ChatHandoffPayload) -> None:
    self.pending_chat_handoff = payload
    self.post_message(NavigateToScreen(TAB_CHAT))
```

This mirrors the existing pending-context pattern already used by `open_study_screen()` and `open_notes_workspace()`.

The first pass should prefer this app-owned helper over introducing a heavier generic cross-screen message bus because:

- the app already owns cross-screen navigation
- surfaces already know their `app_instance`
- the target behavior is one-way and simple in this slice
- it avoids widget-to-widget coupling without adding another indirection layer too early

### Shared Payload Contract

Normalize every handoff into a single payload shape:

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
    metadata: dict[str, Any] = field(default_factory=dict)
```

Recommended source values:

- `notes`
- `media`
- `search-rag`
- `search-web`

Recommended item types:

- `note`
- `media`
- `rag-result`
- `web-result`

The payload should stay deliberately small and portable. It carries only what Chat needs to:

- render the handoff card
- prefill the draft prompt
- preserve source identity and explanatory metadata

## Surface Adapters

### Notes

`NotesScreen` should expose a visible `Use in Chat` action for the current single selection.

The Notes adapter should build a handoff payload from the current selected item already reflected in screen state:

- `selected_note_title`
- `selected_note_content`
- current scope information
- selected note ID/version where available

For the first pass:

- local note, server note, and workspace note selections are in scope
- workspace detail-only selections with no useful body should either disable the action or degrade to a summary-only payload
- the action should be disabled when no valid single item is selected

Recommended payload metadata:

- `scope_type`
- `workspace_id`
- `workspace_subview`
- `note_version`
- `keywords`

Recommended suggested prompt:

- `Use this note as context and help me work with it.`

### Media

`MediaWindow_v2` should expose a visible `Use in Chat` action for the currently selected and hydrated media item.

The Media adapter should build from the same detail record already shown in the viewer/metadata pane. The first pass should not require a separate selection model.

Recommended payload fields:

- title from the hydrated media detail
- excerpt/body from available summary/content fields
- normalized record ID
- backend/runtime metadata
- URL, file path, author, media type, and reading progress when available

Recommended suggested prompt:

- `Use this media item as context and help me analyze or summarize it.`

### Search

Search should use per-result `Use in Chat` actions rather than introducing a synthetic screen-level selected-result state.

This fits the existing Search UI because result cards already own result-specific actions such as `Copy`, `Note`, and `Export`.

The same result-card action should work for:

- RAG results
- Web Search results

The Search adapter should derive the payload directly from the result object:

- title
- content preview or body
- score
- source kind
- citations and metadata when present
- source URL for web results when available

Recommended suggested prompts:

- RAG: `Use this retrieved result as context and answer or reason from it carefully.`
- Web: `Use this web result as source context and preserve attribution in your answer.`

## Chat Behavior

### New Session Rule

Every `Use in Chat` action opens a new Chat session. This is an explicit, deliberate divergence from the earlier high-level rescue plan language about preserving the active session when possible.

For this slice, isolation is the intended behavior:

- no reuse of the current chat session
- no injection into another session’s ongoing context
- each handoff becomes its own fresh unit of work

### Pending Handoff Ownership

The app should store the pending handoff during navigation. `ChatScreen` consumes it when Chat opens.

Recommended target lifecycle:

1. source surface calls `app_instance.open_chat_with_handoff(payload)`
2. app stores `pending_chat_handoff`
3. app navigates to Chat
4. `ChatScreen` consumes the pending handoff
5. `ChatScreen` creates a brand-new chat tab/session
6. `ChatScreen` applies the handoff payload to that session
7. Chat clears the app-level pending handoff after consumption

### Visible Handoff Card

The new Chat session should render a dedicated handoff card near the top of the conversation area with:

- source label, such as `Context added from Notes`
- selected item title
- short explanatory summary
- relevant metadata chips or key-value details
- a clear next-step hint such as `Review the draft prompt below and send when ready.`

This card is required for visibility of system status and handoff confidence.

### Draft Prompt Behavior

The composer should be prefilled with the payload’s `suggested_prompt`.

Nothing is auto-sent.

If the payload is sparse, fall back to a neutral draft:

- `Help me use this context.`

### Session State

The handoff card and its payload should belong to the new chat session rather than to Chat globally. The recommended persistence boundary is session/tab state, so the handoff remains visible if the user navigates away and returns before acting on it.

That implies a small extension to chat session state, for example a session-scoped `pending_handoff` or `handoff_payload`, rather than global app-only transient state.

## UI Placement

The first pass should stay visually direct and avoid extra dialogs:

- `Notes`: add `Use in Chat` in the current Notes control/action area for the active selection
- `Media`: add `Use in Chat` in the current media viewer or metadata action area for the active item
- `Search`: add `Use in Chat` to each search result card action row

This keeps the action visible at the point of selection and supports the chat-first mental model without introducing a separate packaging flow.

## Error Handling

- If Notes or Media has no valid single item selected, the action should be disabled or produce a short warning.
- If Search result data is incomplete, build the best payload possible and degrade gracefully.
- If Chat navigation or new-session creation fails, notify the user and leave the source surface intact.
- If a source item lacks a meaningful body, the handoff card still appears and the draft falls back to a neutral prompt.
- The UI must not imply that content was sent to the model if only the handoff card and draft prompt were created.

## Testing Strategy

Create focused coverage in a new handoff test module plus small surface-specific assertions where useful.

Minimum required coverage:

- Notes payload creation from current selected note state
- Media payload creation from current hydrated selected media state
- Search payload creation from both RAG and Web Search result cards
- app-owned handoff always creating a new Chat session
- `ChatScreen` consuming pending handoff state on navigation
- visible handoff card rendering in the new session
- draft prompt prefilled and not auto-sent
- disabled or warning path when no valid Notes or Media selection exists
- failure path when new Chat session creation fails

Recommended test files:

- create `Tests/UI/test_chat_first_handoffs.py`
- extend existing focused tests in:
  - `Tests/UI/test_notes_screen.py`
  - `Tests/UI/test_chat_window_enhanced.py`
  - `Tests/UI/test_chat_screen_state.py`

## File Impact

Recommended implementation touch points for this slice:

- `tldw_chatbook/app.py`
  add app-owned pending handoff state and `open_chat_with_handoff()`
- `tldw_chatbook/UI/Screens/chat_screen.py`
  consume pending handoff, create new session, and apply handoff state
- `tldw_chatbook/UI/Screens/chat_screen_state.py`
  persist session-scoped handoff payload
- `tldw_chatbook/UI/Screens/notes_screen.py`
  expose current-selection `Use in Chat` and payload builder
- `tldw_chatbook/UI/MediaWindow_v2.py`
  expose current-selection `Use in Chat` and payload builder
- `tldw_chatbook/UI/Views/RAGSearch/search_result.py`
  add per-result `Use in Chat`
- `tldw_chatbook/UI/Views/RAGSearch/search_rag_window.py`
  support forwarding per-result handoff actions if needed
- `Tests/UI/test_chat_first_handoffs.py`
  new focused handoff coverage

Optional but recommended:

- `tldw_chatbook/Widgets/Chat_Widgets/chat_handoff_card.py`
  dedicated widget for consistent handoff presentation

## Recommendation

Implement `Use in Chat` as one app-owned pending-context seam with thin per-surface payload adapters.

This is the smallest design that still gives a coherent UX:

- always new session
- visible handoff confirmation
- draft prompt only
- consistent behavior across Notes, Media, RAG Search, and Web Search
- no duplicated launch logic spread across multiple destination surfaces
