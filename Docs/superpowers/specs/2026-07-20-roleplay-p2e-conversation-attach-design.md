# Roleplay P2e — conversation-attach for Lore (Roleplay Attachments tab)

**Status:** Design.

**Program:** Roleplay (Personas) redesign — P2 (Lore mode), fifth sub-project. Mirrors the merged Dictionaries conversation-attach (P1e). Follows P2a/P2c/P2d-1/P2d-2/P2d-regex (all merged); schema v22.

## Why

A user can create, edit, import/export, and regex-match world books in the Roleplay Lore mode, but there is no Roleplay-mode way to **attach a world book to a conversation** so its entries participate in that conversation's world-info injection. The backend and the live send path already exist; only the Roleplay UI is missing.

## What already exists (verified at dev tip)

- **Junction + backend:** `conversation_world_books(conversation_id TEXT, world_book_id INTEGER, priority INTEGER DEFAULT 0, PK(conversation_id, world_book_id))`. `WorldBookManager.associate_world_book_with_conversation(conversation_id, world_book_id, priority=0)` (an `INSERT OR REPLACE` **upsert** — always returns True, never raises `ConflictError`), `disassociate_world_book_from_conversation(conversation_id, world_book_id)` (DELETE, returns rowcount>0), and `get_world_books_for_conversation(conversation_id, enabled_only=True)` (conversation→books, used on send).
- **Live send application (legacy):** `chat_events.py` already calls `get_world_books_for_conversation(active_conversation_id)` and passes the result to `WorldInfoProcessor`, so an attached book takes effect immediately on the legacy send path.
- **Legacy attach UI:** a "World Books" collapsible in the legacy Chat right sidebar (`chat_events_worldbooks.py`). This is the UI the redesign supersedes; its retirement is P2g.

## Scope

**In scope (mirrors dict P1e):** a Lore **Attachments** tab in `PersonasLoreDetailWidget`; a generic conversation picker; `personas_screen` attach/detach/refresh handlers wired to the existing manager methods; and one small backend method — the reverse query `get_conversations_for_world_book`.

**Explicitly deferred → P2g:** the Console inspector "what's in play" world-book block and native-Console send-path application (Console never applies attached world books yet — only the legacy path does). Retiring the legacy Chat-sidebar attach UI → P2g. No migration (the junction exists).

## Behavior-change framing

Purely additive UI + one read-only backend query. No schema change, no send-path change (legacy already applies attached books), no change to existing lore flows. Only new: a Roleplay surface to attach/detach.

## Ground truths (verified post-#707 reformat, v22)

- `conversation_world_books.conversation_id` is `TEXT`; `conversations.id` is a UUID `TEXT` and `conversations.title` is nullable. `associate_world_book_with_conversation`'s `conversation_id: int` annotation is **misleading** — the runtime value is a string UUID (the P1e string-conv-id / int-book-id lesson). Pass conversation ids as **strings** everywhere.
- **Dictionaries precedent to mirror:** `personas_dictionary_detail.py` has an Attachments `TabPane` (empty-state `Static`, a `#personas-dict-attachments-table` DataTable of conversation/id, `Attach to conversation…` + `Detach` buttons; an I/O-free `load_attachments(rows)`); message classes `DictionaryAttachRequested` / `DictionaryDetachRequested(conversation_id)`; `personas_screen` handlers `_refresh_dictionary_attachments` (on selection), `@on(DictionaryAttachRequested)` (lists conversations, shows a picker, attaches), `@on(DictionaryDetachRequested)`; the picker `dictionary_attach_picker.py::DictionaryAttachPicker(ModalScreen[str|None])` (takes `[{conversation_id, title}]`, returns the picked string id — fully generic apart from its name/CSS).
- **Lore detail today:** `PersonasLoreDetailWidget` has exactly two tabs — `Entries` (`#personas-lore-tab-entries`) and `Settings` (`#personas-lore-tab-settings`); no Attachments tab, no attach messages. Lore CRUD in `personas_screen` uses `self._lore_manager()` (a `WorldBookManager`) directly, off-thread via `asyncio.to_thread` — NOT a scope service (unlike dicts).
- Attachable-conversation source: `ChaChaNotes_DB.search_conversations_page` (used by the dict `_list_attachable_conversations`).

## Architecture

### 1. Backend — reverse query (`world_book_manager.py`)

Add `get_conversations_for_world_book(world_book_id: int) -> List[Dict[str, Any]]`:
```
SELECT cwb.conversation_id, c.title
FROM conversation_world_books cwb
JOIN conversations c ON c.id = cwb.conversation_id
WHERE cwb.world_book_id = ? AND c.deleted = 0
ORDER BY c.last_modified DESC
```
Return `[{"conversation_id": str(row[0]), "title": row[1] or "(untitled)"}]` — conversation_id coerced to `str`, NULL title → `"(untitled)"` (matching the existing `_list_attachable_conversations` convention). (Mirrors the dict `list_dictionary_conversations`; no migration.) Verified: `conversations` has `deleted`, `last_modified`, and `title` columns.

### 2. UI — Lore Attachments tab (`personas_lore_detail.py`)

Add a third `TabPane("Attachments", id="personas-lore-tab-attachments")` to `#personas-lore-tabs`, mirroring the dict Attachments tab:
- An empty-state `Static(id="personas-lore-attachments-empty")` ("Not attached to any conversation yet.").
- A `DataTable(id="personas-lore-attachments-table")` with columns `conversation`, `id` (registered in `on_mount`).
- Buttons `Attach to conversation…` (`#personas-lore-attach-add`) and `Detach` (`#personas-lore-attach-detach`).
- **I/O-free:** `load_attachments(rows: list[dict])` renders `{conversation_id, title}` rows (title in the `conversation` cell, id in the `id` cell); shows the empty-state Static when no rows, the table otherwise.
- New message classes: `LoreAttachRequested` (bare intent) and `LoreDetachRequested(conversation_id: str)`. The `#personas-lore-attach-add` handler posts `LoreAttachRequested()`; the `#personas-lore-attach-detach` handler reads the selected attachment row's conversation id (a `_selected_attachment_id` helper mirroring the dict) and posts `LoreDetachRequested(conversation_id)`. Add both to `__all__`.

### 3. Conversation picker — new generic `ConversationAttachPicker` (`Widgets/Persona_Widgets/conversation_attach_picker.py`)

A `ConversationAttachPicker(ModalScreen[str | None])` — a search-filterable conversation `ListView` that takes `conversations: list[{conversation_id: str, title: str}]` and returns the picked **string** conversation id (or `None` on cancel). Generic name + generic ids (no dict/lore-specific content), so it's reusable. The merged `DictionaryAttachPicker` is left untouched (dict may migrate to this later — out of scope).

### 4. Screen handlers (`personas_screen.py`)

Mirror the dict handlers, using `self._lore_manager()` (WorldBookManager) via `asyncio.to_thread`, guarded (never crash → `self._notify`):
- `_refresh_lore_attachments()` — when a lore book is selected, `get_conversations_for_world_book(book_id)` off-thread → `detail.load_attachments(rows)`. Called from `_select_lore_entry` (the lore-selection hook, mirroring the dict's `_refresh_dictionary_attachments()` call in the dictionary-selection path).
- `@on(LoreAttachRequested) _handle_lore_attach` — guard a selected lore book + `_io_dialog_active`; worker: list conversations by **reusing the existing `_list_attachable_conversations`** (it is generic — returns `[{conversation_id: str, title: str}]` from `search_conversations_page`, not dict-specific), show `ConversationAttachPicker` via `push_screen_wait` → if a conversation is picked, `associate_world_book_with_conversation(str(picked), book_id)` off-thread (no ConflictError — it's an upsert), then `_refresh_lore_attachments` + notify. Re-entrancy guarded (`group="personas-io"`).
- `@on(LoreDetachRequested) _handle_lore_detach` — `disassociate_world_book_from_conversation(str(message.conversation_id), book_id)` off-thread → `_refresh_lore_attachments` + notify.

### 5. Send path — no change

`chat_events.py` already injects conversation-attached world books into `WorldInfoProcessor` on the legacy send, so an attached book takes effect immediately. Console "what's in play" + native-Console send application are P2g.

## Error handling

- Attach is idempotent (upsert) — no conflict path. Detach on a non-attached pair is a harmless no-op (rowcount 0).
- All DB I/O off-thread, wrapped → `_notify` on failure; the widget is I/O-free. Re-entrancy guarded via `_io_dialog_active` / `group="personas-io"`. A missing/unsaved selected book → the handlers no-op gracefully.

## Testing

- **Backend:** `get_conversations_for_world_book` round-trip — associate a book to two conversations, assert both returned with titles (and a NULL-title conversation → "Untitled"); a book attached to none → `[]`.
- **Widget (I/O-free):** `load_attachments([])` shows the empty-state and hides the table; `load_attachments([{conversation_id, title}])` renders the row; the Attach button posts `LoreAttachRequested`; selecting a row + Detach posts `LoreDetachRequested` with the row's conversation id.
- **Screen (real-DB, `LorePersonasTestApp` + `lore_db`):** posting `LoreAttachRequested` with the picker monkeypatched to return a conversation id persists the association (junction row present) and the Attachments table shows it; `LoreDetachRequested` removes it; `_refresh_lore_attachments` reflects the current state.
- **Integration:** an attached book appears in `get_world_books_for_conversation(conv_id)` — proving the (already-live) send path would inject it.
- **Picker:** `ConversationAttachPicker` filters by search and returns the picked string id / `None` on cancel.
- Full gate: `test_world_book_manager`, `test_personas_lore`, plus `import tldw_chatbook.app`.

## Decomposition

Single small plan (mirrors dict P1e). No migration. Console side (what's-in-play + native-send) and legacy-UI retirement → P2g.

## Acceptance criteria

- [ ] `get_conversations_for_world_book(id)` returns each attached conversation's `{conversation_id (str), title (NULL→"Untitled")}`; `[]` when none.
- [ ] The Lore detail widget has an Attachments tab (empty-state + table + Attach/Detach) that is I/O-free and posts `LoreAttachRequested`/`LoreDetachRequested`.
- [ ] A generic `ConversationAttachPicker` returns the picked string conversation id; the merged dict picker is untouched.
- [ ] Attaching the selected book to a picked conversation persists to `conversation_world_books` (idempotent upsert, no ConflictError); detaching removes it; the Attachments table refreshes to reflect state.
- [ ] Conversation ids are handled as strings throughout; no crash on missing selection or empty conversation list.
- [ ] No schema change; no send-path change (legacy already applies); Console what's-in-play + native-send deferred to P2g.
- [ ] Full gate green; `import tldw_chatbook.app` OK.
