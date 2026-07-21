# Roleplay P2g-2 — Console "what's in play" world-book inspector + attach/detach

**Status:** Design.

**Program:** Roleplay (Personas) redesign — P2 (Lore mode), **P2g cycle 2 of 3**. Mirrors the merged Dictionary Console inspector (P1g). P2g-1 (native-Console world-info send) merged; P2g-3 = legacy send gate-bug fix + dead-code cleanup. Schema **v22**.

## Why

The native Console applies conversation-attached world books on send (P2g-1), but the Console run inspector shows nothing about them — a user can't see which world books are in play, and (unlike dictionaries, which have a Console "Chat Dictionaries" block with Attach/Detach) can't manage them without leaving the Console. This adds the world-book equivalent: a "World Books" inspector block that shows what's active and lets the user attach/detach a conversation world book in place.

## What already exists (verified at dev tip)

- **The dict "Chat Dictionaries" inspector block (mirror target):** a **trailing custom block** on `ConsoleRunInspector` — NOT routed through `_ROW_GROUPS`/`_ROW_ID_BY_LABEL` (those drive the main grouped rows). It is threaded through THREE spots: `compose()` (`console_run_inspector.py:415-422`, reads `state.dictionary_rows`/`dictionary_actions`), `_rendered_row_entries()` (`:227`, for in-place row updates), and `_structural_key()` (`:263`, for structural-recompose detection). Missing any of the three breaks updates.
- **State carrier:** `ConsoleInspectorState` (`console_display_state.py`) has extra `dictionary_rows: tuple[ConsoleDisplayRow, ...]` / `dictionary_actions: tuple[ConsoleInspectorAction, ...]` fields. `ConsoleInspectorAction(widget_id, label, enabled, disabled_reason="", classes=...)` — a button spec; `ConsoleDisplayRow(text, status?)`.
- **Row/action projection:** `ChatScreen._console_dictionary_inspector_rows()` and `_console_dictionary_inspector_actions()` (`chat_screen.py:5847`, `:5930`) read ONLY the cache + the active conversation id (never the DB), and are wired into `_build_console_inspector_state()` (`:5769`).
- **Zero-DB-on-recompose:** a cached `self._active_dictionaries_summary` (`:1622`); `refresh_active_dictionaries_summary()` (`:5795`, the ONLY DB summarize call, off-thread via `asyncio.to_thread` + a private `asyncio.run` loop, `:535`); `_refresh_active_dictionaries_summary_if_scope_changed()` (`:5829`, compares `_active_console_dictionary_scope_ids()` vs `_last_console_dictionary_scope_ids`), invoked from `_sync_native_console_chat_ui()` (also on the 0.2s transcript poll while streaming).
- **Action click routing:** `@on(Button.Pressed, "#console-inspector-dictionaries-attach")` / `"#…-detach"` handlers (`chat_screen.py:829`, `:847`) guard a `_console_dictionary_dialog_active` flag and `run_worker(worker, group="console-io")`. The **detach uses a picker** too (pick which attached dictionary to remove).
- **Conversation id:** `ChatScreen._current_console_rail_conversation_id()` (`:3088`, sources from the native session's `persisted_conversation_id`).
- **Reusable pieces:** `summarize`-style resolver seam via P2g-1's `_collect_active_world_books(db, conversation_id, char_data)` (`world_info_resolver.py:17`); `WorldBookPicker(world_books, *, title, confirm_label)` returning an int id (P2f); `WorldBookManager.get_world_books_for_conversation` / `associate_world_book_with_conversation` / `disassociate_world_book_from_conversation` / `list_world_books` (P2a/P2e).

## Scope

**In:** a "World Books" inspector block (read) + Attach/Detach actions (Console-side, conversation world books). **Deferred → P2g-3:** the legacy `chat_events.py` character-gate bug fix + deleting the dead legacy world-book UI. No migration; no send-path change (P2g-1 already applies attached books on native send).

## Design decisions

1. **Full parity with the dict inspector** (read block + Attach/Detach actions) — the user chose Read + Attach/Detach.
2. **Trailing custom block, not routed rows.** The world-book block mirrors the dict block's trailing pattern (`world_book_rows`/`world_book_actions`), threaded through **all three** render spots. It needs **no** `_ROW_ID_BY_LABEL`/`_ROW_GROUPS` registration.
3. **Detach via picker** (mirror dict): a picker of the currently-attached books → `disassociate`. `WorldBookPicker` is reused for both Attach and Detach via its `title`/`confirm_label` params — no new picker.
4. **Conversation-only on native.** The summary uses `_collect_active_world_books(db, conversation_id, char_data)` with `char_data=None` on native Console (no character), so it shows conversation-attached books — matching the send path.
5. **No scope service.** Dictionaries route through `ChatDictionaryScopeService` (local/server split); world books have no such split — `summarize_active_world_books` is a plain function in `world_info_resolver.py`, called off-thread directly.

## Behavior-change framing

Additive: a new inspector block + a summarize function + a cached scope-guarded refresh + two action workers. The dictionary inspector block, its refresh, and its actions are untouched. With no conversation / no attached books, the world-book block is empty (renders nothing), so the inspector is visually unchanged for that case.

## Ground truths

- `_collect_active_world_books` returns `(world_books: list[dict], has_character_book: bool)`; each book dict has `name`, `entries` (list), `enabled`. On native (`char_data=None`) it returns conversation books only.
- `get_world_books_for_conversation` returns book dicts with `id`, `name`, `entries`, `enabled`; map to `{"world_book_id": id, "name": name}` for the picker.
- The inspector projection methods and the action-list method read ONLY the cache + the conversation id — never the DB (the DB read happens only in the scope-guarded off-thread refresh).

## Architecture

### 1. Summarize resolver (`world_info_resolver.py`)
`summarize_active_world_books(db, conversation_id: str | None, char_data: dict | None) -> dict` — reuse `_collect_active_world_books`, return `{"world_books": [{"name": str, "enabled": bool, "entry_count": int}], "source": "local"}` (names normalized to str, `entry_count = len(entries) if list else 0`). **No dedup-by-name** — conversation books are keyed by id (two distinct books may share a name), and the inspector rows are index-keyed (`…-row-{index}`), not name-keyed, so there is no `DuplicateKey` risk (unlike the P2f widget). Never raises (returns `{"world_books": [], "source": "local"}` on any error). Mirrors `summarize_active_dictionaries`.

### 2. Inspector state (`console_display_state.py`)
Add `world_book_rows: tuple[ConsoleDisplayRow, ...] = ()` and `world_book_actions: tuple[ConsoleInspectorAction, ...] = ()` to `ConsoleInspectorState`.

### 3. Zero-DB-on-recompose refresh (`chat_screen.py`)
- Cache `self._active_world_books_summary: dict | None = None`; `self._last_console_world_book_scope_ids`.
- `refresh_active_world_books_summary()` — the ONLY DB call: off-thread `asyncio.to_thread(summarize_active_world_books, db, conversation_id, char_data=None)`, stores the result in the cache.
- `_refresh_active_world_books_summary_if_scope_changed()` — compare `_active_console_world_book_scope_ids()` (`(conversation_id,)` on native) to `_last_…`; refresh only on change. Invoke it from `_sync_native_console_chat_ui()` alongside the dict guard.

### 4. Row/action projection (`chat_screen.py`)
- `_console_world_book_inspector_rows() -> tuple[ConsoleDisplayRow, ...]` — from the cache only: one row per book, `text = name`, value = entry count + a "(disabled)" suffix when not enabled (mirror `_console_dictionary_inspector_rows`).
- `_console_world_book_inspector_actions() -> tuple[ConsoleInspectorAction, ...]` — Attach (`#console-inspector-worldbooks-attach`, enabled when a conversation id is present, `disabled_reason="Start or load a conversation first"`) + Detach (`#console-inspector-worldbooks-detach`, enabled when ≥1 book is attached). Reads cache + conversation id only.
- Wire both into `_build_console_inspector_state()` (`world_book_rows=…, world_book_actions=…`).

### 5. Inspector block render (`console_run_inspector.py`)
Add a "World Books" heading block mirroring the dict block, threaded through **all three**: `compose()` (heading + `state.world_book_rows` + `state.world_book_actions` via `_compose_action`), `_rendered_row_entries()` (append the world-book rows), `_structural_key()` (include `world_book_rows`/`world_book_actions`). No `_ROW_ID_BY_LABEL` change.

### 6. Attach/Detach handlers + workers (`chat_screen.py`)
- `@on(Button.Pressed, "#console-inspector-worldbooks-attach")` / `"#…-detach"` — guard a new `_console_worldbook_dialog_active` flag; `run_worker(worker, group="console-io")`.
- **Attach worker:** `conv_id = _current_console_rail_conversation_id()`; off-thread list standalone books not already attached (`list_world_books` minus `get_world_books_for_conversation`); `WorldBookPicker(books, title="Attach world book", confirm_label="Attach")`; on pick, off-thread `associate_world_book_with_conversation(conv_id, picked)`; then `refresh_active_world_books_summary()` + rebuild the inspector + notify; `finally` reset the flag.
- **Detach worker:** attached books via `get_world_books_for_conversation(conv_id)` → `WorldBookPicker(attached, title="Detach world book", confirm_label="Detach")`; on pick, off-thread `disassociate_world_book_from_conversation(conv_id, picked)`; refresh + rebuild + notify.
- All DB I/O off-thread, wrapped → notify, never crash; re-entrancy guarded; the picker is shown via `push_screen_wait`.

## Error handling
- The summarize function never raises. The refresh swallows errors (cache stays as-is / empties). The action workers wrap DB/picker calls → notify on failure, `finally`-reset the dialog flag; a missing conversation id → no-op.

## Testing (real-integration, no fakes)
- **Summarize:** attach two books to a conversation → `summarize_active_world_books` lists both with entry counts; a disabled book reports its state; none → empty; malformed → empty, no raise.
- **Inspector render (widget):** a `ConsoleInspectorState` carrying `world_book_rows`/`world_book_actions` renders the "World Books" heading + rows + Attach/Detach buttons; empty tuples → nothing rendered; the dict block still renders (no regression).
- **Scope-guard:** `refresh_active_world_books_summary` runs the DB read only when the scope id changes (mirror the dict scope-guard test).
- **Console attach/detach (real-DB):** with a native session pinned to a conversation, the Attach action (picker monkeypatched to return a book id) persists the junction + the block shows it; Detach removes it; re-entrancy guard holds.
- **Full gate:** the new resolver/inspector/screen tests + the dict inspector tests (no regression) + `import tldw_chatbook.app`.

## Decomposition
P2g-2 of 3. No migration. P2g-3 = legacy `chat_events.py` character-gate fix + dead-code deletion.

## Acceptance criteria
- [ ] `summarize_active_world_books` returns the conversation's active world books (`name`/`enabled`/`entry_count`) reusing `_collect_active_world_books`; never raises; conversation-only when `char_data=None`.
- [ ] `ConsoleInspectorState` gains `world_book_rows`/`world_book_actions`; the "World Books" block renders via `compose()`/`_rendered_row_entries()`/`_structural_key()` (all three) with no `_ROW_ID_BY_LABEL` change; the dict block is unaffected.
- [ ] The world-book summary is cached and refreshed ONLY by a scope-guarded off-thread `refresh_active_world_books_summary()`; the projection/action methods never touch the DB.
- [ ] Attach/Detach actions (gated on conversation presence / attached-book presence) open a `WorldBookPicker` and `associate`/`disassociate` a conversation world book off-thread, guarded by `_console_worldbook_dialog_active`, then refresh; never crash.
- [ ] Full gate green; `import tldw_chatbook.app` OK. Legacy fix + cleanup → P2g-3.
