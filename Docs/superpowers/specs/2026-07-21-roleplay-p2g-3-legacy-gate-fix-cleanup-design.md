# Roleplay P2g-3 — legacy send character-gate fix + dead world-book UI deletion

**Status:** Design (revised after spec review).

**Program:** Roleplay (Personas) redesign — P2 (Lore mode), **P2g cycle 3 of 3 — the final cycle**. Completes P2g and the entire P2 Lore program (P2a → P2g). Follows P2g-1 (native-Console world-info send) + P2g-2 (Console inspector), both merged. Schema **v22**.

## Why

Two loose ends on the legacy Chat path:
1. **Character-gate bug:** in `chat_events.py` the world-info block (build + consume) is nested inside `if active_char_data:`, so **conversation-attached world books never apply on the legacy send unless a character is loaded** — even though they don't need one.
2. **Dead world-book UI:** a "World Books" attach/detach sidebar section + its handlers exist but are unreachable — the section is composed only by `create_chat_right_sidebar` in the production-dead `ChatWindow`, and the handlers resolve the conversation via a nonexistent `app.active_conversation_id` (always `None`), so they could never attach anything.

## What already exists (verified at dev tip)

- **Shared helper (P2g-1):** `apply_world_info_to_message(db, conversation_id, char_data, message_text, history) -> str` (`world_info_resolver.py`) — never-raises, replicates the legacy build→process→format→join byte-for-byte.
- **Legacy inline world-info (to replace):** in `chat_events.py`, the build (~868-925: conversation books ∪ `resolve_character_world_books` ∪ `has_character_book` → `WorldInfoProcessor`) is inside `if active_char_data:` (line 846); the consume/join (~1450-1503: `process_messages` → `format_injections` → the `at_start→before_char→msg→after_char→at_end` join) is outside the gate but no-ops when `world_info_processor` is `None` (which it is without a character). `message_text_with_handoff` (~1446), `active_conversation_id` (843), `db` (844), `active_char_data`, `chat_history_for_api` are all in scope at the consume site.
- **User-visible world-info indicator (must preserve):** the legacy sets `app.current_world_info_active` (1495/1499) AND `app.current_world_info_count = len(matched_entries)` (1496/1500). Both are **read** at 1250-1251 to mount a `[World Info: N entries activated]` `ChatMessage` into the chat. `world_info_processor` is used ONLY in the build (918-927) + the consume (1452-1500) — nowhere else — so replacing both is safe.
- **Dead world-book UI graph (verified):**
  - `chat_events_worldbooks.py` — the sidebar handlers. Referenced by: `chat_events.py` import (38) + `refresh_active_worldbooks(app)` at 3156/3235/4327 (all no-op via the `active_conversation_id` attribute bug) + `**CHAT_WORLDBOOK_BUTTON_HANDLERS` merge (6707); and `app.py` import (242) + the `chat-worldbook-*` routing handlers (9130/9220/9294).
  - The world-book UI itself is the collapsible **section** at `chat_right_sidebar.py:551-627` (creates `#chat-worldbook-*` ids).
  - CSS: `css/layout/_sidebars.tcss:370-396` (`.worldbook-association-controls`, `.worldbook-priority-select`, `#chat-worldbook-available-listview`, `#chat-worldbook-active-listview`, `#chat-worldbook-details-display`).
- **Out of scope (verified, deferred):**
  - The whole-file removal of `Chat_Window.py` (production-dead, but instantiated by 5 test files) and `chat_right_sidebar.py` (the only creator of `#chat-right-sidebar`, which is queried by ~5 live-ish sites — `app.py:8381`, `chat_events_sidebar_resize.py`, `chat_events.py:4372/5020`; `ChatWindowEnhanced` "removed the right sidebar"). Deleting these pulls in a broader non-world-book chat-chrome audit → **filed as a backlog task**, not P2g-3.
  - A SEPARATE live CCP world-book surface (`ccp_handlers.populate_ccp_worldbook_list` / `ccp-worldbook-*`).

## Design decisions (locked with user)

1. **Route the legacy path through a count-returning resolver.** Factor a `resolve_world_info_injection(db, conversation_id, char_data, message_text, history) -> tuple[str, int]` (the injected text + matched-entry count) in `world_info_resolver.py`; make the existing `apply_world_info_to_message` a thin wrapper returning `[0]` (so native Console's string call is unchanged). The legacy call site uses the tuple version to preserve BOTH `current_world_info_active` and `current_world_info_count`.
2. **World-book-UI-only deletion** (revised after review): delete the world-book handlers + the world-book sidebar section + CSS + the live-file references. The broader dead-code cleanup (`Chat_Window.py`, all of `chat_right_sidebar.py`, the `#chat-right-sidebar` query web, the 5 `ChatWindow` tests) is **deferred to a filed backlog task** — it touches non-world-book chat chrome and warrants its own audit.

## Behavior-change framing

- **Part A** changes the legacy send in exactly one way: conversation-attached world books now apply **without** a loaded character. The with-character case is **byte-identical** (the resolver mirrors the inline pipeline). `enable_world_info`, the `[World Info: N entries]` indicator, and `current_world_info_active`/`count` are preserved.
- **Part B** removes only unreachable world-book code/CSS + its references — **no runtime behavior change** (the removed `refresh_active_worldbooks` calls already no-op; the sidebar section is only composed by the production-dead `ChatWindow`).

## Architecture

### Part A — gate fix + count-returning resolver
1. In `world_info_resolver.py`, factor the current `apply_world_info_to_message` body into `resolve_world_info_injection(...) -> tuple[str, int]`: same collect→build→`process_messages`→`format_injections`→join, but also return `len(result["matched_entries"])` (0 when no match / no books / error / non-string). `apply_world_info_to_message(...)` becomes `return resolve_world_info_injection(...)[0]` (native Console + existing tests unchanged; add a test that the wrapper returns the text and the count matches).
2. In `chat_events.py`, delete the inline build block (inside the gate) and the inline consume/join block (+ the now-dead `world_info_processor` local), and replace the consume with, **outside** the `if active_char_data:` gate:
```
message_text_with_world_info = message_text_with_handoff
if get_cli_setting("character_chat", "enable_world_info", True):
    message_text_with_world_info, _wi_count = resolve_world_info_injection(
        db, active_conversation_id, active_char_data,
        message_text_with_handoff, chat_history_for_api,
    )
    app.current_world_info_active = message_text_with_world_info != message_text_with_handoff
    app.current_world_info_count = _wi_count
```
This preserves the `enable_world_info` gate, sets both indicator reactives (so the `[World Info: N entries]` message still mounts), and removes the character gate.

### Part B — world-book UI deletion (scoped)
- **Delete file:** `tldw_chatbook/Event_Handlers/Chat_Events/chat_events_worldbooks.py`.
- **`chat_events.py`:** remove the `chat_events_worldbooks` import (38), the 3 `refresh_active_worldbooks(app)` calls (3156/3235/4327), and the `**chat_events_worldbooks.CHAT_WORLDBOOK_BUTTON_HANDLERS` merge (6707).
- **`app.py`:** remove the `chat_events_worldbooks` import (242) and the three `chat-worldbook-*` routing branches (the `on_input_changed`/`on_list_view_selected`/`on_checkbox_changed` blocks that call `chat_events_worldbooks.*`). Leave the `#chat-right-sidebar` query sites alone (broader cleanup → backlog).
- **`chat_right_sidebar.py`:** delete only the world-book collapsible **section** (`:551-627`); keep the rest of the file (its whole-file removal is the deferred backlog task).
- **CSS:** remove the five `worldbook`/`chat-worldbook-*` rules in `_sidebars.tcss:370-396`.
- **Verification (mandatory):** `grep -rn "chat_events_worldbooks\|handle_worldbook\|CHAT_WORLDBOOK_BUTTON_HANDLERS\|chat-worldbook"` across `tldw_chatbook/` + `Tests/` returns **no live references** (only inert `.backup` files remain; the separate `ccp-worldbook` surface is untouched); `import tldw_chatbook.app` OK.

## Error handling
Part A relies on the never-raising resolver. Part B is scoped deletion — the only risk is a dangling world-book reference, caught by the grep sweep + app import.

## Testing
- **Part A (real, load-bearing):** with a conversation that has an attached world book and **`char_data=None`** (no character), `resolve_world_info_injection` returns `(injected_text, count>=1)` — proving the gate no longer blocks conversation books; with a character, the injected text matches the pre-change join (byte-parity) and the count equals the matched entries; no match / no conversation / `enable_world_info` off → `(unchanged_text, 0)`. `apply_world_info_to_message` still returns just the text (native unchanged).
- **Part B:** `import tldw_chatbook.app` OK; the grep sweep shows no live world-book references; the broader chat/console suite still passes (nothing else referenced the deleted symbols).
- Full gate: the world-info resolver tests + a Part-A behavior test + `import tldw_chatbook.app`.

## Decomposition
Two tasks: (A) the count-returning resolver + the `chat_events.py` gate fix + tests; (B) the scoped world-book UI deletion + verification. Plus a **filed backlog task** for the broader `Chat_Window.py`/`chat_right_sidebar.py`/`#chat-right-sidebar`/5-tests dead-code cleanup. This completes P2g and the P2 Lore program.

## Acceptance criteria
- [ ] `resolve_world_info_injection(...) -> (text, count)` is factored from `apply_world_info_to_message` (which becomes a thin wrapper returning the text; native Console unchanged); never raises.
- [ ] The legacy `chat_events.py` send applies conversation-attached world books **without a loaded character** (gate removed), via `resolve_world_info_injection` placed outside `if active_char_data:`; with-character byte-identical; `enable_world_info` + `current_world_info_active` + `current_world_info_count` (the `[World Info: N entries]` indicator) preserved; the inline build/consume/`world_info_processor` local removed.
- [ ] `chat_events_worldbooks.py` is deleted; the world-book section in `chat_right_sidebar.py` (551-627) + the CSS rules + the references in `chat_events.py` (import + 3 refresh calls + merge) and `app.py` (import + 3 routing branches) are removed.
- [ ] A grep sweep confirms no live world-book references (only inert `.backup` + the separate `ccp-worldbook` surface remain); `import tldw_chatbook.app` OK; broader suite passes.
- [ ] A backlog task is filed for the broader `Chat_Window.py`/`chat_right_sidebar.py`/`#chat-right-sidebar`/5-tests cleanup. No schema change. **P2 Lore program complete.**
