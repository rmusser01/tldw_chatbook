# Roleplay P2g-3 — legacy send character-gate fix + dead world-book UI deletion

**Status:** Design.

**Program:** Roleplay (Personas) redesign — P2 (Lore mode), **P2g cycle 3 of 3 — the final cycle**. Completes P2g and the entire P2 Lore program (P2a → P2g). Follows P2g-1 (native-Console world-info send) + P2g-2 (Console inspector), both merged. Schema **v22**.

## Why

Two loose ends on the legacy Chat path:
1. **Character-gate bug:** in `chat_events.py` the world-info block (build + consume) is nested inside `if active_char_data:`, so **conversation-attached world books never apply on the legacy send unless a character is loaded** — even though they don't need one. A user who attaches a world book to a conversation (via the Roleplay Lore tab) sees it apply on native Console (P2g-1) but silently do nothing in the legacy Chat window without a character.
2. **Dead world-book UI:** a "World Books" attach/detach sidebar exists but is unreachable — it's composed only by `create_chat_right_sidebar` in `Chat_Window.py`'s `ChatWindow`, which is **never instantiated in production** (the app uses `ChatWindowEnhanced`), and its handlers resolve the conversation via a nonexistent `app.active_conversation_id` (always `None`), so it could never attach anything even if reached.

## What already exists (verified at dev tip)

- **Shared helper (P2g-1):** `apply_world_info_to_message(db, conversation_id, char_data, message_text, history) -> str` (`world_info_resolver.py`) — never-raises, replicates the legacy build→process→format→join **byte-for-byte** (it was built for exactly this reuse).
- **Legacy inline world-info (to replace):** in `chat_events.py`, the build (~868-925, conversation books ∪ `resolve_character_world_books` ∪ `has_character_book` → `WorldInfoProcessor`) is inside `if active_char_data:` (line 846); the consume/join (~1445-1503: `process_messages` → `format_injections` → `at_start→before_char→msg→after_char→at_end` join, and `app.current_world_info_active = True`) is outside the gate but no-ops when `world_info_processor` is `None` (which it is without a character). `message_text_with_handoff` (~1446), `active_conversation_id` (843), `db` (844), `active_char_data` are all in scope at the consume site.
- **Dead-code graph (verified):**
  - `chat_events_worldbooks.py` — the sidebar handlers. Referenced by: `chat_events.py` import (38) + `refresh_active_worldbooks(app)` at 3156/3235/4327 + `**CHAT_WORLDBOOK_BUTTON_HANDLERS` merge (6707); and `app.py` import (242) + routing handlers (9130/9220/9294) in the `chat-worldbook-*` `on_input_changed`/`on_list_view_selected`/`on_checkbox_changed` branches. The `refresh_active_worldbooks` calls no-op (the `active_conversation_id` attribute bug).
  - `Chat_Window.py` (`ChatWindow`) — never instantiated in production; instantiated by 5 test files (`test_chat_window_tooltips.py`, `test_chat_window_tooltips_fixed.py`, `test_send_stop_button.py`, `test_ui_example_best_practices.py`, `test_chat_image_integration_real.py`) that exercise general chat-window behavior against the dead class.
  - `chat_right_sidebar.py` — exports only `create_chat_right_sidebar`; its only live importer is `Chat_Window.py` (the `.backup` files are inert non-`.py` files).
  - CSS: `css/layout/_sidebars.tcss:370-396` (`.worldbook-association-controls`, `.worldbook-priority-select`, `#chat-worldbook-available-listview`, `#chat-worldbook-active-listview`, `#chat-worldbook-details-display`).
- **Out of scope:** a SEPARATE, apparently-live CCP world-book surface (`ccp_handlers.populate_ccp_worldbook_list` / `ccp-worldbook-*` in `app.py`) — untouched by P2g-3.

## Design decisions (locked with user)

1. **Route the legacy path through the shared helper** (not an inline ungate) — DRY, byte-parity for the with-character case, single source of truth.
2. **Full removal of the dead world-book UI including `Chat_Window.py`/`chat_right_sidebar.py` and their 5 tests** (the tests exercise a production-dead class; their coverage is illusory).

## Behavior-change framing

- **Part A** changes the legacy send in exactly one way: conversation-attached world books now apply **without** a loaded character. The with-character case is **byte-identical** (the helper mirrors the inline pipeline). `enable_world_info` and the `app.current_world_info_active` indicator are preserved.
- **Part B** removes only unreachable code + tests of a production-dead class — **no runtime behavior change** (the removed `refresh_active_worldbooks` calls already no-op).

## Architecture

### Part A — gate fix (`chat_events.py`)
Delete the inline world-info **build** block (inside `if active_char_data:`) and the inline **consume/join** block, and replace the consume with a single call, placed **outside** the `if active_char_data:` gate, where `message_text_with_handoff` is computed:
```
message_text_with_world_info = message_text_with_handoff
if get_cli_setting("character_chat", "enable_world_info", True):
    injected = apply_world_info_to_message(
        db, active_conversation_id, active_char_data,
        message_text_with_handoff, chat_history_for_api,
    )
    message_text_with_world_info = injected
    app.current_world_info_active = injected != message_text_with_handoff
```
- Preserves the `enable_world_info` config gate (the helper doesn't check it — the call site does, mirroring P2g-1's native applier).
- Preserves the `app.current_world_info_active` indicator (set from whether the helper changed the text).
- Removes the now-dead `world_info_processor` local + its gated init. The consume site keeps feeding `message_text_with_world_info` to the API dispatch exactly as before.

### Part B — dead-code deletion
- **Delete files:** `tldw_chatbook/Event_Handlers/Chat_Events/chat_events_worldbooks.py`; `tldw_chatbook/UI/Chat_Window.py`; `tldw_chatbook/Widgets/Chat_Widgets/chat_right_sidebar.py`; the 5 test files above.
- **`chat_events.py`:** remove the `chat_events_worldbooks` import (38), the 3 `refresh_active_worldbooks(app)` calls (3156/3235/4327), and the `**chat_events_worldbooks.CHAT_WORLDBOOK_BUTTON_HANDLERS` merge (6707).
- **`app.py`:** remove the `chat_events_worldbooks` import (242) and the three `chat-worldbook-*` routing branches (the `on_input_changed`/`on_list_view_selected`/`on_checkbox_changed` handlers that call `chat_events_worldbooks.*`). Check the `# ID from create_chat_right_sidebar` comment (8369) — remove/retarget only if it references a now-deleted id (it appears to be a comment; verify it doesn't break).
- **CSS:** remove the five `worldbook`/`chat-worldbook-*` rules in `_sidebars.tcss:370-396`.
- **Verification (mandatory):** after removal, `grep -rn "chat_events_worldbooks\|handle_worldbook\|CHAT_WORLDBOOK_BUTTON_HANDLERS\|create_chat_right_sidebar\|from.*Chat_Window import\|ChatWindow\b\|chat-worldbook"` across `tldw_chatbook/` + `Tests/` returns **no live references** (only inert `.backup` files and the separate `ccp-worldbook`/`ChatWindowEnhanced` matches remain); `import tldw_chatbook.app` OK; the CSS build (if any) still succeeds.

## Error handling
Part A relies on the never-raising helper. Part B is pure deletion — the only risk is a dangling reference, caught by the grep sweep + app import.

## Testing
- **Part A (real-DB, the load-bearing behavior test):** the fix's whole point — with a conversation that has an attached world book and **`active_char_data=None`** (no character), the outgoing message gets the world-info injected (proving the gate no longer blocks it). Test at the helper/integration level the way P2g-1's send-path test does (build the same inputs `chat_events` now passes). Plus: with a character present, the injected output matches the pre-change behavior (byte-parity — assert the join order/content). `enable_world_info=False` → no injection.
- **Part B:** `import tldw_chatbook.app` OK; the grep sweep shows no live dangling references; the broader chat/console test suite still passes (the 5 deleted tests are gone; nothing else referenced the deleted symbols).
- Full gate: the world-info resolver tests + a chat_events-level Part-A test + `import tldw_chatbook.app`.

## Decomposition
Two tasks: (A) the gate fix (route through the helper) + its behavior test; (B) the coordinated dead-code deletion + verification. This completes P2g and the P2 Lore program.

## Acceptance criteria
- [ ] The legacy `chat_events.py` send applies conversation-attached world books **without a loaded character** (gate removed), via `apply_world_info_to_message` placed outside `if active_char_data:`; the with-character case is byte-identical; `enable_world_info` + `app.current_world_info_active` preserved.
- [ ] The inline world-info build/consume/`world_info_processor` local is removed (single source of truth = the shared helper).
- [ ] `chat_events_worldbooks.py`, `Chat_Window.py`, `chat_right_sidebar.py`, and the 5 `ChatWindow` test files are deleted; their references in `chat_events.py` (import + 3 refresh calls + handler-merge) and `app.py` (import + 3 routing branches) and the CSS are removed.
- [ ] A grep sweep confirms no live dangling references to any deleted symbol (only inert `.backup` files + the separate `ccp-worldbook` surface remain); `import tldw_chatbook.app` OK; the broader test suite passes.
- [ ] No schema change; the separate CCP world-book surface is untouched. **P2 Lore program complete.**
