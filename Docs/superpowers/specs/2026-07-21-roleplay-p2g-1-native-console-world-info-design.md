# Roleplay P2g-1 — shared world-info resolver + native-Console world-info send

**Status:** Design.

**Program:** Roleplay (Personas) redesign — P2 (Lore mode), **P2g cycle 1 of 3**. P2g mirrors the merged Dictionary Console work (P1g inspector + native-send #664). This cycle = native-Console send application. P2g-2 = Console "what's in play" world-book inspector. P2g-3 = legacy send gate-bug fix + dead-code cleanup. Follows P2a/P2c/P2d-1/2/regex/P2e/P2f (all merged); schema **v22**.

## Why

Conversation-attached world books (P2e) take effect on the **legacy** Chat send path but **never on the native Console** — `console_chat_controller.py` has zero world-info code. The native Console is a live shipping surface, so a user who attaches a world book to a conversation (via the Roleplay Lore Attachments tab) sees it apply in the legacy Chat window but silently do nothing in the native Console. This cycle closes that gap, mirroring the merged native-Console dictionary send (`_apply_chat_dictionaries`, #664).

## What already exists (verified at dev tip)

- **Dictionary native-send precedent (mirror target):** `ConsoleChatController._apply_chat_dictionaries(provider_messages, session_id)` (`console_chat_controller.py:1645-1719`), called at the **4 real send sites** — `submit_draft` (:431), retry (:1066), `continue_from_message` (:1114), `regenerate_message` (:1170) — each right after `_apply_skill_substitution`. Offloaded via `asyncio.to_thread`; handles string + list/multimodal content per-part (:1683-1711); `CancelledError` re-raised; any other failure returns the payload unchanged; resolves `conversation_id` from `session.persisted_conversation_id`. The `build_context_snapshot` call (:1241) is preview-only (NOT a send site). Wired via `chat_dictionary_applier` on `ConsoleChatController.__init__`; the bound `_console_chat_dictionary_applier(conversation_id, text) -> str` (`chat_screen.py:5753-5775`) hard-codes `char_data=None` ("native sessions carry no character card").
- **World-info processing engine (unchanged):** `WorldInfoProcessor(character_data, world_books)` (`world_info_processor.py:64-97`); `process_messages(current_message: str, conversation_history: List[Dict[str,str]], scan_depth=None, apply_token_budget=True) -> {"injections": {pos:[str]}, "matched_entries":[...], "tokens_used": int}` (:262); `format_injections(injections) -> {pos: "joined\n\n"}` keyed `before_char/after_char/at_start/at_end` (:625).
- **The legacy join (the exact target to replicate)** — `chat_events.py:1470-1489`:
  ```
  parts = []
  if at_start:  parts.append(at_start)
  if before_char: parts.append(before_char)
  parts.append(message_text)
  if after_char: parts.append(after_char)
  if at_end:  parts.append(at_end)
  message_text_with_world_info = "\n\n".join(parts)
  ```
- **Book sources:** `WorldBookManager.get_world_books_for_conversation(conversation_id, enabled_only=True)` (conversation-attached, entries populated); `resolve_character_world_books(char_data, exclude_names)` (character-attached snapshots, P2f); the processor also reads a character's native `character_book` via `character_data`.
- **Native Console has no character** — `ConsoleSessionSettings.character_label` is a free-text display string, never a DB id (`chat_screen.py:5785-5791`).

## Scope

**In (this cycle):** a shared, testable send-path helper that builds the world-info-injected message text; a native applier `_apply_world_info` at the 4 send sites; the conversation-only wiring. **Purely additive to native Console** — a conversation with no attached books is byte-identical to today.

**Deferred:** the Console "what's in play" inspector block → **P2g-2**; the legacy `chat_events.py` gate-bug fix (routing it through the shared helper) + deleting the dead legacy world-book UI → **P2g-3**. This cycle does NOT touch `chat_events.py` or `world_info_processor.py`.

## Design decisions

1. **World-info runs AFTER dictionaries on native Console** (`_apply_world_info` immediately after `_apply_chat_dictionaries` at each send site). Rationale — NOT a legacy-parity match (the legacy `chat_events.py` path applies no dictionaries at all): world-info should fire on **what the LLM actually sees**, i.e. the dict-substituted text. The injected world-info block itself is authored content and is not dict-substituted.
2. **Conversation-only wiring.** The native bound applier passes `char_data=None`, so only conversation-attached books apply (native Console has no character). The shared helper itself is general (takes `char_data`) so P2g-3 can route the legacy path — which does have a character — through the same code.
3. **Shared helper replicates the legacy pipeline faithfully**, so P2g-3's "route legacy through the helper" is a pure refactor (byte-parity), not a behavior change.
4. **Multimodal-safe.** `process_messages` types content as `str`; the applier normalizes the current message + history to plain strings (text parts only) before scanning, and injects into the **text** of the final user message (string content → wrap it; list/multimodal content → wrap the text part(s), leave image parts intact), mirroring `_apply_chat_dictionaries`'s per-part handling.

## Behavior-change framing

Additive to the native Console send: a new applier at the 4 send sites + a new shared lib helper + one new `world_info_applier` constructor param. No change to the legacy path, `world_info_processor.py`, or the schema. No conversation attached ⇒ the native send is byte-identical to today.

## Ground truths

- `process_messages` returns `{}`-injections when `self.entries` is empty; `format_injections` returns only non-empty positions. The helper must no-op (return `message_text` unchanged) whenever nothing matches, there are no books, there is no conversation id, or the DB read fails — **never raise** (embedded/imported content is already hardened by P2f's `resolve_character_world_books`; the helper additionally wraps the processor build/scan in try/except).
- `get_world_books_for_conversation` type-hints `conversation_id: int` but the real value is a string UUID (SQLite loose typing) — pass strings, matching P2e.
- History content may be a multimodal list (`[{"type":"text","text":...}, {"type":"image_url",...}]`); `_build_scan_text` expects string content, so normalization is mandatory.

## Architecture

### 1. Shared send-path helper (new module `tldw_chatbook/Character_Chat/world_info_resolver.py`)
`apply_world_info_to_message(db, conversation_id: str | None, char_data: dict | None, message_text: str, history: list[dict]) -> str` — parallels the dict lib's `apply_active_chatdicts_to_text`:
- If no `conversation_id` and no `char_data` → return `message_text` unchanged.
- Collect books exactly as the legacy path does: `world_books = get_world_books_for_conversation(conversation_id, enabled_only=True)` (when a conversation id is present) unioned with `resolve_character_world_books(char_data, {names of those books})`; detect `has_character_book`.
- `processor = WorldInfoProcessor(character_data=char_data if has_character_book else None, world_books=world_books or None)`.
- `result = processor.process_messages(message_text, history)`; if no `matched_entries` → return `message_text` unchanged.
- `formatted = processor.format_injections(result["injections"])`; build the join (`at_start → before_char → message_text → after_char → at_end`, `"\n\n"`).
- Wrap the whole thing in try/except → on any error return `message_text` unchanged (log at debug).
- A small internal `_collect_active_world_books(db, conversation_id, char_data)` (the book-gathering half — conversation ∪ character, mirroring the dict `_resolve_active_dictionaries`) is factored so **P2g-2's summarize can reuse it** (the shared-core seam). The helper assumes string-content `history` (the native controller normalizes multimodal → text before calling; the legacy caller already passes string-content history).

### 2. Native applier (`console_chat_controller.py`)
`async def _apply_world_info(self, provider_messages, session_id) -> provider_messages` mirroring `_apply_chat_dictionaries`:
- Resolve `conversation_id = session.persisted_conversation_id`; if falsy → return `provider_messages` unchanged.
- Find the final `role=="user"` message. Build the **history** = the messages before it, and the **current text** = that message's text — both **normalized to strings** (extract `text` parts from multimodal content; skip image parts).
- `injected = await asyncio.to_thread(self._world_info_applier, conversation_id, current_text, history)`.
- Write `injected` back into the final user message: string content → replace; list/multimodal content → replace the text part(s), leave image parts.
- `CancelledError` re-raised; any other exception → return `provider_messages` unchanged.
- New constructor param `world_info_applier: Callable[[str | None, str, list], str] | None = None` (mirror `chat_dictionary_applier`).

### 3. Call sites + ordering
Call `_apply_world_info` at the same 4 send sites (submit / retry / continue / regenerate), immediately **after** `_apply_chat_dictionaries`. Leave `build_context_snapshot` (preview) untouched — matching how the dict applier treats it.

### 4. Wiring (`chat_screen.py`)
A bound `_console_world_info_applier(conversation_id: str | None, message_text: str, history: list) -> str` mirroring `_console_chat_dictionary_applier`, hard-coding `char_data=None` (conversation-only), delegating to `apply_world_info_to_message(self.app.chachanotes_db, conversation_id, None, message_text, history)`. Passed to `ConsoleChatController(..., world_info_applier=self._console_world_info_applier)` at construction (`chat_screen.py:2749` area).

## Error handling
- The helper never raises: no conversation/books/match → unchanged text; DB or processor error → unchanged text (debug log). The applier never raises except re-raising `CancelledError`.
- Malformed embedded character content is already sanitized by `resolve_character_world_books` (P2f); with `char_data=None` on native it isn't even reached.

## Testing (real-integration, no fakes)
- **Helper:** a conversation with an attached book whose entry keys match the message → the returned text contains the injected content in the correct join order; no attached book / no match / no conversation id → text unchanged; a DB error (e.g. bad db) → text unchanged, no raise; multimodal history (list content) → does not raise and still matches on the text parts; `char_data=None` never pulls character books (attach a book to the character only and confirm it does NOT inject when `char_data=None`).
- **Native applier:** with a session whose `persisted_conversation_id` has an attached matching book, `_apply_world_info` injects into the final user message (string content AND multimodal content shapes); no `persisted_conversation_id` → payload unchanged; ordering — runs after `_apply_chat_dictionaries`; `CancelledError` propagates. Mirror the merged `Tests/UI/test_console_dictionaries_*` harness.
- **Full gate:** the new resolver tests + console-controller tests, `test_world_book_manager`, `import tldw_chatbook.app`.

## Decomposition
P2g-1 of 3. No migration. P2g-2 (inspector) reuses `_collect_active_world_books` for its summary; P2g-3 routes the legacy `chat_events.py` path through `apply_world_info_to_message` (fixing the character-gate bug) and deletes the dead legacy world-book UI.

## Acceptance criteria
- [ ] A shared `apply_world_info_to_message(db, conversation_id, char_data, message_text, history)` builds the world-info-injected message (legacy join order) or returns the text unchanged on no-match/no-books/no-conversation/error; never raises; multimodal-history-safe.
- [ ] `ConsoleChatController._apply_world_info` applies it at the 4 send sites after `_apply_chat_dictionaries`, on the final user message (string + multimodal shapes), off-thread, `CancelledError`-safe, no-op without a `persisted_conversation_id`.
- [ ] Native wiring is conversation-only (`char_data=None`) — a character-only attached book does NOT inject on native send.
- [ ] No conversation attached ⇒ native send byte-identical to today; no change to `chat_events.py`, `world_info_processor.py`, or the schema.
- [ ] Full gate green; `import tldw_chatbook.app` OK. Inspector → P2g-2; legacy fix + cleanup → P2g-3.
