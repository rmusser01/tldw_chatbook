# Roleplay P2g-1 — native-Console world-info send — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make conversation-attached world books take effect on the **native Console** send (they apply on the legacy Chat send today, never on native Console). Mirror the merged native-Console dictionary send.

**Architecture:** A shared, never-raising lib helper `apply_world_info_to_message` (faithful to the legacy build→process→format→join) + a native applier `ConsoleChatController._apply_world_info` called at the 4 send sites after `_apply_chat_dictionaries`, wired conversation-only (`char_data=None`).

**Tech Stack:** Python 3.11+, Textual, SQLite (`CharactersRAGDB`), pytest.

**Spec:** `Docs/superpowers/specs/2026-07-21-roleplay-p2g-1-native-console-world-info-design.md`.

## Global Constraints

- **No schema migration** (v22). **No change to `chat_events.py` or `world_info_processor.py`** (legacy path + engine are P2g-3 / unchanged).
- The helper **never raises** — no conversation/books/match/error → returns the text unchanged.
- **World-info runs AFTER `_apply_chat_dictionaries`** at each of the **4 send sites** (submit / retry / continue / regenerate). The **preview** `_apply_chat_dictionaries` call inside `build_context_snapshot` is NOT a send site — do not add world-info there.
- **Conversation-only wiring:** the bound native applier passes `char_data=None` (native Console has no character). The shared helper is general (takes `char_data`) for P2g-3's legacy reuse.
- **Multimodal-safe:** the controller normalizes the message + history to plain text (text parts only) before scanning; injects into the text of the final user message, preserving image parts.
- Legacy join order is exactly `at_start → before_char → message → after_char → at_end`, joined `"\n\n"`.
- Leave `_apply_chat_dictionaries` and the dict lib untouched.
- **Staging:** each task stages ONLY its files. Never `git add -A`; never stage `.superpowers/`.
- **Test env** (venv in MAIN checkout; run from the worktree root; UI/integration tests slow):
  ```
  HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <targets> \
  -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
  ```

---

### Task 1: Shared helper `apply_world_info_to_message`

**Files:**
- Create: `tldw_chatbook/Character_Chat/world_info_resolver.py`
- Test: `Tests/Character_Chat/test_apply_world_info_to_message.py`

**Interfaces:**
- Consumes: `WorldBookManager.get_world_books_for_conversation`, `resolve_character_world_books`, `WorldInfoProcessor` (all existing).
- Produces: `apply_world_info_to_message(db, conversation_id: str | None, char_data: dict | None, message_text: str, history: list[dict]) -> str`; internal `_collect_active_world_books(db, conversation_id, char_data) -> tuple[list[dict], bool]` (for P2g-2 reuse).

- [ ] **Step 1: Write the failing tests**

Create `Tests/Character_Chat/test_apply_world_info_to_message.py`:
```python
import pytest

from tldw_chatbook.Character_Chat.world_info_resolver import (
    apply_world_info_to_message,
)
from tldw_chatbook.Character_Chat.world_book_manager import WorldBookManager
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def wb_db(tmp_path):
    db = CharactersRAGDB(tmp_path / "world_info_resolver.db", "test-client")
    yield db
    db.close_connection()


def _attach_book(db, conv_id, key, content, title="Case"):
    db.add_conversation({"id": conv_id, "title": title})
    wb = WorldBookManager(db)
    book_id = wb.create_world_book("Lore")
    wb.create_world_book_entry(book_id, keys=[key], content=content)
    wb.associate_world_book_with_conversation(conv_id, book_id)
    return book_id


def test_injects_matched_conversation_book(wb_db):
    _attach_book(wb_db, "conv-1", "dragon", "Dragons breathe fire.")
    out = apply_world_info_to_message(wb_db, "conv-1", None, "a dragon appears", [])
    assert "Dragons breathe fire." in out
    assert out.endswith("a dragon appears") or "a dragon appears" in out
    assert out != "a dragon appears"


def test_no_match_returns_unchanged(wb_db):
    _attach_book(wb_db, "conv-2", "dragon", "Dragons breathe fire.")
    assert apply_world_info_to_message(wb_db, "conv-2", None, "hello there", []) == "hello there"


def test_no_conversation_returns_unchanged(wb_db):
    assert apply_world_info_to_message(wb_db, None, None, "a dragon appears", []) == "a dragon appears"


def test_no_books_returns_unchanged(wb_db):
    wb_db.add_conversation({"id": "conv-3", "title": "Empty"})
    assert apply_world_info_to_message(wb_db, "conv-3", None, "a dragon appears", []) == "a dragon appears"


def test_db_error_returns_unchanged():
    # A bogus db object: the helper must swallow the error and return the text.
    assert apply_world_info_to_message(object(), "conv-x", None, "a dragon appears", []) == "a dragon appears"


def test_non_string_message_returned_as_is(wb_db):
    assert apply_world_info_to_message(wb_db, "conv-1", None, None, []) is None


def test_character_only_book_not_injected_when_char_data_none(wb_db):
    # A book attached to a CHARACTER (char_data=None here) must not inject.
    char_id = wb_db.add_character_card({"name": "Hero"})
    wb = WorldBookManager(wb_db)
    book_id = wb.create_world_book("CharLore")
    wb.create_world_book_entry(book_id, keys=["griffin"], content="Griffins soar.")
    wb.attach_world_book_to_character(book_id, char_id)
    wb_db.add_conversation({"id": "conv-4", "title": "NoConvBook"})
    out = apply_world_info_to_message(wb_db, "conv-4", None, "a griffin flies", [])
    assert out == "a griffin flies"  # char_data is None → character books never apply
```

- [ ] **Step 2: Run to verify they fail**
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/Character_Chat/test_apply_world_info_to_message.py -q -p no:cacheprovider -o addopts="" --timeout=120 --timeout-method=thread
```
Expected: FAIL — `ModuleNotFoundError: ...world_info_resolver`.

- [ ] **Step 3: Create the helper** (replicates the legacy `chat_events.py:868-925` build + the `:1470-1489` join)

Create `tldw_chatbook/Character_Chat/world_info_resolver.py`:
```python
"""Shared world-info send-path resolver (Roleplay P2g).

Builds the world-info-injected message text for a send, composing the same
sources the legacy chat_events path does (conversation-attached books ∪
character-attached snapshots ∪ a native character_book) — so the native Console
(P2g-1) and, later, the legacy path (P2g-3) share one faithful implementation.
Never raises: any problem returns the message text unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


def _collect_active_world_books(
    db: Any, conversation_id: Optional[str], char_data: Optional[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], bool]:
    """Collect the world books that apply to this send.

    Args:
        db: A ``CharactersRAGDB`` (or None).
        conversation_id: The active conversation (string UUID) or None.
        char_data: The active character record, or None (native Console).

    Returns:
        ``(world_books, has_character_book)`` — the conversation-attached books
        unioned with character-attached snapshots (conversation wins on a name
        collision), and whether ``char_data`` carries a native ``character_book``.
        Never raises.
    """
    world_books: List[Dict[str, Any]] = []
    if conversation_id and db is not None:
        try:
            from .world_book_manager import WorldBookManager

            world_books = WorldBookManager(db).get_world_books_for_conversation(
                str(conversation_id), enabled_only=True
            )
        except Exception:
            logger.opt(exception=True).debug(
                "world-info: could not load conversation world books"
            )
            world_books = []

    has_character_book = False
    extensions = char_data.get("extensions", {}) if isinstance(char_data, dict) else {}
    if isinstance(extensions, dict) and extensions.get("character_book"):
        has_character_book = True

    try:
        from .world_book_manager import resolve_character_world_books

        character_books = resolve_character_world_books(
            char_data, {str(b.get("name")) for b in world_books}
        )
    except Exception:
        character_books = []
    if character_books:
        world_books = world_books + character_books

    return world_books, has_character_book


def apply_world_info_to_message(
    db: Any,
    conversation_id: Optional[str],
    char_data: Optional[Dict[str, Any]],
    message_text: str,
    history: List[Dict[str, Any]],
) -> str:
    """Return ``message_text`` with matched world-info injected, or unchanged.

    Args:
        db: A ``CharactersRAGDB`` (or None).
        conversation_id: The active conversation (string UUID) or None.
        char_data: The active character record, or None (conversation-only).
        message_text: The current user message text (already plain string).
        history: Prior messages as ``{"role","content": str}`` (string content;
            the caller normalizes multimodal content to text before calling).

    Returns:
        The message text wrapped with world-info injections in the order
        ``at_start → before_char → message → after_char → at_end`` (``"\\n\\n"``
        separated), or the original ``message_text`` when nothing matches / no
        books / no conversation / any error. Never raises.
    """
    if not isinstance(message_text, str):
        return message_text
    try:
        world_books, has_character_book = _collect_active_world_books(
            db, conversation_id, char_data
        )
        if not (has_character_book or world_books):
            return message_text
        from .world_info_processor import WorldInfoProcessor

        processor = WorldInfoProcessor(
            character_data=char_data if has_character_book else None,
            world_books=world_books or None,
        )
        result = processor.process_messages(message_text, history or [])
        if not result.get("matched_entries"):
            return message_text
        formatted = processor.format_injections(result.get("injections", {}))
        parts: List[str] = []
        if formatted.get("at_start"):
            parts.append(formatted["at_start"])
        if formatted.get("before_char"):
            parts.append(formatted["before_char"])
        parts.append(message_text)
        if formatted.get("after_char"):
            parts.append(formatted["after_char"])
        if formatted.get("at_end"):
            parts.append(formatted["at_end"])
        return "\n\n".join(parts)
    except Exception:
        logger.opt(exception=True).debug(
            "world-info: apply failed; returning message text unchanged"
        )
        return message_text


__all__ = ["apply_world_info_to_message"]
```

- [ ] **Step 4: Run to verify they pass**

Run the Step-2 command. Expected: PASS (7 tests).

- [ ] **Step 5: Commit**
```bash
git add tldw_chatbook/Character_Chat/world_info_resolver.py Tests/Character_Chat/test_apply_world_info_to_message.py
git commit -m "feat(lore): apply_world_info_to_message shared send-path resolver"
```

---

### Task 2: `ConsoleChatController._apply_world_info` + param + 4 send sites

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Test: `Tests/Chat/test_console_world_info_application.py` (new)

**Interfaces:**
- Consumes: `self._world_info_applier` (new), `ConsoleMessageRole`, `COMMAND_PREFIX` (already imported/used by `_apply_chat_dictionaries`).
- Produces: `ConsoleChatController.__init__` gains `world_info_applier: Callable[[str | None, str, list], str] | None = None`; `async _apply_world_info(provider_messages, session_id) -> provider_messages`; called after `_apply_chat_dictionaries` at the 4 send sites. The applier callback contract is `applier(conversation_id, message_text, history) -> str`.

- [ ] **Step 1: Write the failing tests**

Read `Tests/Chat/test_console_chat_controller.py` / `Tests/Chat/test_console_dictionary_application.py` for how a `ConsoleChatController` is constructed with a stub applier in a unit test, and mirror that harness. Create `Tests/Chat/test_console_world_info_application.py` with tests that:
- Build a controller with a **stub** `world_info_applier` that returns `f"[WI]\n\n{text}"` (so the test asserts the controller's splicing, not real world-info — the real resolver is covered by Task 1 and Task 3).
- `test_apply_world_info_wraps_final_user_message`: a session pinned to a conversation id, `provider_messages` ending in a user string message → after `_apply_world_info`, the final user message content is `"[WI]\n\n<original>"`; earlier messages untouched.
- `test_apply_world_info_noop_without_conversation`: session with no `persisted_conversation_id` → payload unchanged.
- `test_apply_world_info_multimodal_history_and_message`: the final user message has list/multimodal content `[{"type":"text","text":"a dragon"},{"type":"image_url",...}]` and history contains a multimodal message → does not raise; the applier is called with a **string** message + string-content history (assert the stub captured string types); the text part is wrapped and the image part preserved.
- `test_apply_world_info_command_message_skipped`: a final user message starting with `COMMAND_PREFIX` → unchanged.
- `test_apply_world_info_applier_none`: controller with `world_info_applier=None` → unchanged.

(Match the exact construction/session-seeding pattern from the existing controller tests; adapt fixture names to reality.)

- [ ] **Step 2: Run to verify they fail**
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/Chat/test_console_world_info_application.py -q -p no:cacheprovider -o addopts="" --timeout=180 --timeout-method=thread
```
Expected: FAIL (no `_apply_world_info` / no `world_info_applier` param).

- [ ] **Step 3: Add the param**

In `ConsoleChatController.__init__`, next to `chat_dictionary_applier` (param ~:281, assignment ~:308), add:
```python
        world_info_applier: "Callable[[str | None, str, list], str] | None" = None,
```
and in the body:
```python
        self._world_info_applier = world_info_applier
```

- [ ] **Step 4: Add `_apply_world_info` + a history normalizer**

Add a module-level helper (near the top of `console_chat_controller.py`, after the imports):
```python
def _normalize_world_info_history(
    messages: "list[dict[str, Any]]",
) -> "list[dict[str, Any]]":
    """Flatten messages to ``{"role","content": str}`` for world-info scanning.

    ``WorldInfoProcessor.process_messages`` types content as ``str``; native
    provider messages may carry multimodal list content, so extract the text
    parts (joined) and drop images before scanning.
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "\n".join(
                part["text"]
                for part in content
                if isinstance(part, dict)
                and part.get("type") == "text"
                and isinstance(part.get("text"), str)
            )
        else:
            text = ""
        out.append({"role": message.get("role", ""), "content": text})
    return out
```

Add the applier method next to `_apply_chat_dictionaries` (mirror its structure; adapt for the `(conv_id, text, history)` callback + the wrap-not-substitute semantics):
```python
    async def _apply_world_info(
        self, provider_messages: list[dict[str, Any]], session_id: str
    ) -> list[dict[str, Any]]:
        """Inject conversation world-info into the final user message of the
        ephemeral provider payload (never the stored transcript).

        Runs AFTER `_apply_chat_dictionaries` so world-info matches the
        dict-substituted text the model will see. Conversation-only (the bound
        applier passes `char_data=None`). Offloaded via `asyncio.to_thread`;
        any failure returns the payload unchanged; `CancelledError` re-raised.
        """
        applier = self._world_info_applier
        if applier is None:
            return provider_messages

        session = next((s for s in self.store.sessions() if s.id == session_id), None)
        conversation_id = (
            session.persisted_conversation_id if session is not None else None
        )
        if not conversation_id:
            return provider_messages

        final_index: int | None = None
        for index in range(len(provider_messages) - 1, -1, -1):
            if provider_messages[index].get("role") == ConsoleMessageRole.USER.value:
                final_index = index
                break
        if final_index is None:
            return provider_messages

        message = provider_messages[final_index]
        content = message.get("content")
        if isinstance(content, str) and content.startswith(COMMAND_PREFIX):
            return provider_messages

        history = _normalize_world_info_history(provider_messages[:final_index])

        try:
            if isinstance(content, str):
                injected: Any = await asyncio.to_thread(
                    applier, conversation_id, content, history
                )
                if injected == content:
                    return provider_messages
                new_content = injected
            elif isinstance(content, list):
                combined = "\n".join(
                    part["text"]
                    for part in content
                    if isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                )
                if not combined:
                    return provider_messages
                injected = await asyncio.to_thread(
                    applier, conversation_id, combined, history
                )
                if injected == combined:
                    return provider_messages
                new_parts: list[Any] = []
                first_text_done = False
                for part in content:
                    if (
                        isinstance(part, dict)
                        and part.get("type") == "text"
                        and isinstance(part.get("text"), str)
                    ):
                        if not first_text_done:
                            new_parts.append({**part, "text": injected})
                            first_text_done = True
                        # subsequent text parts are folded into the injected block
                        continue
                    new_parts.append(part)
                new_content = new_parts
            else:
                return provider_messages
        except asyncio.CancelledError:
            raise
        except Exception:
            return provider_messages

        new_messages = list(provider_messages)
        new_messages[final_index] = {**message, "content": new_content}
        return new_messages
```

- [ ] **Step 5: Call it at the 4 send sites**

After each of the FOUR two-line send-site calls `provider_messages = await self._apply_chat_dictionaries(provider_messages, session_id)` — at `submit_draft` (~:431), retry (~:1066), `continue_from_message` (~:1114), `regenerate_message` (~:1170) — add immediately after:
```python
        provider_messages = await self._apply_world_info(
            provider_messages, session_id
        )
```
**Do NOT** add it after the single-line `_apply_chat_dictionaries` call inside `build_context_snapshot` (~:1241) — that is a preview builder, not a send.

- [ ] **Step 6: Run to verify + import check**

Run the Step-2 command (PASS), then:
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c "import tldw_chatbook.Chat.console_chat_controller; print('OK')"
```

- [ ] **Step 7: Commit**
```bash
git add tldw_chatbook/Chat/console_chat_controller.py Tests/Chat/test_console_world_info_application.py
git commit -m "feat(console): native-Console world-info applier at the 4 send sites"
```

---

### Task 3: `chat_screen` wiring + end-to-end real-DB test

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Test: `Tests/UI/test_console_world_info_send_integration.py` (new)

**Interfaces:**
- Consumes: `apply_world_info_to_message` (Task 1), the `world_info_applier` param (Task 2).
- Produces: `ChatScreen._console_world_info_applier(conversation_id, message_text, history) -> str`; passed to `ConsoleChatController(..., world_info_applier=...)`.

- [ ] **Step 1: Write the failing test**

Read `Tests/UI/test_console_dictionary_send_integration.py` (the `_build_test_app` / `ConsoleHarness` / `_CapturingGateway` end-to-end harness) and mirror it for world books. Create `Tests/UI/test_console_world_info_send_integration.py` that: seeds a real `CharactersRAGDB` with a conversation-attached world book (`WorldBookManager.associate_world_book_with_conversation`, entry keyed to a word), pins a native session to that conversation, drives a native send with a message containing that word, and asserts the **captured outbound `provider_messages`** final user message contains the injected world-info content while the persisted transcript keeps the raw text. Skeleton (adapt to the real harness):
```python
# Mirror Tests/UI/test_console_dictionary_send_integration.py's harness exactly,
# swapping the dictionary seam for the world-book seam:
#   wb = WorldBookManager(db); book = wb.create_world_book("Lore")
#   wb.create_world_book_entry(book, keys=["dragon"], content="Dragons breathe fire.")
#   db.add_conversation({"id": conv_id, ...}); wb.associate_world_book_with_conversation(conv_id, book)
#   ... pin the native session to conv_id, send "a dragon appears",
#   assert "Dragons breathe fire." in _final_user_content(gateway.captured)
#   assert the persisted transcript row still has the raw "a dragon appears"
```

- [ ] **Step 2: Run to verify it fails**
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/UI/test_console_world_info_send_integration.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: FAIL — the outbound payload has no world-info (no wiring yet).

- [ ] **Step 3: Add the bound applier** (mirror `_console_chat_dictionary_applier`, `chat_screen.py:5753-5775`)

Add near `_console_chat_dictionary_applier`:
```python
    def _console_world_info_applier(
        self, conversation_id: str | None, message_text: str, history: list
    ) -> str:
        """Bound applier handed to the native Console controller: inject the
        active CONVERSATION world-info into a send's text (never raises).

        Resolves the db lazily. Conversation-only: ``char_data`` is ``None``
        (native sessions carry no character card).
        """
        db = getattr(self.app_instance, "chachanotes_db", None)
        if db is None or not conversation_id or not isinstance(message_text, str):
            return message_text
        from ...Character_Chat.world_info_resolver import apply_world_info_to_message

        return apply_world_info_to_message(
            db, conversation_id, None, message_text, history or []
        )
```

- [ ] **Step 4: Wire it into the controller construction**

At the `ConsoleChatController(...)` construction (`chat_screen.py:2724`), next to `chat_dictionary_applier=self._console_chat_dictionary_applier,` (~:2749), add:
```python
                world_info_applier=self._console_world_info_applier,
```

- [ ] **Step 5: Run to verify + full gate + app import**

Run the Step-2 command (PASS), then:
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/Character_Chat/test_apply_world_info_to_message.py Tests/Chat/test_console_world_info_application.py \
Tests/UI/test_console_world_info_send_integration.py Tests/UI/test_console_dictionary_send_integration.py \
Tests/Character_Chat/test_world_book_manager.py \
-q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c "import tldw_chatbook.app; print('APP IMPORT OK')"
```
Expected: all pass (the dict send-integration test confirms no native-send regression); `APP IMPORT OK`.

- [ ] **Step 6: Commit**
```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_world_info_send_integration.py
git commit -m "feat(console): wire native-Console world-info applier (conversation-only)"
```

---

## Notes for reviewers

- **No migration; no `chat_events.py` / `world_info_processor.py` change.** The only production files touched are `world_info_resolver.py` (new), `console_chat_controller.py`, and `chat_screen.py`.
- **Ordering:** world-info is applied AFTER `_apply_chat_dictionaries` at the 4 send sites, and NOT in the `build_context_snapshot` preview.
- **Conversation-only:** the bound applier passes `char_data=None`; a character-only attached book must not inject on native send (Task 1 test).
- **Never-raises:** the helper returns text unchanged on any failure; the applier re-raises only `CancelledError`.
- **Multimodal:** history + message normalized to text before scanning; image parts preserved.
- Inspector "what's in play" → P2g-2 (reuses `_collect_active_world_books`); legacy gate fix + dead-code cleanup → P2g-3.
