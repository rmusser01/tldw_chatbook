# Roleplay P2g-3 — legacy gate fix + dead world-book UI deletion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Fix the legacy character-gate bug (conversation-attached world books never apply without a loaded character) by routing the legacy `chat_events.py` send through a count-returning resolver, and delete the confirmed-dead world-book sidebar UI. Completes P2g and the P2 Lore program.

**Architecture:** Factor `resolve_world_info_injection(...) -> (text, count)` from P2g-1's helper (which becomes a thin wrapper); replace the inline gated world-info logic in `chat_events.py` with a call to it outside the `if active_char_data:` gate; delete the world-book handlers/section/CSS/references.

**Tech Stack:** Python 3.11+, Textual, SQLite (`CharactersRAGDB`), pytest.

**Spec:** `Docs/superpowers/specs/2026-07-21-roleplay-p2g-3-legacy-gate-fix-cleanup-design.md`.

## Global Constraints

- **No schema migration** (v22).
- Part A: with-character output is **byte-identical** to the current inline pipeline (the resolver mirrors it). Preserve the `enable_world_info` config gate AND both indicator reactives `app.current_world_info_active` + `app.current_world_info_count` (they feed the user-visible `[World Info: N entries activated]` message).
- Part B is **world-book-UI-only**: do NOT delete `Chat_Window.py`, the rest of `chat_right_sidebar.py`, the 5 `ChatWindow` tests, or touch the `#chat-right-sidebar` query web (deferred to backlog task 412). Do NOT touch the separate `ccp-worldbook` surface.
- **Subagents:** first verify `git rev-parse --show-toplevel` == the worktree + branch `claude/roleplay-p2g-3-legacy-cleanup` (Agent-tool subagents can start in the MAIN checkout). Run tests **foreground, scoped** — NO background jobs, NO broad sweeps. Stage ONLY the task's files; never `git add -A`; never stage `.superpowers/`.
- **Test env** (venv in MAIN checkout; run from the worktree root):
  ```
  HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <targets> \
  -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
  ```

---

### Task 1: Factor `resolve_world_info_injection` (text, count)

**Files:**
- Modify: `tldw_chatbook/Character_Chat/world_info_resolver.py`
- Test: `Tests/Character_Chat/test_apply_world_info_to_message.py` (extend) + `Tests/Character_Chat/test_resolve_world_info_injection.py` (new)

**Interfaces:**
- Produces: `resolve_world_info_injection(db, conversation_id, char_data, message_text, history) -> tuple[str, int]`; `apply_world_info_to_message` becomes `return resolve_world_info_injection(...)[0]`.

- [ ] **Step 1: Write the failing tests**

Create `Tests/Character_Chat/test_resolve_world_info_injection.py`:
```python
import pytest

from tldw_chatbook.Character_Chat.world_info_resolver import (
    resolve_world_info_injection,
    apply_world_info_to_message,
)
from tldw_chatbook.Character_Chat.world_book_manager import WorldBookManager
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def wb_db(tmp_path):
    db = CharactersRAGDB(tmp_path / "resolve_wi.db", "test-client")
    yield db
    db.close_connection()


def _attach(db, conv_id, key, content, name="Lore"):
    db.add_conversation({"id": conv_id, "title": "C"})
    wb = WorldBookManager(db)
    book_id = wb.create_world_book(name)
    wb.create_world_book_entry(book_id, keys=[key], content=content)
    wb.associate_world_book_with_conversation(conv_id, book_id)
    return book_id


def test_returns_text_and_count_on_match(wb_db):
    _attach(wb_db, "c1", "dragon", "Dragons breathe fire.")
    text, count = resolve_world_info_injection(wb_db, "c1", None, "a dragon appears", [])
    assert "Dragons breathe fire." in text and count == 1


def test_conversation_book_applies_without_character(wb_db):
    # The gate-fix behavior: char_data=None (no character) still injects.
    _attach(wb_db, "c2", "griffin", "Griffins soar.")
    text, count = resolve_world_info_injection(wb_db, "c2", None, "a griffin flies", [])
    assert "Griffins soar." in text and count == 1


def test_no_match_returns_unchanged_zero(wb_db):
    _attach(wb_db, "c3", "dragon", "x")
    assert resolve_world_info_injection(wb_db, "c3", None, "hello", []) == ("hello", 0)


def test_no_conversation_and_db_error_zero(wb_db):
    assert resolve_world_info_injection(wb_db, None, None, "a dragon appears", []) == ("a dragon appears", 0)
    assert resolve_world_info_injection(object(), "cX", None, "a dragon appears", []) == ("a dragon appears", 0)


def test_apply_wrapper_returns_only_text(wb_db):
    _attach(wb_db, "c4", "dragon", "Dragons breathe fire.")
    text = apply_world_info_to_message(wb_db, "c4", None, "a dragon appears", [])
    text2, _ = resolve_world_info_injection(wb_db, "c4", None, "a dragon appears", [])
    assert text == text2 and isinstance(text, str)
```

- [ ] **Step 2: Run to verify they fail**
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/Character_Chat/test_resolve_world_info_injection.py -q -p no:cacheprovider -o addopts="" --timeout=120 --timeout-method=thread
```
Expected: FAIL — `ImportError: cannot import name 'resolve_world_info_injection'`.

- [ ] **Step 3: Factor the resolver**

In `world_info_resolver.py`, add `Tuple` to the typing import. Add `resolve_world_info_injection` (the current `apply_world_info_to_message` body, but returning `(text, count)`), and replace `apply_world_info_to_message`'s body with a thin wrapper:
```python
def resolve_world_info_injection(
    db: Any,
    conversation_id: Optional[str],
    char_data: Optional[Dict[str, Any]],
    message_text: str,
    history: List[Dict[str, Any]],
) -> Tuple[str, int]:
    """Return ``(injected_message_text, matched_entry_count)``.

    Same collect→build→process→format→join as ``apply_world_info_to_message``,
    but also reports how many world-info entries matched (for the legacy
    ``[World Info: N entries]`` indicator). Never raises; returns
    ``(message_text, 0)`` on no-match / no-books / no-conversation / error.
    """
    if not isinstance(message_text, str):
        return message_text, 0
    try:
        world_books, has_character_book = _collect_active_world_books(
            db, conversation_id, char_data
        )
        if not (has_character_book or world_books):
            return message_text, 0
        from .world_info_processor import WorldInfoProcessor

        processor = WorldInfoProcessor(
            character_data=char_data if has_character_book else None,
            world_books=world_books or None,
        )
        result = processor.process_messages(message_text, history or [])
        matched = result.get("matched_entries") or []
        if not matched:
            return message_text, 0
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
        return "\n\n".join(parts), len(matched)
    except Exception:
        logger.opt(exception=True).debug(
            "world-info: apply failed; returning message text unchanged"
        )
        return message_text, 0
```
Replace the entire body of `apply_world_info_to_message` (keep its signature + docstring) with:
```python
    return resolve_world_info_injection(
        db, conversation_id, char_data, message_text, history
    )[0]
```
Add `resolve_world_info_injection` to `__all__`.

- [ ] **Step 4: Run to verify + no regression**

Run the Step-2 command (PASS, 5 tests), then the existing helper tests (unchanged behavior):
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/Character_Chat/test_apply_world_info_to_message.py Tests/Character_Chat/test_resolve_world_info_injection.py \
-q -p no:cacheprovider -o addopts="" --timeout=120 --timeout-method=thread
```
Expected: all pass (the wrapper preserves `apply_world_info_to_message` behavior).

- [ ] **Step 5: Commit**
```bash
git add tldw_chatbook/Character_Chat/world_info_resolver.py Tests/Character_Chat/test_resolve_world_info_injection.py
git commit -m "feat(lore): resolve_world_info_injection returns (text, count); wrap apply_"
```

---

### Task 2: Legacy send gate fix (route `chat_events.py` through the resolver)

**Files:**
- Modify: `tldw_chatbook/Event_Handlers/Chat_Events/chat_events.py`
- Test: `Tests/Character_Chat/test_character_world_book_send_path.py` (extend, if the file exists) OR a focused new test

**Interfaces:**
- Consumes: `resolve_world_info_injection` (Task 1).

- [ ] **Step 1: Write the failing/guarding test**

The gate-fix behavior (conversation books apply with `char_data=None`) is already proven by Task 1's `test_conversation_book_applies_without_character`. For this task, add a **structural + regression** guard. Extend `Tests/Character_Chat/test_character_world_book_send_path.py` (the P2g-1 send-path test) with a test replicating the NEW chat_events wiring (no character, conversation book → injected + count), mirroring how that file builds inputs:
```python
def test_legacy_style_wiring_applies_without_character():
    # Mirrors the post-fix chat_events.py call: no character, conversation book.
    from tldw_chatbook.Character_Chat.world_info_resolver import (
        resolve_world_info_injection,
    )
    # (build a real db + conversation-attached book as the other tests do)
    text, count = resolve_world_info_injection(db, conv_id, None, "a dragon appears", [])
    assert count >= 1 and "Dragons breathe fire." in text
```
(Adapt to the file's existing fixtures. If the file has no real-db fixture, put this in `test_resolve_world_info_injection.py` instead — the key is proving char_data=None injects.)

- [ ] **Step 2: Edit `chat_events.py`** (find each region by content — line numbers drift)

(a) **Remove** the `world_info_processor = None` initialization (~840).

(b) **Remove** the inline world-info **build** block — the `if get_cli_setting("character_chat", "enable_world_info", True):` block that builds `world_books` + `WorldInfoProcessor` (~868-931), which is nested inside `if active_char_data:`. Leave the rest of the `if active_char_data:` block (system-prompt handling, the `else:` branch) intact.

(c) **Replace** the consume/join block (the `if world_info_processor:` block spanning ~1450-1503, from `message_text_with_world_info = message_text_with_handoff` through the `except Exception as e: ... # Continue without world info on error`) with — placed at the same spot, OUTSIDE the `if active_char_data:` gate (it already is, ~1450):
```python
    message_text_with_world_info = message_text_with_handoff
    if get_cli_setting("character_chat", "enable_world_info", True):
        from tldw_chatbook.Character_Chat.world_info_resolver import (
            resolve_world_info_injection,
        )

        message_text_with_world_info, _wi_count = resolve_world_info_injection(
            db,
            active_conversation_id,
            active_char_data,
            message_text_with_handoff,
            chat_history_for_api,
        )
        app.current_world_info_active = (
            message_text_with_world_info != message_text_with_handoff
        )
        app.current_world_info_count = _wi_count
```
Keep the `# --- 10.7 ...` comment header and the `message_text_with_handoff = apply_current_handoff_context(...)` line above it unchanged. `message_text_with_world_info` still flows to the API dispatch exactly as before.

- [ ] **Step 3: Verify no dangling `world_info_processor`**
```bash
grep -n "world_info_processor\|world_info_injections\|world_info_result" tldw_chatbook/Event_Handlers/Chat_Events/chat_events.py || echo "(none — clean)"
```
Expected: no matches (all removed). Then confirm the handler imports:
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c "import tldw_chatbook.Event_Handlers.Chat_Events.chat_events; print('OK')"
```

- [ ] **Step 4: Run the test + app import**

Run the Step-1 test file (PASS), then `import tldw_chatbook.app` (OK).

- [ ] **Step 5: Commit**
```bash
git add tldw_chatbook/Event_Handlers/Chat_Events/chat_events.py Tests/Character_Chat/test_character_world_book_send_path.py
git commit -m "fix(lore): legacy send applies conversation world books without a character"
```

---

### Task 3: Delete the dead world-book sidebar UI (scoped)

**Files:**
- Delete: `tldw_chatbook/Event_Handlers/Chat_Events/chat_events_worldbooks.py`
- Modify: `tldw_chatbook/Event_Handlers/Chat_Events/chat_events.py`, `tldw_chatbook/app.py`, `tldw_chatbook/Widgets/Chat_Widgets/chat_right_sidebar.py`, `tldw_chatbook/css/layout/_sidebars.tcss`
- Delete (if any): tests of the removed world-book handlers

- [ ] **Step 1: Establish the baseline (what references the world-book code)**
```bash
grep -rn "chat_events_worldbooks\|handle_worldbook\|CHAT_WORLDBOOK_BUTTON_HANDLERS\|chat-worldbook" tldw_chatbook/ Tests/ 2>/dev/null | grep -v "\.backup" | grep -v "ccp"
```
Note every live reference (expected: `chat_events.py` import + 3 `refresh_active_worldbooks` calls + the handler merge; `app.py` import + 3 `chat-worldbook-*` routing branches; the section in `chat_right_sidebar.py`; the CSS). Confirm none are the separate `ccp-worldbook` surface.

- [ ] **Step 2: Remove the references, then delete the file**

- `chat_events.py`: remove the `from ... import chat_events_worldbooks` import, the three `await chat_events_worldbooks.refresh_active_worldbooks(app)` calls (leave the surrounding conversation-state code), and the `**chat_events_worldbooks.CHAT_WORLDBOOK_BUTTON_HANDLERS,` line in the button-handler dict.
- `app.py`: remove the `chat_events_worldbooks` import and the three `chat-worldbook-*` routing branches (the `elif`/`if` blocks in `on_input_changed`/`on_list_view_selected`/`on_checkbox_changed` that call `chat_events_worldbooks.handle_worldbook_*`). Ensure the surrounding `elif` chains stay syntactically valid.
- `chat_right_sidebar.py`: delete only the world-book collapsible **section** (the `Collapsible(title="World Books", ...)` block, ~551-627). Keep the rest of `create_chat_right_sidebar`.
- `_sidebars.tcss`: delete the five world-book rules (`.worldbook-association-controls`, `.worldbook-priority-select`, `#chat-worldbook-available-listview`, `#chat-worldbook-active-listview`, `#chat-worldbook-details-display`, ~370-396).
- Delete any test file that specifically tests the removed world-book handlers (grep from Step 1; likely none — the sidebar was unreachable).
- Then delete `chat_events_worldbooks.py`.

- [ ] **Step 3: Verify no dangling references + app import**
```bash
grep -rn "chat_events_worldbooks\|handle_worldbook\|CHAT_WORLDBOOK_BUTTON_HANDLERS\|chat-worldbook" tldw_chatbook/ Tests/ 2>/dev/null | grep -v "\.backup" | grep -v "ccp"
```
Expected: **no output** (all live references gone; `.backup` and `ccp-worldbook` excluded). Then:
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c "import tldw_chatbook.app; print('APP IMPORT OK')"
```
Expected: `APP IMPORT OK`.

- [ ] **Step 4: Broader regression + full P2g-3 gate**

Run a chat/console-focused slice to confirm nothing broke (the deleted symbols were referenced only by the removed code), plus the P2g-3 gate:
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/Character_Chat/test_resolve_world_info_injection.py Tests/Character_Chat/test_apply_world_info_to_message.py \
Tests/Character_Chat/test_character_world_book_send_path.py \
-q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: all pass; `APP IMPORT OK` (from Step 3).

- [ ] **Step 5: Commit**
```bash
git add tldw_chatbook/Event_Handlers/Chat_Events/chat_events.py tldw_chatbook/app.py \
  tldw_chatbook/Widgets/Chat_Widgets/chat_right_sidebar.py tldw_chatbook/css/layout/_sidebars.tcss
git rm tldw_chatbook/Event_Handlers/Chat_Events/chat_events_worldbooks.py
git commit -m "chore(chat): delete dead world-book sidebar UI + handlers"
```

---

## Notes for reviewers

- **No migration.** Part A: with-character byte-identical; the only behavior change is conversation books applying without a character. `enable_world_info` + `current_world_info_active` + `current_world_info_count` (the `[World Info: N entries]` indicator) preserved.
- `apply_world_info_to_message` is now a thin wrapper over `resolve_world_info_injection` — native Console (P2g-1) is unchanged.
- Part B is **world-book-UI-only**: `Chat_Window.py`, the rest of `chat_right_sidebar.py`, the `#chat-right-sidebar` web, and the 5 `ChatWindow` tests are intentionally untouched (deferred to backlog task 412). The separate `ccp-worldbook` surface is untouched.
- The grep sweep in Task 3 must return no live references before merge.
