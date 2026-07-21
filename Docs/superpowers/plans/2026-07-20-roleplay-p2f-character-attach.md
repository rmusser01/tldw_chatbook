# Roleplay P2f — character-level world-book attach — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user attach a standalone Lore (world) book to a *character* as a portable embedded snapshot, so its entries apply whenever that character is active — the world-book parallel to the merged character-dictionary attach (P1f).

**Architecture:** Snapshots live in a new `character_cards.extensions['character_world_books']` key (no schema migration — `extensions` is arbitrary JSON, optimistic-locked). A pure resolver unions them into the legacy send path with conversation-wins name dedup; a new picker + a sibling docked widget + screen handlers mirror the P1f dictionary machinery. `WorldInfoProcessor` is unchanged.

**Tech Stack:** Python 3.11+, Textual, SQLite (`CharactersRAGDB`), pytest.

**Spec:** `Docs/superpowers/specs/2026-07-20-roleplay-p2f-character-attach-design.md`.

## Global Constraints

- **No schema migration.** Schema stays **v22**. Storage is `extensions['character_world_books']`, written via `update_character_card({"extensions": ext}, expected_version=record["version"])` (read-modify-write the whole `extensions` dict).
- **Snapshot = `export_world_book(id)` output augmented with the source book's `enabled`** (export omits book-level `enabled`).
- **Conversation-wins dedup by name.** A character snapshot is dropped if its name is among the (already enabled-only) conversation-attached book names.
- **Union runs BEFORE** `chat_events`' `if has_character_book or world_books:` processor-init guard (else an attached-only character applies nothing).
- **All name comparisons use `str(...)`** (hostile/imported cards may carry non-str names).
- **Never raise on untrusted embedded content**; **dedup by name at BOTH the widget render and the resolver**; use `_coerce_bool` for embedded booleans.
- **Widget is I/O-free**; screen does all DB work off-thread (`asyncio.to_thread`), wrapped → `_notify`, `_io_dialog_active` / `group="personas-io"` re-entrancy guard.
- Button handlers are named `_attach_pressed` / `_detach_pressed` (NOT `_attach`/`_detach` — those shadow Textual `DOMNode` internals).
- **Leave untouched:** `DictionaryPicker`, `PersonasCharacterDictionariesWidget`, the dict service, and `WorldInfoProcessor`.
- **Console "what's in play" + native-Console send + legacy-UI retirement are OUT of scope** (P2g).
- **Staging:** each task stages ONLY its own files. Never `git add -A`; never stage `.superpowers/`.
- **Test env** (venv in MAIN checkout; run from the worktree root; UI tests are slow ~30-60s):
  ```
  HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <targets> \
  -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
  ```

---

### Task 1: `WorldBookManager` character-attach service methods

**Files:**
- Modify: `tldw_chatbook/Character_Chat/world_book_manager.py` (add a `_coerce_bool` module helper next to `_coerce_int`; add the character methods after `get_conversations_for_world_book`)
- Test: `Tests/Character_Chat/test_world_book_manager.py`

**Interfaces:**
- Consumes: existing `get_world_book`, `export_world_book`, `create_world_book`, `create_world_book_entry`; `self.db.get_character_card_by_id`, `self.db.update_character_card`, `self.db.add_character_card`.
- Produces: `attach_world_book_to_character(world_book_id: int, character_id: int) -> Dict[str,Any]` (`{"world_book_id","character_id","name","attached":bool}`); `detach_world_book_from_character(character_id: int, name: str) -> Dict[str,Any]` (`{"character_id","name","detached":bool}`); `get_world_books_for_character(character_id: int) -> List[Dict[str,Any]]` (`[{"name":str,"entry_count":int,"enabled":bool}]`); module helper `_coerce_bool(value, default) -> bool`.

- [ ] **Step 1: Write the failing tests**

Append to `Tests/Character_Chat/test_world_book_manager.py` (uses the existing `wb_manager` fixture; `wb_manager.db` is a real `CharactersRAGDB`):
```python
def _make_character(db, name="Hero"):
    return db.add_character_card({"name": name})


def test_attach_world_book_to_character_round_trip(wb_manager):
    db = wb_manager.db
    char_id = _make_character(db)
    book_id = wb_manager.create_world_book("Lore")
    wb_manager.create_world_book_entry(book_id, keys=["dragon"], content="A dragon.")

    res = wb_manager.attach_world_book_to_character(book_id, char_id)
    assert res["attached"] is True and res["name"] == "Lore"

    got = wb_manager.get_world_books_for_character(char_id)
    assert [g["name"] for g in got] == ["Lore"]
    assert got[0]["entry_count"] == 1 and got[0]["enabled"] is True

    # Idempotent by name.
    assert wb_manager.attach_world_book_to_character(book_id, char_id)["attached"] is False
    assert len(wb_manager.get_world_books_for_character(char_id)) == 1

    # Detach.
    assert wb_manager.detach_world_book_from_character(char_id, "Lore")["detached"] is True
    assert wb_manager.get_world_books_for_character(char_id) == []
    # Detach again = harmless no-op.
    assert wb_manager.detach_world_book_from_character(char_id, "Lore")["detached"] is False


def test_attach_snapshot_carries_matcher_fields_and_enabled(wb_manager):
    db = wb_manager.db
    char_id = _make_character(db, "Mage")
    book_id = wb_manager.create_world_book("Regexy", enabled=False)
    wb_manager.create_world_book_entry(
        book_id, keys=["k"], content="c", regex=True,
        secondary_keys=["s"], priority=7, selective=True,
    )
    wb_manager.attach_world_book_to_character(book_id, char_id)
    record = db.get_character_card_by_id(char_id)
    block = record["extensions"]["character_world_books"][0]
    assert block["enabled"] is False  # augmented from the source book
    entry = block["entries"][0]
    assert entry["regex"] is True and entry["secondary_keys"] == ["s"]
    assert entry["priority"] == 7 and entry["selective"] is True


def test_get_world_books_for_character_dedups_hostile_duplicate_names(wb_manager):
    db = wb_manager.db
    char_id = _make_character(db, "Rogue")
    dup = {"name": "Dupe", "entries": [{"keys": ["a"], "content": "x"}], "enabled": True}
    db.update_character_card(
        char_id,
        {"extensions": {"character_world_books": [dup, dup]}},
        expected_version=db.get_character_card_by_id(char_id)["version"],
    )
    rows = wb_manager.get_world_books_for_character(char_id)
    assert [r["name"] for r in rows] == ["Dupe"]  # deduped, no crash
```

- [ ] **Step 2: Run to verify they fail**

```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/Character_Chat/test_world_book_manager.py -k "character" \
-q -p no:cacheprovider -o addopts="" --timeout=120 --timeout-method=thread
```
Expected: FAIL — `AttributeError: 'WorldBookManager' object has no attribute 'attach_world_book_to_character'`.

- [ ] **Step 3: Add the `_coerce_bool` module helper**

In `world_book_manager.py`, immediately after the existing `_coerce_int` function, add:
```python
def _coerce_bool(value: Any, default: bool) -> bool:
    """Best-effort bool coercion for loosely-typed / imported embedded fields.

    Accepts real bools, and the strings ``"true"/"false"/"1"/"0"/"yes"/"no"``
    (case-insensitive). Anything else falls back to ``default``. Mirrors the
    processor's ``_coerce_bool`` so a hand-edited/imported ``enabled`` never
    misreads (e.g. the string ``"false"`` must not be truthy).
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("true", "1", "yes", "on"):
            return True
        if s in ("false", "0", "no", "off"):
            return False
    return default
```
Also add `Set` to the typing import: change `from typing import List, Dict, Any, Optional` to `from typing import List, Dict, Any, Optional, Set`.

- [ ] **Step 4: Add the character-attach methods**

In `world_book_manager.py`, after `get_conversations_for_world_book` (and before the `# --- Import/Export` section), add:
```python
    # --- Character Association Functions (embedded snapshots) ---

    @staticmethod
    def _normalize_extensions(record: Dict[str, Any]) -> Dict[str, Any]:
        """Return a character record's ``extensions`` as a dict (defensive)."""
        ext = record.get("extensions")
        if isinstance(ext, str):
            try:
                ext = json.loads(ext or "{}")
            except (TypeError, ValueError):
                ext = {}
        if not isinstance(ext, dict):
            ext = {}
        return ext

    @staticmethod
    def _embedded_character_world_books(
        record: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        ext = WorldBookManager._normalize_extensions(record)
        raw = ext.get("character_world_books") or []
        if not isinstance(raw, list):
            raw = []
        return [b for b in raw if isinstance(b, dict) and b.get("name")]

    def _load_character_or_raise(self, character_id: int) -> Dict[str, Any]:
        record = self.db.get_character_card_by_id(int(character_id))
        if record is None:
            raise InputError(f"Character '{character_id}' was not found.")
        return record

    def _write_character_world_books(
        self, record: Dict[str, Any], character_id: int, blocks: List[Dict[str, Any]]
    ) -> None:
        ext = self._normalize_extensions(record)
        ext["character_world_books"] = blocks
        self.db.update_character_card(
            int(character_id),
            {"extensions": ext},
            expected_version=record["version"],
        )

    def attach_world_book_to_character(
        self, world_book_id: int, character_id: int
    ) -> Dict[str, Any]:
        """Embed a world book's content snapshot into a character (idempotent by name).

        The snapshot is ``export_world_book`` output augmented with the source
        book's ``enabled`` (export omits book-level ``enabled``). Names are
        compared as strings so a hostile/imported card whose embedded name is a
        non-str still dedups against the freshly exported (always-str) name.

        Raises:
            InputError: If the world book or the character does not exist.
            ConflictError: If the character's version is stale at write time.
        """
        book = self.get_world_book(int(world_book_id))
        if book is None:
            raise InputError(f"World book {world_book_id} not found")
        block = self.export_world_book(int(world_book_id))
        block["enabled"] = bool(book.get("enabled", True))
        name = block.get("name")
        record = self._load_character_or_raise(character_id)
        blocks = self._embedded_character_world_books(record)
        attached = False
        if not any(str(b.get("name")) == str(name) for b in blocks):
            blocks = blocks + [block]
            self._write_character_world_books(record, character_id, blocks)
            attached = True
        return {
            "world_book_id": int(world_book_id),
            "character_id": int(character_id),
            "name": name,
            "attached": attached,
        }

    def detach_world_book_from_character(
        self, character_id: int, name: str
    ) -> Dict[str, Any]:
        """Remove an embedded world book from a character by name (no-op when absent).

        Raises:
            InputError: If the character does not exist.
            ConflictError: If the character's version is stale at write time.
        """
        record = self._load_character_or_raise(character_id)
        blocks = self._embedded_character_world_books(record)
        detached = False
        if any(str(b.get("name")) == str(name) for b in blocks):
            blocks = [b for b in blocks if str(b.get("name")) != str(name)]
            self._write_character_world_books(record, character_id, blocks)
            detached = True
        return {
            "character_id": int(character_id),
            "name": str(name),
            "detached": detached,
        }

    def get_world_books_for_character(
        self, character_id: int
    ) -> List[Dict[str, Any]]:
        """Summarize a character's embedded world books (from snapshots only).

        Deduped by name (a hostile card can carry two same-named blocks; the
        panel keys DataTable rows by name and would ``DuplicateKey``-crash on a
        dup). Entry counts degrade to 0 for a malformed non-list ``entries``.

        Raises:
            InputError: If the character does not exist.
        """
        record = self._load_character_or_raise(character_id)
        result: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        for b in self._embedded_character_world_books(record):
            name = str(b.get("name"))
            if name in seen:
                continue
            seen.add(name)
            raw_entries = b.get("entries")
            entry_count = len(raw_entries) if isinstance(raw_entries, list) else 0
            result.append(
                {
                    "name": name,
                    "entry_count": entry_count,
                    "enabled": _coerce_bool(b.get("enabled"), True),
                }
            )
        return result
```

- [ ] **Step 5: Run to verify they pass + no regressions**

Run the Step-2 command (PASS, 3 tests), then the whole file:
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/Character_Chat/test_world_book_manager.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: all pass.

- [ ] **Step 6: Commit**
```bash
git add tldw_chatbook/Character_Chat/world_book_manager.py Tests/Character_Chat/test_world_book_manager.py
git commit -m "feat(lore): WorldBookManager character-attach service (embedded snapshots)"
```

---

### Task 2: Pure send-path resolver `resolve_character_world_books`

**Files:**
- Modify: `tldw_chatbook/Character_Chat/world_book_manager.py` (add a module-level function after `_coerce_bool`)
- Test: `Tests/Character_Chat/test_resolve_character_world_books.py` (new)

**Interfaces:**
- Consumes: `_coerce_bool` (Task 1).
- Produces: `resolve_character_world_books(char_data: Optional[Dict[str,Any]], exclude_names: Set[str]) -> List[Dict[str,Any]]` — deduped, enabled, name-excluded snapshot book dicts, ready for `WorldInfoProcessor._process_world_books`.

- [ ] **Step 1: Write the failing tests**

Create `Tests/Character_Chat/test_resolve_character_world_books.py`:
```python
from tldw_chatbook.Character_Chat.world_book_manager import (
    resolve_character_world_books,
)


def _char(blocks):
    return {"extensions": {"character_world_books": blocks}}


def test_returns_enabled_attached_books():
    blocks = [{"name": "A", "enabled": True, "entries": [{"keys": ["x"], "content": "c"}]}]
    out = resolve_character_world_books(_char(blocks), set())
    assert [b["name"] for b in out] == ["A"]


def test_conversation_wins_by_name():
    blocks = [{"name": "Shared", "enabled": True, "entries": []},
              {"name": "Solo", "enabled": True, "entries": []}]
    out = resolve_character_world_books(_char(blocks), {"Shared"})
    assert [b["name"] for b in out] == ["Solo"]


def test_disabled_book_dropped():
    blocks = [{"name": "Off", "enabled": False, "entries": []}]
    assert resolve_character_world_books(_char(blocks), set()) == []


def test_string_false_enabled_is_falsey():
    blocks = [{"name": "Off", "enabled": "false", "entries": []}]
    assert resolve_character_world_books(_char(blocks), set()) == []


def test_dedup_by_name_first_wins():
    blocks = [{"name": "Dup", "enabled": True, "entries": [{"keys": ["1"], "content": "a"}]},
              {"name": "Dup", "enabled": True, "entries": [{"keys": ["2"], "content": "b"}]}]
    out = resolve_character_world_books(_char(blocks), set())
    assert len(out) == 1 and out[0]["entries"][0]["keys"] == ["1"]


def test_malformed_inputs_never_raise():
    assert resolve_character_world_books(None, set()) == []
    assert resolve_character_world_books({}, set()) == []
    assert resolve_character_world_books({"extensions": "not-a-dict"}, set()) == []
    assert resolve_character_world_books({"extensions": {"character_world_books": "x"}}, set()) == []
    assert resolve_character_world_books({"extensions": {"character_world_books": [None, 3, {"no": "name"}]}}, set()) == []
```

- [ ] **Step 2: Run to verify they fail**
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/Character_Chat/test_resolve_character_world_books.py -q -p no:cacheprovider -o addopts="" --timeout=120 --timeout-method=thread
```
Expected: FAIL — `ImportError: cannot import name 'resolve_character_world_books'`.

- [ ] **Step 3: Add the resolver**

In `world_book_manager.py`, add this module-level function directly after `_coerce_bool` (before the `class WorldBookManager`):
```python
def resolve_character_world_books(
    char_data: Optional[Dict[str, Any]],
    exclude_names: Set[str],
) -> List[Dict[str, Any]]:
    """Character-attached world books to apply on the send path.

    Reads snapshot blocks from ``char_data['extensions']['character_world_books']``,
    dedups by name (first wins), drops any whose name is in ``exclude_names``
    (an enabled conversation-attached book already covers it — conversation
    wins) or whose book-level ``enabled`` is false, and returns the survivors
    as ``WorldInfoProcessor._process_world_books``-ready book dicts. Never
    raises on malformed embedded/imported card content.
    """
    if not isinstance(char_data, dict):
        return []
    ext = char_data.get("extensions")
    if isinstance(ext, str):
        try:
            ext = json.loads(ext or "{}")
        except (TypeError, ValueError):
            ext = {}
    if not isinstance(ext, dict):
        return []
    raw = ext.get("character_world_books")
    if not isinstance(raw, list):
        return []
    resolved: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for block in raw:
        if not isinstance(block, dict) or not block.get("name"):
            continue
        name = str(block.get("name"))
        if name in seen or name in exclude_names:
            continue
        seen.add(name)
        if not _coerce_bool(block.get("enabled"), True):
            continue
        resolved.append(block)
    return resolved
```

- [ ] **Step 4: Run to verify they pass**

Run the Step-2 command. Expected: PASS (6 tests).

- [ ] **Step 5: Commit**
```bash
git add tldw_chatbook/Character_Chat/world_book_manager.py Tests/Character_Chat/test_resolve_character_world_books.py
git commit -m "feat(lore): resolve_character_world_books send-path resolver (conversation-wins)"
```

---

### Task 3: Send-path union in `chat_events.py`

**Files:**
- Modify: `tldw_chatbook/Event_Handlers/Chat_Events/chat_events.py` (inside the `enable_world_info` block, before the processor-init guard)
- Test: `Tests/Character_Chat/test_character_world_book_send_path.py` (new)

**Interfaces:**
- Consumes: `resolve_character_world_books` (Task 2), `WorldInfoProcessor`.

- [ ] **Step 1: Write the failing test**

Create `Tests/Character_Chat/test_character_world_book_send_path.py`. This mirrors exactly how `chat_events` builds the processor (conversation `world_books` + the resolver's character books, `character_data` only for a native book), so it proves the union behaves end-to-end without invoking the full send handler:
```python
from tldw_chatbook.Character_Chat.world_book_manager import (
    resolve_character_world_books,
)
from tldw_chatbook.Character_Chat.world_info_processor import WorldInfoProcessor


def _build_processor(conversation_books, char_data, has_native_book):
    # The exact shape of the chat_events union (union BEFORE the init guard).
    world_books = list(conversation_books)
    exclude_names = {str(b.get("name")) for b in world_books}
    world_books = world_books + resolve_character_world_books(char_data, exclude_names)
    if not (has_native_book or world_books):
        return None
    return WorldInfoProcessor(
        character_data=char_data if has_native_book else None,
        world_books=world_books if world_books else None,
    )


def _book(name, key, enabled=True):
    return {"name": name, "enabled": enabled,
            "entries": [{"keys": [key], "content": f"{name} lore", "enabled": True}]}


def test_attached_only_character_fires():
    # No conversation books, no native book: the attached book must still apply.
    char = {"extensions": {"character_world_books": [_book("Attached", "dragon")]}}
    proc = _build_processor([], char, has_native_book=False)
    assert proc is not None
    matched = proc.process_messages([{"role": "user", "content": "a dragon appears"}])
    assert any("Attached lore" in e["content"] for e in matched["matched_entries"])


def test_conversation_wins_same_name():
    conv = [_book("World", "dragon")]
    char = {"extensions": {"character_world_books": [_book("World", "dragon")]}}
    proc = _build_processor(conv, char, has_native_book=False)
    fired = [e for e in proc.process_messages(
        [{"role": "user", "content": "dragon"}])["matched_entries"]]
    assert len(fired) == 1  # the character copy was excluded, only one fires


def test_disabled_attached_book_does_not_fire():
    char = {"extensions": {"character_world_books": [_book("Off", "dragon", enabled=False)]}}
    proc = _build_processor([], char, has_native_book=False)
    assert proc is None  # nothing to process


def test_no_attachment_is_noop():
    assert _build_processor([], {"extensions": {}}, has_native_book=False) is None
```
(Verify `process_messages` returns a dict with `"matched_entries"` and each entry has `"content"`; if the real key names differ, adapt the assertions to the actual return shape — do NOT change production code to fit the test.)

- [ ] **Step 2: Run to verify it fails**
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/Character_Chat/test_character_world_book_send_path.py -q -p no:cacheprovider -o addopts="" --timeout=120 --timeout-method=thread
```
Expected: initially these may PASS at the resolver level (they exercise the resolver + processor directly). The production edit in Step 3 makes the SAME union live in `chat_events`. If any assertion fails, fix the assertion to the real `process_messages` shape first, then proceed — the test's purpose is to pin the union semantics that Step 3 wires into the handler.

- [ ] **Step 3: Wire the union into `chat_events.py`**

Find the `enable_world_info` block (search for `if has_character_book or world_books:`). Insert the union **immediately before** that `if` line (after the `has_character_book = True` detection), at the same indentation as the `has_character_book = False` line:
```python
            # P2f: union character-attached world books (snapshots in
            # extensions['character_world_books']), deduped against enabled
            # conversation books by name (conversation wins). MUST be before the
            # init guard below so an attached-only character still builds a
            # processor.
            from tldw_chatbook.Character_Chat.world_book_manager import (
                resolve_character_world_books,
            )

            character_world_books = resolve_character_world_books(
                active_char_data,
                {str(b.get("name")) for b in world_books},
            )
            if character_world_books:
                world_books = world_books + character_world_books
```
Do not change the `if has_character_book or world_books:` line or the `WorldInfoProcessor(...)` call — with `world_books` now including character books, the existing guard initializes the processor for an attached-only character.

- [ ] **Step 4: Run to verify + import check**

Run the Step-2 command (PASS), then:
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c "import tldw_chatbook.Event_Handlers.Chat_Events.chat_events; print('OK')"
```
Expected: PASS; `OK`.

- [ ] **Step 5: Commit**
```bash
git add tldw_chatbook/Event_Handlers/Chat_Events/chat_events.py Tests/Character_Chat/test_character_world_book_send_path.py
git commit -m "feat(lore): apply character-attached world books on the legacy send path"
```

---

### Task 4: `WorldBookPicker` modal

**Files:**
- Create: `tldw_chatbook/Widgets/Persona_Widgets/world_book_picker.py`
- Test: `Tests/UI/test_world_book_picker.py`

**Interfaces:**
- Produces: `WorldBookPicker(world_books: list[dict[str,Any]])` — a `ModalScreen[int | None]` that dismisses with the picked `world_book_id` (int) or `None`. Rows are `{"world_book_id": int, "name": str}`.

- [ ] **Step 1: Write the failing test**

Create `Tests/UI/test_world_book_picker.py`:
```python
import pytest
from textual.app import App, ComposeResult
from textual.widgets import Input, ListView

from tldw_chatbook.Widgets.Persona_Widgets.world_book_picker import WorldBookPicker


class _Host(App):
    def __init__(self, books):
        super().__init__()
        self._books = books
        self.result = "unset"

    def compose(self) -> ComposeResult:
        yield from ()

    async def on_mount(self) -> None:
        self.run_worker(self._drive)

    async def _drive(self) -> None:
        self.result = await self.push_screen_wait(WorldBookPicker(self._books))


@pytest.mark.asyncio
async def test_pick_returns_int_id():
    books = [{"world_book_id": 10, "name": "Alpha"}, {"world_book_id": 20, "name": "Beta"}]
    app = _Host(books)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.screen.query_one("#worldbook-pick-list", ListView).index = 1
        await pilot.pause()
        await pilot.click("#worldbook-pick-confirm")
        await pilot.pause()
    assert app.result == 20


@pytest.mark.asyncio
async def test_filter_then_select():
    books = [{"world_book_id": 10, "name": "Alpha"}, {"world_book_id": 20, "name": "Beta"}]
    app = _Host(books)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.screen.query_one("#worldbook-pick-search", Input).value = "beta"
        await pilot.pause()
        app.screen.query_one("#worldbook-pick-list", ListView).index = 0
        await pilot.pause()
        await pilot.click("#worldbook-pick-confirm")
        await pilot.pause()
    assert app.result == 20


@pytest.mark.asyncio
async def test_cancel_returns_none():
    app = _Host([{"world_book_id": 10, "name": "Alpha"}])
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.click("#worldbook-pick-cancel")
        await pilot.pause()
    assert app.result is None
```

- [ ] **Step 2: Run to verify it fails**
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/UI/test_world_book_picker.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: FAIL — `ModuleNotFoundError: ...world_book_picker`.

- [ ] **Step 3: Create the picker** (a generic clone of `dictionary_picker.py` with world-book ids/names)

Create `tldw_chatbook/Widgets/Persona_Widgets/world_book_picker.py`:
```python
"""A small modal for picking a world book to attach to a character (Roleplay P2f).

Distinct from ``ConversationAttachPicker`` (which picks a conversation and
returns a string id); this one lists world books and returns the picked int
world_book id.
"""

from __future__ import annotations

from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, ListItem, ListView, Static


class WorldBookPicker(ModalScreen[int | None]):
    """Pick one world book (by int id) to attach to the current character.

    Args:
        world_books: ``{"world_book_id": int, "name": str}`` rows to choose from
            (already filtered to those not yet attached to the character).
    """

    DEFAULT_CSS = """
    WorldBookPicker { align: center middle; }
    WorldBookPicker > Vertical {
        width: 60%; max-width: 80; height: auto; max-height: 80%;
        padding: 1 2; border: round $panel;
    }
    WorldBookPicker #worldbook-pick-list { height: auto; max-height: 16; }
    """

    def __init__(
        self,
        world_books: list[dict[str, Any]],
        *,
        title: str = "Attach world book",
        confirm_label: str = "Attach",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._world_books = list(world_books)
        self._row_ids: list[int] = []
        self._title = title
        self._confirm_label = confirm_label

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self._title, markup=False)
            yield Input(placeholder="Search world books…", id="worldbook-pick-search")
            yield ListView(id="worldbook-pick-list")
            with Vertical(id="worldbook-pick-actions"):
                yield Button(
                    self._confirm_label,
                    id="worldbook-pick-confirm",
                    classes="console-action-secondary",
                )
                yield Button(
                    "Cancel", id="worldbook-pick-cancel", classes="console-action-secondary"
                )

    def on_mount(self) -> None:
        self._populate(self._world_books)

    def _populate(self, rows: list[dict[str, Any]]) -> None:
        listing = self.query_one("#worldbook-pick-list", ListView)
        listing.clear()
        self._row_ids = []
        for row in rows:
            listing.append(
                ListItem(Static(str(row.get("name") or "(unnamed)"), markup=False))
            )
            self._row_ids.append(int(row.get("world_book_id")))
        listing.index = None

    @on(Input.Changed, "#worldbook-pick-search")
    def _filter(self, event: Input.Changed) -> None:
        event.stop()
        needle = event.value.strip().lower()
        rows = (
            [b for b in self._world_books if needle in str(b.get("name") or "").lower()]
            if needle
            else self._world_books
        )
        self._populate(rows)

    def _selected_id(self) -> int | None:
        listing = self.query_one("#worldbook-pick-list", ListView)
        index = listing.index
        if index is None or not 0 <= index < len(self._row_ids):
            return None
        return self._row_ids[index]

    @on(Button.Pressed, "#worldbook-pick-confirm")
    def _confirm(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(self._selected_id())

    @on(Button.Pressed, "#worldbook-pick-cancel")
    def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)


__all__ = ["WorldBookPicker"]
```

- [ ] **Step 4: Run to verify it passes**

Run the Step-2 command. Expected: PASS (3 tests).

- [ ] **Step 5: Commit**
```bash
git add tldw_chatbook/Widgets/Persona_Widgets/world_book_picker.py Tests/UI/test_world_book_picker.py
git commit -m "feat(personas): WorldBookPicker modal for character attach"
```

---

### Task 5: `PersonasCharacterWorldBooksWidget` + messages

**Files:**
- Create: `tldw_chatbook/Widgets/Persona_Widgets/personas_character_world_books.py`
- Test: `Tests/UI/test_personas_character_world_books.py`

**Interfaces:**
- Produces: `PersonasCharacterWorldBooksWidget` (I/O-free; `load_world_books(rows: list[dict])`), messages `CharacterWorldBookAttachRequested` (bare) and `CharacterWorldBookDetachRequested(name: str)`; ids `#personas-char-worldbooks-table`, `#personas-char-worldbooks-empty`, `#personas-char-worldbooks-add`, `#personas-char-worldbooks-detach`; widget id `#personas-character-world-books`.

- [ ] **Step 1: Write the failing tests**

Create `Tests/UI/test_personas_character_world_books.py`:
```python
import pytest
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Static

from tldw_chatbook.Widgets.Persona_Widgets.personas_character_world_books import (
    PersonasCharacterWorldBooksWidget,
    CharacterWorldBookAttachRequested,
    CharacterWorldBookDetachRequested,
)


class _Host(App):
    def __init__(self):
        super().__init__()
        self.attach_posts = []
        self.detach_posts = []

    def compose(self) -> ComposeResult:
        yield PersonasCharacterWorldBooksWidget()

    def on_character_world_book_attach_requested(self, message) -> None:
        self.attach_posts.append(message)

    def on_character_world_book_detach_requested(self, message) -> None:
        self.detach_posts.append(message.name)


@pytest.mark.asyncio
async def test_empty_then_render():
    app = _Host()
    async with app.run_test(size=(140, 40)) as pilot:
        w = app.query_one(PersonasCharacterWorldBooksWidget)
        w.load_world_books([])
        await pilot.pause()
        assert app.query_one("#personas-char-worldbooks-empty", Static).display is True
        assert app.query_one("#personas-char-worldbooks-table", DataTable).row_count == 0
        w.load_world_books([{"name": "Lore", "entry_count": 3, "enabled": True}])
        await pilot.pause()
        assert app.query_one("#personas-char-worldbooks-empty", Static).display is False
        assert app.query_one("#personas-char-worldbooks-table", DataTable).row_count == 1


@pytest.mark.asyncio
async def test_duplicate_names_do_not_crash():
    app = _Host()
    async with app.run_test(size=(140, 40)) as pilot:
        w = app.query_one(PersonasCharacterWorldBooksWidget)
        dup = {"name": "Dup", "entry_count": 1, "enabled": True}
        w.load_world_books([dup, dup])  # would DuplicateKey without the guard
        await pilot.pause()
        assert app.query_one("#personas-char-worldbooks-table", DataTable).row_count == 1


@pytest.mark.asyncio
async def test_attach_button_posts():
    app = _Host()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.click("#personas-char-worldbooks-add")
        await pilot.pause()
        assert len(app.attach_posts) == 1


@pytest.mark.asyncio
async def test_detach_posts_selected_name():
    app = _Host()
    async with app.run_test(size=(140, 40)) as pilot:
        w = app.query_one(PersonasCharacterWorldBooksWidget)
        w.load_world_books([{"name": "Lore", "entry_count": 1, "enabled": True}])
        await pilot.pause()
        app.query_one("#personas-char-worldbooks-table", DataTable).move_cursor(row=0)
        await pilot.pause()
        await pilot.click("#personas-char-worldbooks-detach")
        await pilot.pause()
        assert app.detach_posts == ["Lore"]
```

- [ ] **Step 2: Run to verify they fail**
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/UI/test_personas_character_world_books.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Create the widget** (a clone of `personas_character_dictionaries.py` with world-book naming — preserve the `_attach_pressed`/`_detach_pressed` naming and the dedup-before-`add_row` guard verbatim)

Create `tldw_chatbook/Widgets/Persona_Widgets/personas_character_world_books.py`:
```python
"""Roleplay P2f: an I/O-free panel listing a character's embedded world books.

The panel renders what the screen feeds via ``load_world_books`` and posts
intent messages; the screen owns all service/DB work. Each embedded world book
is a snapshot (an embedded copy — editing the source book does not update it).
"""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.message import Message
from textual.widgets import Button, DataTable, Static


class CharacterWorldBookAttachRequested(Message):
    """Request the attach-world-book picker for the current character."""


class CharacterWorldBookDetachRequested(Message):
    """Detach one embedded world book from the current character.

    Args:
        name: The embedded world book to remove (by name).
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name


class PersonasCharacterWorldBooksWidget(Container):
    """List + attach/detach a character's embedded world books (snapshots)."""

    DEFAULT_CSS = """
    PersonasCharacterWorldBooksWidget #personas-char-worldbooks-table { height: auto; max-height: 8; }
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("id", "personas-character-world-books")
        super().__init__(**kwargs)
        self._rows: list[dict[str, Any]] = []

    def compose(self) -> ComposeResult:
        yield Static("World Books (embedded copies)", classes="destination-section")
        yield Static(
            "No world books attached to this character yet.",
            id="personas-char-worldbooks-empty",
            markup=False,
        )
        yield DataTable(id="personas-char-worldbooks-table", cursor_type="row")
        with Horizontal(classes="personas-dict-form-row"):
            yield Button(
                "Attach world book…",
                id="personas-char-worldbooks-add",
                classes="console-action-secondary",
            )
            yield Button(
                "Detach",
                id="personas-char-worldbooks-detach",
                classes="console-action-secondary",
            )

    def on_mount(self) -> None:
        self.query_one("#personas-char-worldbooks-table", DataTable).add_columns(
            "world book", "entries"
        )
        self.load_world_books([])

    def load_world_books(self, rows: list[dict[str, Any]]) -> None:
        """Render the character's embedded world books.

        Args:
            rows: ``{"name": str, "entry_count": int, "enabled": bool}`` entries.

        Dedup by name (first wins) before touching the table: ``DataTable`` keys
        rows by ``str(name)``, so a hostile/imported card with two same-named
        embedded blocks would otherwise raise ``DuplicateKey`` — which would
        propagate uncaught through the import worker and exit the app.
        """
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            name = str(row.get("name"))
            if name in seen:
                continue
            seen.add(name)
            deduped.append(row)
        self._rows = deduped
        table = self.query_one("#personas-char-worldbooks-table", DataTable)
        table.clear()
        for row in self._rows:
            table.add_row(
                Text(str(row.get("name") or "(unnamed)")),
                Text(
                    str(
                        row.get("entry_count")
                        if row.get("entry_count") is not None
                        else ""
                    )
                ),
                key=str(row.get("name")),
            )
        empty = self.query_one("#personas-char-worldbooks-empty", Static)
        empty.display = not self._rows
        table.display = bool(self._rows)

    def _selected_name(self) -> str | None:
        table = self.query_one("#personas-char-worldbooks-table", DataTable)
        if table.row_count == 0 or table.cursor_row is None or table.cursor_row < 0:
            return None
        try:
            return str(
                table.coordinate_to_cell_key((table.cursor_row, 0)).row_key.value
            )
        except Exception:
            return None

    @on(Button.Pressed, "#personas-char-worldbooks-add")
    def _attach_pressed(self, event: Button.Pressed) -> None:
        # `_attach_pressed` (not `_attach`) avoids shadowing DOMNode._attach.
        event.stop()
        self.post_message(CharacterWorldBookAttachRequested())

    @on(Button.Pressed, "#personas-char-worldbooks-detach")
    def _detach_pressed(self, event: Button.Pressed) -> None:
        # `_detach_pressed` (not `_detach`) avoids shadowing DOMNode._detach.
        event.stop()
        name = self._selected_name()
        if name is not None:
            self.post_message(CharacterWorldBookDetachRequested(name))


__all__ = [
    "PersonasCharacterWorldBooksWidget",
    "CharacterWorldBookAttachRequested",
    "CharacterWorldBookDetachRequested",
]
```

- [ ] **Step 4: Run to verify they pass**

Run the Step-2 command. Expected: PASS (4 tests).

- [ ] **Step 5: Commit**
```bash
git add tldw_chatbook/Widgets/Persona_Widgets/personas_character_world_books.py Tests/UI/test_personas_character_world_books.py
git commit -m "feat(personas): I/O-free character world-books panel + messages"
```

---

### Task 6: Screen display wiring (mount, CSS, editor sync, refresh on selection)

**Files:**
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_character_editor_widget.py`
- Test: `Tests/UI/test_personas_character_world_books_screen.py` (new)

**Interfaces:**
- Consumes: `PersonasCharacterWorldBooksWidget` (Task 5), `WorldBookManager.get_world_books_for_character` (Task 1), the existing `self._lore_manager()`.
- Produces: `PersonasCharacterEditorWidget.sync_attached_world_books(blocks, new_version)`; `PersonasScreen._refresh_character_worldbooks()`.

- [ ] **Step 1: Write the failing test** (real-DB: a character with a snapshot already embedded shows in the panel on selection)

Create `Tests/UI/test_personas_character_world_books_screen.py`. Read `Tests/UI/test_personas_character_attach.py` for the character-selection harness (`PersonasTestApp`, `stub_characters`, selecting a character) and `Tests/UI/test_personas_lore.py` for the real-`chachanotes_db` pattern; build a harness that (a) uses a REAL `CharactersRAGDB` as `app_instance.chachanotes_db`, (b) has one character with `extensions['character_world_books']` pre-seeded via `WorldBookManager.attach_world_book_to_character`, (c) selects that character, and (d) asserts `PersonasCharacterWorldBooksWidget` shows one row. Skeleton:
```python
# Adapt fixtures/harness to the real ones in the two referenced test files.
import pytest
from textual.widgets import DataTable

pytestmark = pytest.mark.asyncio


async def test_selecting_character_shows_attached_world_books(...):
    # 1. real db; create a character (db.add_character_card({"name": "Hero"}))
    # 2. book = wb_manager.create_world_book("Lore"); add an entry;
    #    wb_manager.attach_world_book_to_character(book, char_id)
    # 3. mount the screen with app_instance.chachanotes_db = db
    # 4. select the character; await pilot.pause()
    # 5. panel = screen.query_one(PersonasCharacterWorldBooksWidget)
    #    assert panel.query_one("#personas-char-worldbooks-table", DataTable).row_count == 1
    ...


async def test_editor_sync_patches_base_without_conflict(...):
    # After sync_attached_world_books(blocks, new_version), the editor's
    # get_character_data()["extensions"]["character_world_books"] == blocks and
    # ["version"] == new_version (no clobber of in-progress edits).
    ...
```

- [ ] **Step 2: Run to verify they fail**
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/UI/test_personas_character_world_books_screen.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: FAIL (panel not mounted / method missing).

- [ ] **Step 3: Add `sync_attached_world_books` to the editor**

In `personas_character_editor_widget.py`, immediately after `sync_attached_dictionaries`, add (identical shape, `character_world_books` key):
```python
    def sync_attached_world_books(
        self, character_world_books: list, new_version: int
    ) -> None:
        """Patch the loaded base after an out-of-band world-book attach/detach.

        Updates only ``extensions['character_world_books']`` and ``version`` on
        the base copy the Save path starts from, so an instant attach is neither
        clobbered by a later Save nor forces a version conflict. No-op when no
        character is loaded (empty base).
        """
        if not self._character_data:
            return
        ext = self._character_data.get("extensions")
        if not isinstance(ext, dict):
            ext = {}
        ext["character_world_books"] = list(character_world_books)
        self._character_data["extensions"] = ext
        self._character_data["version"] = new_version
```

- [ ] **Step 4: Import + mount + CSS in `personas_screen.py`**

(a) Add to the `personas_character_world_books` import group (near the `PersonasCharacterDictionariesWidget` import at ~:48):
```python
from ...Widgets.Persona_Widgets.personas_character_world_books import (
    PersonasCharacterWorldBooksWidget,
    CharacterWorldBookAttachRequested,
    CharacterWorldBookDetachRequested,
)
from ...Widgets.Persona_Widgets.world_book_picker import WorldBookPicker
```
(b) Mount it in the detail stack — after `yield PersonasCharacterDictionariesWidget()`:
```python
                        yield PersonasCharacterWorldBooksWidget()
```
(c) Add a docked-bottom CSS rule after the `PersonasCharacterDictionariesWidget` CSS block (mirror it; keep both `max-height` values so the two stacked panels + card/editor fit — see the geometry check):
```css
    #personas-detail-stack PersonasCharacterWorldBooksWidget {
        dock: bottom;
        height: auto;
        max-height: 10;
        width: 100%;
    }
```

- [ ] **Step 5: Add `_refresh_character_worldbooks` + call it on character selection**

In `personas_screen.py`, after `_refresh_character_dictionaries`, add:
```python
    async def _refresh_character_worldbooks(self) -> None:
        """Re-feed the character world-books panel (best-effort)."""
        entity_id = self.state.selected_entity_id
        if self.state.selected_entity_kind != "character" or not entity_id:
            return
        manager = self._lore_manager()
        if manager is None:
            return
        try:
            rows = await asyncio.to_thread(
                manager.get_world_books_for_character, int(entity_id)
            )
        except Exception:
            logger.opt(exception=True).warning(
                f"Could not list world books for character {entity_id}."
            )
            rows = []
        if self.state.selected_entity_id != entity_id or self.state.selected_entity_kind != "character":
            return
        try:
            self.query_one(PersonasCharacterWorldBooksWidget).load_world_books(rows)
        except QueryError:
            pass
```
Then, in `_select_character_entry`, immediately after the existing `await self._refresh_character_dictionaries()` line, add:
```python
        await self._refresh_character_worldbooks()
```

- [ ] **Step 6: Geometry check** (add to the new screen test file)

Add a test that mounts the screen, selects a character, and asserts neither the card nor the world-books panel is clipped to zero and both docked panels fit:
```python
async def test_two_docked_panels_do_not_clip_card(...):
    # At size (100, 30) AND (160, 50): after selecting a character,
    #   card = screen.query_one(PersonasCharacterCardWidget)
    #   wb = screen.query_one(PersonasCharacterWorldBooksWidget)
    #   dicts = screen.query_one(PersonasCharacterDictionariesWidget)
    # assert card.size.height > 0 and wb.size.height > 0 and dicts.size.height > 0
    # (i.e. stacking the two bottom-docked panels did not squeeze the card to nothing)
    ...
```
If the card collapses at the smaller size, reduce the world-books panel `max-height` (Step 4c) until all three have height > 0, and note the final values in the report.

- [ ] **Step 7: Run to verify they pass + no regressions**

```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/UI/test_personas_character_world_books_screen.py Tests/UI/test_personas_character_attach.py \
-q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: all pass (the existing character-dict tests confirm no layout regression).

- [ ] **Step 8: Commit**
```bash
git add tldw_chatbook/UI/Screens/personas_screen.py tldw_chatbook/Widgets/Persona_Widgets/personas_character_editor_widget.py Tests/UI/test_personas_character_world_books_screen.py
git commit -m "feat(personas): mount character world-books panel + editor sync + refresh"
```

---

### Task 7: Screen attach/detach handlers + real-DB round-trip

**Files:**
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Test: `Tests/UI/test_personas_character_world_books_screen.py` (extend Task 6's file)

**Interfaces:**
- Consumes: `CharacterWorldBookAttachRequested`/`CharacterWorldBookDetachRequested` (Task 5), `WorldBookPicker` (Task 4), `WorldBookManager.attach_world_book_to_character`/`detach_world_book_from_character`/`list_world_books` (Task 1), `_refresh_character_worldbooks` + `sync_attached_world_books` (Task 6).

- [ ] **Step 1: Write the failing real-DB tests**

Extend `Tests/UI/test_personas_character_world_books_screen.py` (mirror the P1f `_character_dictionary_attach_worker` flow but with the REAL `WorldBookManager` on the real db — no fake service):
```python
async def test_attach_via_picker_then_detach_real_db(...):
    # real db + a selectable character + a standalone world book (with an entry)
    # monkeypatch screen.app.push_screen_wait to return the book_id for a WorldBookPicker
    # monkeypatch screen._list_attachable_world_books -> [{"world_book_id": book_id, "name": "Lore"}]
    # post CharacterWorldBookAttachRequested(); wait_for_complete
    #   assert wb_manager.get_world_books_for_character(char_id) has "Lore"
    #   assert the panel table row_count == 1
    # post CharacterWorldBookDetachRequested("Lore"); wait_for_complete
    #   assert get_world_books_for_character(char_id) == []
    ...
```

- [ ] **Step 2: Run to verify they fail**

Same command as Task 6 Step 2 (new tests error: no `@on(CharacterWorldBookAttachRequested)` handler → nothing persists).

- [ ] **Step 3: Add the handlers + `_list_attachable_world_books` + `_sync_character_editor_worldbooks`**

In `personas_screen.py`, near the character-dictionary handlers, add:
```python
    @on(CharacterWorldBookAttachRequested)
    async def _handle_character_worldbook_attach(
        self, message: CharacterWorldBookAttachRequested
    ) -> None:
        message.stop()
        if (
            self.state.selected_entity_kind != "character"
            or not self.state.selected_entity_id
        ):
            return
        if self._io_dialog_active:
            return
        self._io_dialog_active = True
        self.run_worker(self._character_worldbook_attach_worker(), group="personas-io")

    async def _character_worldbook_attach_worker(self) -> None:
        try:
            entity_id = self.state.selected_entity_id
            manager = self._lore_manager()
            if manager is None or not entity_id:
                return
            char_id = int(entity_id)
            try:
                books = await asyncio.to_thread(
                    self._list_attachable_world_books, char_id
                )
            except Exception:
                logger.opt(exception=True).warning(
                    "Could not load world books for the attach picker."
                )
                self._notify("Attach failed: could not list world books.", "error")
                return
            try:
                picked = await self.app.push_screen_wait(WorldBookPicker(books))
            except Exception:
                logger.opt(exception=True).warning(
                    "Could not show the world-book picker."
                )
                return
            if not picked:
                return
            try:
                await asyncio.to_thread(
                    manager.attach_world_book_to_character, int(picked), char_id
                )
            except ConflictError:
                self._notify(
                    "Attach failed: the character changed since it was loaded. Try again.",
                    "warning",
                )
                return
            except Exception as exc:
                logger.opt(exception=True).warning(
                    f"Could not attach world book to character {char_id}."
                )
                self._notify(f"Attach failed: {exc}", "error")
                return
            await self._refresh_character_worldbooks()
            await self._sync_character_editor_worldbooks(char_id)
            self._notify("Attached to character.", "information")
        finally:
            self._io_dialog_active = False

    def _list_attachable_world_books(self, character_id: int) -> list[dict]:
        """Standalone world books NOT already attached to this character (sync DB read)."""
        manager = self._lore_manager()
        if manager is None:
            return []
        attached = {
            str(r.get("name"))
            for r in manager.get_world_books_for_character(int(character_id))
        }
        rows = []
        for b in manager.list_world_books(include_disabled=False) or []:
            name = b.get("name")
            if str(name) in attached:
                continue
            rows.append({"world_book_id": int(b.get("id")), "name": str(name)})
        return rows

    async def _sync_character_editor_worldbooks(self, character_id: int) -> None:
        """Keep the editor's base coherent after an out-of-band attach/detach."""
        db = getattr(self.app_instance, "chachanotes_db", None)
        if db is None:
            return
        try:
            record = await asyncio.to_thread(
                db.get_character_card_by_id, int(character_id)
            )
        except Exception:
            return
        if not record:
            return
        ext = (
            record.get("extensions")
            if isinstance(record.get("extensions"), dict)
            else {}
        )
        try:
            editor = self.query_one(PersonasCharacterEditorWidget)
        except Exception:
            return
        if int(editor._character_data.get("id") or 0) == int(character_id):
            editor.sync_attached_world_books(
                ext.get("character_world_books") or [], record.get("version")
            )

    @on(CharacterWorldBookDetachRequested)
    async def _handle_character_worldbook_detach(
        self, message: CharacterWorldBookDetachRequested
    ) -> None:
        message.stop()
        entity_id = self.state.selected_entity_id
        if (
            self.state.selected_entity_kind != "character"
            or not entity_id
        ):
            return
        manager = self._lore_manager()
        if manager is None:
            return
        char_id = int(entity_id)
        try:
            await asyncio.to_thread(
                manager.detach_world_book_from_character, char_id, str(message.name)
            )
        except ConflictError:
            self._notify(
                "Detach failed: the character changed since it was loaded. Try again.",
                "warning",
            )
            return
        except Exception:
            logger.opt(exception=True).warning(
                f"Could not detach world book from character {char_id}."
            )
            return
        await self._refresh_character_worldbooks()
        await self._sync_character_editor_worldbooks(char_id)
        self._notify("Detached from character.", "information")
```
Confirm `ConflictError` is imported in `personas_screen.py` (it is used by the dict handlers); if not, add it to the existing DB imports.

- [ ] **Step 4: Run to verify they pass + full P2f gate + app import**
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/Character_Chat/test_world_book_manager.py Tests/Character_Chat/test_resolve_character_world_books.py \
Tests/Character_Chat/test_character_world_book_send_path.py Tests/UI/test_world_book_picker.py \
Tests/UI/test_personas_character_world_books.py Tests/UI/test_personas_character_world_books_screen.py \
-q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c "import tldw_chatbook.app; print('APP IMPORT OK')"
```
Expected: all pass; `APP IMPORT OK`.

- [ ] **Step 5: Commit**
```bash
git add tldw_chatbook/UI/Screens/personas_screen.py Tests/UI/test_personas_character_world_books_screen.py
git commit -m "feat(personas): wire character world-book attach/detach to WorldBookManager"
```

---

## Notes for reviewers

- **No migration, no send-path behavior change for the no-attachment case:** verify the only `chat_events` change is the union block, placed BEFORE the `if has_character_book or world_books:` guard; a character with no `character_world_books` key is byte-identical to today.
- **Conversation-wins dedup** happens in `resolve_character_world_books` via `exclude_names` (the enabled conversation-book names); the native `character_book` is deliberately NOT deduped (stability).
- **Crash-class:** dedup-by-name at BOTH `get_world_books_for_character` (service) and `load_world_books` (widget); `str(...)` name comparisons throughout; `_coerce_bool` for embedded `enabled`. A hostile card with two same-named blocks must not `DuplicateKey`-crash the panel.
- **`WorldInfoProcessor`, `DictionaryPicker`, `PersonasCharacterDictionariesWidget`, and the dict service are untouched.**
- **Console what's-in-play + native-Console send + legacy-UI retirement are P2g.**
