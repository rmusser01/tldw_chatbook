# Roleplay P2d-2 — world-book import/export UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user import a world book from a `.json` file and export the selected world book to a file in the Roleplay Lore mode, wired to the existing `WorldBookManager.export_world_book`/`import_world_book`.

**Architecture:** A new pure module `world_book_import.py` maps any supported input shape (tldw export, character-book array, SillyTavern World Info object-form) to tldw's field names and validates the whole file up front (so `import_world_book`'s non-atomic loop never gets bad data → no partial book). The screen adds an import path-handler + a file-picker worker (mirroring the dictionary import flow) and an export worker (mirroring the character `EnhancedFileSave` flow); the lore detail widget gains an Export button + typed message.

**Tech Stack:** Python ≥3.11, Textual, pytest + pytest-asyncio, SQLite (`CharactersRAGDB`), `WorldBookManager`.

## Global Constraints

- NO change to `world_book_manager.py` (manager) or `ChaChaNotes_DB.py` (schema stays `_CURRENT_SCHEMA_VERSION = 21`).
- `normalize_world_book_import` validates the ENTIRE file before any DB write — a bad file yields NO partial book.
- Import and export must NEVER crash the UI: all I/O and parsing wrapped, errors surfaced via `self._notify(msg, level)`.
- I/O workers run via `self.run_worker(..., group="personas-io")` guarded by `self._io_dialog_active` (set True before, reset in `finally`).
- After an import/export worker finishes, if `self.state.active_mode != "lore"` the user navigated away — do NOT yank them back (mirror the dictionary/character import guards).
- JSON only (no Markdown export).
- Implementers stage ONLY their task's files (`git add <explicit paths>`; never `git add -A`; never `.superpowers/`).
- **Test environment (run from the worktree root):**
  ```bash
  HOME=/private/tmp/tldw-chatbook-test-home \
  XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <targets> \
  -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
  ```
  The venv is in the MAIN checkout; `import tldw_chatbook` resolves to the worktree source (cwd on `sys.path`).

## File Structure

- **Create** `tldw_chatbook/Character_Chat/world_book_import.py` — pure, DB-free normalization/validation. One responsibility: turn imported JSON into a validated tldw-shaped dict or raise `ValueError`.
- **Modify** `tldw_chatbook/Widgets/Persona_Widgets/personas_lore_detail.py` — add `LoreBookExportRequested` message + Export button + handler + `__all__`.
- **Modify** `tldw_chatbook/Widgets/Persona_Widgets/personas_library_pane.py` — un-gate the Import button for `lore` in `set_mode`.
- **Modify** `tldw_chatbook/UI/Screens/personas_screen.py` — import branch + import dialog/worker/path-handler; export handler/worker; the `WORLDBOOK` size constant; the `normalize_world_book_import` import.
- **Create** `Tests/Character_Chat/test_world_book_import.py` — pure adapter unit tests.
- **Modify** `Tests/UI/test_personas_lore.py` — real-DB import/export + round-trip tests.

---

### Task 1: `world_book_import.py` normalization adapter

**Files:**
- Create: `tldw_chatbook/Character_Chat/world_book_import.py`
- Test: `Tests/Character_Chat/test_world_book_import.py`

**Interfaces:**
- Produces: `normalize_world_book_import(data: Any) -> dict` — returns `{**data, "entries": [<normalized entry dict>...]}` where each entry has exactly `{keys, secondary_keys, content, insertion_order, position, selective, case_sensitive, enabled, priority, extensions}`; raises `ValueError` (user-facing message) on any invalid shape / empty-keys / empty-content entry.

- [ ] **Step 1: Write the failing unit tests**

Create `Tests/Character_Chat/test_world_book_import.py`:

```python
import pytest

from tldw_chatbook.Character_Chat.world_book_import import normalize_world_book_import


def _tldw_entry(**kw):
    base = {"keys": ["Warden"], "content": "grim jailer", "position": "before_char",
            "insertion_order": 0, "selective": False, "secondary_keys": [],
            "case_sensitive": False, "enabled": True, "priority": 0}
    base.update(kw)
    return base


def test_tldw_export_passthrough():
    data = {"name": "B", "description": "d", "scan_depth": 3, "token_budget": 500,
            "recursive_scanning": False, "entries": [_tldw_entry(priority=42)]}
    out = normalize_world_book_import(data)
    assert out["name"] == "B" and out["scan_depth"] == 3
    e = out["entries"][0]
    assert e["keys"] == ["Warden"] and e["content"] == "grim jailer"
    assert e["priority"] == 42 and e["position"] == "before_char"


def test_character_book_array_passthrough():
    data = {"entries": [{"keys": ["a", "b"], "content": "c", "secondary_keys": ["s"],
                         "insertion_order": 2, "selective": True, "case_sensitive": True}]}
    e = normalize_world_book_import(data)["entries"][0]
    assert e["keys"] == ["a", "b"] and e["secondary_keys"] == ["s"]
    assert e["insertion_order"] == 2 and e["selective"] is True and e["case_sensitive"] is True


def test_sillytavern_world_info_object_form_remaps():
    data = {"entries": {"0": {"key": ["Warden"], "keysecondary": ["jail"], "content": "x",
                              "order": 5, "position": 1, "disable": True,
                              "caseSensitive": True, "selective": True}}}
    e = normalize_world_book_import(data)["entries"][0]
    assert e["keys"] == ["Warden"] and e["secondary_keys"] == ["jail"]
    assert e["insertion_order"] == 5 and e["position"] == "after_char"
    assert e["enabled"] is False and e["case_sensitive"] is True and e["selective"] is True


def test_missing_entries_yields_empty_list():
    assert normalize_world_book_import({"name": "B"})["entries"] == []


def test_priority_and_extensions_preserved_and_defaulted():
    data = {"entries": [{"keys": ["k"], "content": "c", "extensions": {"x": 1}},
                        {"keys": ["k2"], "content": "c2"}]}
    out = normalize_world_book_import(data)["entries"]
    assert out[0]["extensions"] == {"x": 1} and out[0]["priority"] == 0
    assert out[1]["extensions"] == {} and out[1]["insertion_order"] == 1


def test_non_dict_top_level_raises():
    with pytest.raises(ValueError, match="must be a JSON object"):
        normalize_world_book_import([1, 2, 3])


def test_entries_not_list_or_dict_raises():
    with pytest.raises(ValueError, match="must be a list or an object"):
        normalize_world_book_import({"entries": 42})


def test_empty_keys_entry_raises():
    with pytest.raises(ValueError, match="Entry 1 has no keys"):
        normalize_world_book_import({"entries": [{"keys": [], "content": "c"}]})


def test_empty_content_entry_raises():
    with pytest.raises(ValueError, match="Entry 1 has no content"):
        normalize_world_book_import({"entries": [{"keys": ["k"], "content": "   "}]})


def test_non_dict_entry_raises():
    with pytest.raises(ValueError, match="Entry 1 is not an object"):
        normalize_world_book_import({"entries": ["not a dict"]})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/Character_Chat/test_world_book_import.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: FAIL — `ModuleNotFoundError: No module named 'tldw_chatbook.Character_Chat.world_book_import'`.

- [ ] **Step 3: Write the module**

Create `tldw_chatbook/Character_Chat/world_book_import.py`:

```python
"""Pure normalization/validation for world-book (Lore) import files.

Maps the supported input shapes — tldw's own export, the character-book array
form, and SillyTavern 'World Info' object-form — to tldw's ``world_book_entries``
field names, and validates the whole file up front so the import screen can
reject a bad file before any DB write (``WorldBookManager.import_world_book`` is
not atomic). Pure and DB-free; raises ``ValueError`` with a user-facing message.
"""
from __future__ import annotations

from typing import Any, Dict, List

_VALID_POSITIONS = {"before_char", "after_char", "at_start", "at_end"}
_INT_POSITION_MAP = {0: "before_char", 1: "after_char"}


def _coerce_int(value: Any, default: int) -> int:
    """Best-effort int coercion; ``default`` on None/non-numeric."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_str_list(value: Any) -> List[str]:
    """Coerce a keys-like field to a list of non-blank strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    return []


def _normalize_position(pos: Any) -> str:
    """Map a position (tldw string or SillyTavern int) to a tldw position."""
    if isinstance(pos, str) and pos in _VALID_POSITIONS:
        return pos
    if isinstance(pos, bool):
        return "before_char"
    if isinstance(pos, int):
        return _INT_POSITION_MAP.get(pos, "before_char")
    return "before_char"


def _normalize_entry(entry: Any, index: int) -> Dict[str, Any]:
    """Map one entry to tldw fields; raise ValueError (1-based index) if invalid."""
    if not isinstance(entry, dict):
        raise ValueError(f"Entry {index + 1} is not an object.")
    raw_keys = entry.get("keys")
    if raw_keys is None:
        raw_keys = entry.get("key")
    keys = _as_str_list(raw_keys)
    if not keys:
        raise ValueError(f"Entry {index + 1} has no keys.")
    content = str(entry.get("content", ""))
    if not content.strip():
        raise ValueError(f"Entry {index + 1} has no content.")
    raw_secondary = entry.get("secondary_keys")
    if raw_secondary is None:
        raw_secondary = entry.get("keysecondary")
    extensions = entry.get("extensions")
    return {
        "keys": keys,
        "secondary_keys": _as_str_list(raw_secondary),
        "content": content,
        "insertion_order": _coerce_int(
            entry.get("insertion_order", entry.get("order", index)), index
        ),
        "position": _normalize_position(entry.get("position")),
        "selective": bool(entry.get("selective", False)),
        "case_sensitive": bool(entry.get("case_sensitive", entry.get("caseSensitive", False))),
        "enabled": bool(entry.get("enabled", not entry.get("disable", False))),
        "priority": _coerce_int(entry.get("priority", 0), 0),
        "extensions": extensions if isinstance(extensions, dict) else {},
    }


def normalize_world_book_import(data: Any) -> Dict[str, Any]:
    """Normalize + validate an imported world-book payload.

    Args:
        data: The parsed JSON from an import file.

    Returns:
        A dict with tldw metadata keys preserved and ``entries`` mapped to
        tldw's field names as a list, ready for ``import_world_book``.

    Raises:
        ValueError: If the payload is not a dict, ``entries`` is neither a list
            nor an object, or any entry is not a dict / has no keys / no content.
    """
    if not isinstance(data, dict):
        raise ValueError("World book file must be a JSON object.")
    raw_entries = data.get("entries")
    if raw_entries is None:
        entries_list: List[Any] = []
    elif isinstance(raw_entries, dict):
        entries_list = list(raw_entries.values())
    elif isinstance(raw_entries, list):
        entries_list = raw_entries
    else:
        raise ValueError("'entries' must be a list or an object.")
    normalized = [_normalize_entry(entry, i) for i, entry in enumerate(entries_list)]
    return {**data, "entries": normalized}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run the same command as Step 2. Expected: PASS (11 tests).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Character_Chat/world_book_import.py Tests/Character_Chat/test_world_book_import.py
git commit -m "feat(lore): pure world-book import normalizer (tldw/character-book/SillyTavern)"
```

---

### Task 2: Export flow

**Files:**
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_lore_detail.py` (message class near `:63`; Export button after `:151`; handler near `:456`; `__all__` at `:477`)
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py` (import block `:90-98`; new handler + worker near `_handle_lore_settings_save` `:2037`)
- Test: `Tests/UI/test_personas_lore.py`

**Interfaces:**
- Consumes: `WorldBookManager.export_world_book(id) -> dict` (already exists); `EnhancedFileSave`/`Filters` from `Widgets/enhanced_file_picker.py`.
- Produces: message class `LoreBookExportRequested` (no payload); screen handler `_handle_lore_export`; worker `_lore_export_worker`; button id `#personas-lore-export`.

- [ ] **Step 1: Write the failing export test**

Append to `Tests/UI/test_personas_lore.py` (after the last real-DB test). It monkeypatches the save-picker to return a temp path. `Switch` etc. are already imported; add `import json` at the top of the file if not present (check first).

```python
@pytest.mark.asyncio
async def test_export_selected_lore_book_writes_json_file(
    mock_app_instance, stub_characters_lore, lore_db, seeded_lore_book, tmp_path, monkeypatch
):
    """Exporting the selected lore book writes a JSON file that parses back to the
    book's export payload (name + entries)."""
    import json as _json
    mock_app_instance.chachanotes_db = lore_db
    app = LorePersonasTestApp(mock_app_instance)
    async with app.run_test(size=(200, 60)) as pilot:
        screen = await _enter_lore(pilot)
        await _select_first_lore(pilot, screen)
        target = tmp_path / "exported.json"

        async def _fake_save(picker):
            return target

        monkeypatch.setattr(app, "push_screen_wait", _fake_save)
        screen.post_message(LoreBookExportRequested())
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        assert target.exists()
        payload = _json.loads(target.read_text("utf-8"))
        assert payload["name"] == "Blackreach"
        assert any(e["keys"] == ["Warden"] for e in payload["entries"])
```

Add `LoreBookExportRequested` to the test file's import from `personas_lore_detail` (the block that already imports `LoreBookEnableToggled`, `LoreBookSettingsSaveRequested`, `LoreEntryAddRequested`).

- [ ] **Step 2: Run to verify it fails**

Run:
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/UI/test_personas_lore.py -k "export_selected_lore" -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: FAIL — `ImportError` (`LoreBookExportRequested` does not exist).

- [ ] **Step 3: Add the message class + Export button + handler to the detail widget**

In `personas_lore_detail.py`, add the message class next to `LoreBookSettingsSaveRequested` (after its definition, around `:66`):

```python
class LoreBookExportRequested(Message):
    """Intent: export the currently selected world book to a file (JSON)."""
```

In compose, add the Export button right after the "Save settings" button (`:151`):

```python
                yield Button("Save settings", id="personas-lore-settings-save", classes="console-action-secondary")
                yield Button("Export", id="personas-lore-export", classes="console-action-secondary")
```

Add the button handler next to the existing settings-save handler (`:456-459`):

```python
    @on(Button.Pressed, "#personas-lore-export")
    def _export_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(LoreBookExportRequested())
```

Add `"LoreBookExportRequested",` to `__all__` (`:477`, keep it alpha-ish next to `LoreBookEnableToggled`):

```python
__all__ = [
    "LoreBookEnableToggled",
    "LoreBookExportRequested",
    "LoreBookSettingsSaveRequested",
    ...
]
```

- [ ] **Step 4: Wire the screen handler + worker**

In `personas_screen.py`, add `LoreBookExportRequested` to the import block (`:90-98`):

```python
from ...Widgets.Persona_Widgets.personas_lore_detail import (
    LoreBookEnableToggled,
    LoreBookExportRequested,
    LoreBookSettingsSaveRequested,
    ...
)
```

Add the handler + worker right after `_handle_lore_settings_save` (`:2037` region). `asyncio`, `json` are already imported at the top:

```python
    @on(LoreBookExportRequested)
    async def _handle_lore_export(self, message: LoreBookExportRequested) -> None:
        message.stop()
        if self.state.selected_entity_kind != "lore" or not self.state.selected_entity_id:
            return
        if self._io_dialog_active:
            return
        self._io_dialog_active = True
        self.run_worker(self._lore_export_worker(), group="personas-io")

    async def _lore_export_worker(self) -> None:
        """Save the selected world book to a user-chosen JSON path."""
        from ...Widgets.enhanced_file_picker import EnhancedFileSave, Filters

        try:
            manager = self._lore_manager()
            entity_id = self.state.selected_entity_id
            if manager is None or not entity_id:
                return
            try:
                data = await asyncio.to_thread(manager.export_world_book, int(entity_id))
            except Exception as exc:
                logger.opt(exception=True).warning(f"Could not export world book {entity_id}.")
                self._notify(f"Export failed: {exc}", "error")
                return
            raw_name = str(data.get("name") or "world_book")
            safe_name = "".join(c for c in raw_name if c.isalnum() or c in " -_").rstrip()
            default_filename = f"{safe_name or 'world_book'}.json"
            picker = EnhancedFileSave(
                title="Export World Book",
                default_filename=default_filename,
                filters=Filters(
                    ("JSON Files", lambda p: p.suffix.lower() == ".json"),
                    ("All Files", lambda p: True),
                ),
                context="lore_export",
            )
            try:
                target = await self.app.push_screen_wait(picker)
            except Exception:
                logger.opt(exception=True).warning("Could not show the export file dialog.")
                return
            if not target:
                return
            body = json.dumps(data, indent=2, ensure_ascii=False)
            try:
                await asyncio.to_thread(target.write_text, body, "utf-8")
            except OSError as exc:
                logger.opt(exception=True).warning("Could not write the world-book export file.")
                self._notify(f"Export failed: {exc}", "error")
                return
            self._notify(f"Exported to {target}", "information")
        except Exception as exc:
            logger.opt(exception=True).error("Unexpected error exporting the world book.")
            self._notify(f"Export failed: {exc}", "error")
        finally:
            self._io_dialog_active = False
```

- [ ] **Step 5: Run to verify it passes**

Run the Step-2 command. Expected: PASS. Then run the whole test file to catch regressions:
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/UI/test_personas_lore.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Widgets/Persona_Widgets/personas_lore_detail.py tldw_chatbook/UI/Screens/personas_screen.py Tests/UI/test_personas_lore.py
git commit -m "feat(lore): export the selected world book to a JSON file"
```

---

### Task 3: Import flow

**Files:**
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_library_pane.py` (`set_mode` `:148-155`)
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py` (size constant near `:156`; `normalize_world_book_import` import; import branch `:2415-2419`; new `_open_lore_import_dialog` + `_lore_import_dialog_worker` + `_import_world_book_from_path` near the dictionary import methods `:2969`)
- Test: `Tests/UI/test_personas_lore.py`

**Interfaces:**
- Consumes: `normalize_world_book_import` (Task 1); `WorldBookManager.import_world_book(data, name_override) -> int`; `self._unique_lore_name(base)`; `self._render_lore_rows(query="")`; `self._select_lore_entry(id, name)`; `self._lore_manager()`; `validate_path_simple`; `ConflictError`.
- Produces: `_import_world_book_from_path(path: str)` (directly testable); import wired via the library Import button in lore mode.

- [ ] **Step 1: Write the failing import tests**

Append to `Tests/UI/test_personas_lore.py`:

```python
@pytest.mark.asyncio
async def test_import_world_book_from_file_creates_book_and_entries(
    mock_app_instance, stub_characters_lore, lore_db, seeded_lore_book, tmp_path
):
    """_import_world_book_from_path imports a tldw-shaped file, preserving priority
    and matching fields."""
    import json as _json
    mock_app_instance.chachanotes_db = lore_db
    app = LorePersonasTestApp(mock_app_instance)
    payload = {"name": "Imported Realm", "description": "", "scan_depth": 3,
               "token_budget": 500, "recursive_scanning": False,
               "entries": [{"keys": ["Sword"], "content": "a blade", "priority": 55,
                            "selective": True, "secondary_keys": ["hilt"],
                            "case_sensitive": True, "insertion_order": 0,
                            "position": "before_char", "enabled": True}]}
    f = tmp_path / "realm.json"
    f.write_text(_json.dumps(payload), "utf-8")
    async with app.run_test(size=(200, 60)) as pilot:
        screen = await _enter_lore(pilot)
        await _select_first_lore(pilot, screen)
        await screen._import_world_book_from_path(str(f))
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        manager = WorldBookManager(lore_db)
        books = manager.list_world_books(True)
        realm = next(b for b in books if b["name"] == "Imported Realm")
        entries = manager.get_world_book_entries(realm["id"])
        assert len(entries) == 1
        e = entries[0]
        assert e["keys"] == ["Sword"] and e["priority"] == 55
        assert e["selective"] is True and e["secondary_keys"] == ["hilt"] and e["case_sensitive"] is True


@pytest.mark.asyncio
async def test_import_sillytavern_world_info_object_form(
    mock_app_instance, stub_characters_lore, lore_db, seeded_lore_book, tmp_path
):
    """A SillyTavern World Info object-form file imports with fields remapped."""
    import json as _json
    mock_app_instance.chachanotes_db = lore_db
    app = LorePersonasTestApp(mock_app_instance)
    payload = {"name": "ST Book",
               "entries": {"0": {"key": ["Dragon"], "keysecondary": [], "content": "a wyrm",
                                 "order": 3, "position": 0, "disable": False}}}
    f = tmp_path / "st.json"
    f.write_text(_json.dumps(payload), "utf-8")
    async with app.run_test(size=(200, 60)) as pilot:
        screen = await _enter_lore(pilot)
        await _select_first_lore(pilot, screen)
        await screen._import_world_book_from_path(str(f))
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        manager = WorldBookManager(lore_db)
        book = next(b for b in manager.list_world_books(True) if b["name"] == "ST Book")
        e = manager.get_world_book_entries(book["id"])[0]
        assert e["keys"] == ["Dragon"] and e["content"] == "a wyrm" and e["insertion_order"] == 3


@pytest.mark.asyncio
async def test_import_malformed_world_book_creates_no_book(
    mock_app_instance, stub_characters_lore, lore_db, seeded_lore_book, tmp_path
):
    """A file whose entry has no keys is rejected up front — no partial book."""
    import json as _json
    mock_app_instance.chachanotes_db = lore_db
    app = LorePersonasTestApp(mock_app_instance)
    payload = {"name": "Bad Book", "entries": [{"keys": [], "content": "x"}]}
    f = tmp_path / "bad.json"
    f.write_text(_json.dumps(payload), "utf-8")
    async with app.run_test(size=(200, 60)) as pilot:
        screen = await _enter_lore(pilot)
        await _select_first_lore(pilot, screen)
        await screen._import_world_book_from_path(str(f))
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        manager = WorldBookManager(lore_db)
        assert all(b["name"] != "Bad Book" for b in manager.list_world_books(True))


@pytest.mark.asyncio
async def test_import_name_collision_renames(
    mock_app_instance, stub_characters_lore, lore_db, seeded_lore_book, tmp_path
):
    """Importing a book whose name clashes with an existing one imports under a
    unique name."""
    import json as _json
    mock_app_instance.chachanotes_db = lore_db
    app = LorePersonasTestApp(mock_app_instance)
    payload = {"name": "Blackreach",  # same as the seeded book
               "entries": [{"keys": ["Echo"], "content": "a sound"}]}
    f = tmp_path / "dup.json"
    f.write_text(_json.dumps(payload), "utf-8")
    async with app.run_test(size=(200, 60)) as pilot:
        screen = await _enter_lore(pilot)
        await _select_first_lore(pilot, screen)
        await screen._import_world_book_from_path(str(f))
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        manager = WorldBookManager(lore_db)
        names = [b["name"] for b in manager.list_world_books(True)]
        assert "Blackreach" in names
        assert any(n != "Blackreach" and n.startswith("Blackreach") for n in names)


@pytest.mark.asyncio
async def test_library_import_button_visible_in_lore_mode(
    mock_app_instance, stub_characters_lore, lore_db, seeded_lore_book
):
    """The library Import button is displayed in lore mode (un-gated)."""
    mock_app_instance.chachanotes_db = lore_db
    app = LorePersonasTestApp(mock_app_instance)
    async with app.run_test(size=(200, 60)) as pilot:
        screen = await _enter_lore(pilot)
        assert screen.query_one("#personas-library-import", Button).display is True
```

(`Button` is already imported at the top of the test file.) The four async import tests plus this button test are the load-bearing coverage — before Task 3 the button test goes RED because `set_mode` gates Import to characters/dictionaries.

- [ ] **Step 2: Run to verify they fail**

Run:
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/UI/test_personas_lore.py -k "import_world_book or sillytavern_world_info or malformed_world_book or name_collision_renames or library_import_button" \
-q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: FAIL — the four import tests error with `AttributeError: 'PersonasScreen' object has no attribute '_import_world_book_from_path'`, and `test_library_import_button_visible_in_lore_mode` fails its `display is True` assertion (Import is still gated to characters/dictionaries).

- [ ] **Step 3: Un-gate the Import button for lore**

In `personas_library_pane.py::set_mode` (`:148`), add `"lore"` and a lore tooltip:

```python
        self._import_visible = mode in ("characters", "dictionaries", "lore")
        import_button = self.query_one("#personas-library-import", Button)
        import_button.display = self._import_visible
        if mode == "dictionaries":
            import_button.tooltip = "Import a dictionary (JSON or Markdown)."
        elif mode == "lore":
            import_button.tooltip = "Import a world book (JSON)."
        else:
            import_button.tooltip = "Import a character card (PNG or JSON)."
```

(Replace the existing `import_button.tooltip = (...)` ternary with this if/elif/else.)

- [ ] **Step 4: Add the constant, import, action branch, and import methods to the screen**

In `personas_screen.py`, add the size constant beside `PERSONAS_DICTIONARY_IMPORT_MAX_BYTES` (`:156`):

```python
PERSONAS_WORLDBOOK_IMPORT_MAX_BYTES = 10 * 1024 * 1024
```

Add the normalizer import near the other `Character_Chat` imports (top of file):

```python
from ...Character_Chat.world_book_import import normalize_world_book_import
```

In `_handle_action_requested`, add a `lore` branch under `message.action == "import"` (after the dictionaries branch at `:2418`):

```python
        elif message.action == "import":
            if self.state.active_mode == "characters":
                await self._run_guarded(self._open_import_dialog)
            elif self.state.active_mode == "dictionaries":
                await self._run_guarded(self._open_dictionary_import_dialog)
            elif self.state.active_mode == "lore":
                await self._run_guarded(self._open_lore_import_dialog)
```

Add the three import methods near `_open_dictionary_import_dialog` (`:2969`):

```python
    async def _open_lore_import_dialog(self) -> None:
        """Continuation for the guarded lore-import action."""
        if self._io_dialog_active:
            logger.debug("Import/export dialog already active; ignoring import request.")
            return
        self._io_dialog_active = True
        self.run_worker(self._lore_import_dialog_worker(), group="personas-io")

    async def _lore_import_dialog_worker(self) -> None:
        """Show the import file picker and hand the chosen path off to import."""
        from ...Widgets.enhanced_file_picker import EnhancedFileOpen, Filters

        try:
            picker = EnhancedFileOpen(
                title="Import World Book",
                filters=Filters(
                    ("World books (JSON)", lambda p: p.suffix.lower() == ".json"),
                    ("All Files", lambda p: True),
                ),
                context="lore_import",
            )
            try:
                file_path = await self.app.push_screen_wait(picker)
            except Exception:
                logger.opt(exception=True).warning("Could not show the world-book import dialog.")
                return
            if file_path:
                await self._import_world_book_from_path(str(file_path))
        except Exception as exc:
            logger.opt(exception=True).error("Unexpected error in the world-book import worker.")
            self._notify(f"Import failed: {exc}", "error")
        finally:
            self._io_dialog_active = False

    async def _import_world_book_from_path(self, path: str) -> None:
        """Validate + normalize a world-book file and import it; rename on clash.

        Args:
            path: Filesystem path to the ``.json`` file chosen via the picker.
        """
        manager = self._lore_manager()
        if manager is None:
            self._notify("Lore is not configured: the database is unavailable.", "error")
            return
        try:
            source = validate_path_simple(path, require_exists=True)
        except (ValueError, OSError) as exc:
            logger.opt(exception=True).warning(f"Rejected world-book import path {path}.")
            self._notify(f"Import failed: {exc}", "error")
            return
        try:
            if source.stat().st_size > PERSONAS_WORLDBOOK_IMPORT_MAX_BYTES:
                self._notify(
                    f"Import failed: file is larger than "
                    f"{PERSONAS_WORLDBOOK_IMPORT_MAX_BYTES // (1024 * 1024)} MB.",
                    "error",
                )
                return
        except OSError as exc:
            self._notify(f"Import failed: {exc}", "error")
            return
        try:
            text = await asyncio.to_thread(source.read_text, "utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            logger.opt(exception=True).warning(f"Could not read world-book file {path}.")
            self._notify(f"Import failed: {exc}", "error")
            return
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, RecursionError, ValueError) as exc:
            self._notify(f"Import failed: not valid JSON ({exc})", "error")
            return
        try:
            data = normalize_world_book_import(parsed)
        except ValueError as exc:
            self._notify(f"Import failed: {exc}", "error")
            return
        base = str(data.get("name") or "Imported world book")
        name = self._unique_lore_name(base)
        try:
            new_id = await asyncio.to_thread(manager.import_world_book, data, name_override=name)
        except ConflictError:
            self._notify("A lore book with that name already exists.", "error")
            return
        except Exception as exc:
            logger.opt(exception=True).warning(f"World-book import failed for {path}.")
            self._notify(f"Import failed: {exc}", "error")
            return
        if self.state.active_mode != "lore":
            self._notify(f"Imported '{name}' — open Lore to see it.", "information")
            return
        await self._render_lore_rows(query="")
        await self._select_lore_entry(str(new_id), name)
        suffix = " Renamed to avoid a name clash." if name != base else ""
        self._notify(f"Imported '{name}'.{suffix}", "information")
```

- [ ] **Step 5: Run to verify they pass**

Run the Step-2 command. Expected: PASS (5 tests, including the button test).

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Widgets/Persona_Widgets/personas_library_pane.py tldw_chatbook/UI/Screens/personas_screen.py Tests/UI/test_personas_lore.py
git commit -m "feat(lore): import a world book from a JSON file (with SillyTavern World Info support)"
```

---

### Task 4: Round-trip integration + full gate

**Files:**
- Test: `Tests/UI/test_personas_lore.py`

**Interfaces:**
- Consumes: `_lore_export_worker`/`export_world_book` (Task 2) + `_import_world_book_from_path` (Task 3).

- [ ] **Step 1: Write the round-trip test**

Append to `Tests/UI/test_personas_lore.py`:

```python
@pytest.mark.asyncio
async def test_export_then_import_round_trip_preserves_entries(
    mock_app_instance, stub_characters_lore, lore_db, seeded_lore_book, tmp_path, monkeypatch
):
    """Export the seeded book to a file, then import that file — the new book's
    entry matches the original (keys/content preserved through export→import)."""
    mock_app_instance.chachanotes_db = lore_db
    app = LorePersonasTestApp(mock_app_instance)
    manager = WorldBookManager(lore_db)
    # Give the seeded entry non-default matching fields to prove they survive.
    entry = manager.get_world_book_entries(seeded_lore_book["book_id"])[0]
    manager.update_world_book_entry(entry["id"], priority=70, selective=True,
                                    secondary_keys=["oath"], case_sensitive=True)
    target = tmp_path / "roundtrip.json"
    async with app.run_test(size=(200, 60)) as pilot:
        screen = await _enter_lore(pilot)
        await _select_first_lore(pilot, screen)

        async def _fake_save(picker):
            return target

        monkeypatch.setattr(app, "push_screen_wait", _fake_save)
        screen.post_message(LoreBookExportRequested())
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        assert target.exists()
        await screen._import_world_book_from_path(str(target))
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
    books = manager.list_world_books(True)
    imported = next(b for b in books if b["name"] != "Blackreach" and b["name"].startswith("Blackreach"))
    e = manager.get_world_book_entries(imported["id"])[0]
    assert e["keys"] == ["Warden"] and e["priority"] == 70
    assert e["selective"] is True and e["secondary_keys"] == ["oath"] and e["case_sensitive"] is True
```

- [ ] **Step 2: Run to verify it passes**

Run:
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/UI/test_personas_lore.py::test_export_then_import_round_trip_preserves_entries \
-q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: PASS (Tasks 2 + 3 make it green).

- [ ] **Step 3: Run the full gate**

```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
Tests/Character_Chat/test_world_book_import.py Tests/UI/test_personas_lore.py \
Tests/Character_Chat/test_world_book_manager.py Tests/Character_Chat/test_world_info.py \
-q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: all pass. Then the import smoke:
```bash
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c "import tldw_chatbook.app; print('APP IMPORT OK')"
```
Expected: `APP IMPORT OK`.

- [ ] **Step 4: Commit**

```bash
git add Tests/UI/test_personas_lore.py
git commit -m "test(lore): export→import round-trip preserves entries + matching fields"
```

---

## Notes for the reviewer

- **No manager/schema change.** Any edit to `world_book_manager.py` or `ChaChaNotes_DB.py` is out of scope — reject it.
- **Validation is up front.** `normalize_world_book_import` must validate the whole file before `import_world_book` runs, so a malformed file leaves NO partial book (Task 3's `test_import_malformed_world_book_creates_no_book` pins this).
- **Never-crash.** Every import/export failure path must `self._notify(...)` and return, not raise.
- **`_import_world_book_from_path` is directly testable** (no picker) — that's deliberate; the picker workers are thin wrappers.
- The export/round-trip tests monkeypatch `app.push_screen_wait` to stand in for the save dialog — this is the seam for driving `EnhancedFileSave` in tests.
