# Preserve Embedded Lorebook on Character Import (TASK-429) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Importing a V2 character card with an embedded `character_book` converts it into one managed `extensions['character_world_books']` snapshot on the saved character (visible/attached/injected once), drops the legacy `character_book` key to avoid double-injection, and toasts what happened.

**Architecture:** A lenient converter (`world_book_import.py`) reuses the existing `_normalize_entry` to salvage entries; `parse_v2_card` calls it for both the top-level V2 field and the nested legacy key, merges the block deduped-by-name, and pops `character_book` only after a block with entries is built; the Personas import handler re-reads the saved character to name the book in the toast.

**Tech Stack:** Python 3.11+, pytest, SQLite (ChaChaNotes_DB), Textual.

## Global Constraints

- **Run tests via the repo venv, from the worktree root:** `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest ...`
- **No ChaChaNotes migration** (schema v22; `character_world_books` is an `extensions` JSON blob).
- **No export changes** (extensions round-trips verbatim → AC#3).
- **Design source of truth:** `Docs/superpowers/specs/2026-07-21-lorebook-import-preservation-design.md` (rev 2).
- **Snapshot block shape (must match `export_world_book`, `world_book_manager.py:954-977`, + book-level `enabled`):** book = `{name, description, scan_depth, token_budget, recursive_scanning, enabled, entries[]}`; entry = `_normalize_entry` output = `{keys, secondary_keys, content, insertion_order, position, selective, case_sensitive, enabled, priority, extensions, regex}`.
- **Correctness invariant:** never leave both `extensions['character_book']` and `extensions['character_world_books']` on the same card (double-injects at send-time — `world_info_resolver.py:47-63`); and never pop a `character_book` that was not converted into a block with ≥1 entry.
- **Commit trailer:** end each commit body with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

## Task 1: Lenient `character_book` → snapshot-block converter

**Files:**
- Modify: `tldw_chatbook/Character_Chat/world_book_import.py`
- Test: `Tests/Character_Chat/test_world_book_import.py`

**Interfaces:**
- Produces: `character_book_to_world_book_block(book: Any, fallback_name: str) -> tuple[dict | None, int, int]` returning `(block_or_None, imported_count, skipped_count)`. `None` only when `book` is not a dict.

- [ ] **Step 1: Write the failing tests**

Add to `Tests/Character_Chat/test_world_book_import.py`:

```python
from tldw_chatbook.Character_Chat.world_book_import import (
    character_book_to_world_book_block,
)


def test_character_book_to_block_basic():
    book = {
        "name": "Second Chance Lore",
        "description": "ship lore",
        "scan_depth": 5,
        "token_budget": 300,
        "recursive_scanning": True,
        "entries": [
            {"keys": ["coffee"], "content": "The machine explodes.",
             "enabled": True, "insertion_order": 1, "position": 0},
            {"keys": ["airlock"], "content": "It sticks.",
             "enabled": True, "insertion_order": 2},
        ],
    }
    block, imported, skipped = character_book_to_world_book_block(book, "X Lorebook")
    assert imported == 2 and skipped == 0
    assert block["name"] == "Second Chance Lore"
    assert block["scan_depth"] == 5 and block["token_budget"] == 300
    assert block["recursive_scanning"] is True and block["enabled"] is True
    # int position (0) normalized to the string enum
    assert block["entries"][0]["position"] == "before_char"
    assert block["entries"][0]["keys"] == ["coffee"]
    assert block["entries"][0]["regex"] is False


def test_character_book_to_block_skips_unsalvageable_and_counts():
    book = {"name": "B", "entries": [
        {"keys": ["ok"], "content": "good", "enabled": True, "insertion_order": 1},
        {"content": "no keys", "enabled": True, "insertion_order": 2},   # no keys -> skip
        {"keys": ["x"], "enabled": True, "insertion_order": 3},          # no content -> skip
    ]}
    block, imported, skipped = character_book_to_world_book_block(book, "X Lorebook")
    assert imported == 1 and skipped == 2
    assert len(block["entries"]) == 1


def test_character_book_to_block_empty_name_uses_fallback():
    block, _, _ = character_book_to_world_book_block(
        {"name": "", "entries": []}, "Elara Lorebook")
    assert block["name"] == "Elara Lorebook"


def test_character_book_to_block_non_dict_returns_none():
    assert character_book_to_world_book_block(None, "X") == (None, 0, 0)
    assert character_book_to_world_book_block([1, 2], "X") == (None, 0, 0)


def test_character_book_to_block_entries_as_object_form():
    book = {"name": "B", "entries": {"0": {"keys": ["k"], "content": "c",
            "enabled": True, "insertion_order": 1}}}
    block, imported, skipped = character_book_to_world_book_block(book, "X")
    assert imported == 1 and block["entries"][0]["keys"] == ["k"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Character_Chat/test_world_book_import.py -k character_book_to_block -v`
Expected: FAIL — `ImportError: cannot import name 'character_book_to_world_book_block'`.

- [ ] **Step 3: Implement the converter**

In `world_book_import.py`, confirm a module logger exists (grep `logger`); if absent, add `from loguru import logger` near the top imports. Append after `normalize_world_book_import`:

```python
def character_book_to_world_book_block(
    book: Any, fallback_name: str
) -> tuple["dict | None", int, int]:
    """Convert a V2 card ``character_book`` into one embedded
    ``character_world_books`` snapshot block (task-429).

    Salvage-and-count: entries that ``_normalize_entry`` rejects (not a dict,
    no keys, no content, or an invalid regex pattern) are skipped and counted,
    so one bad entry never sinks the book. Never raises.

    Returns:
        ``(block_or_None, imported_count, skipped_count)``. ``None`` only when
        ``book`` is not a dict.
    """
    if not isinstance(book, dict):
        return None, 0, 0
    raw_name = book.get("name")
    name = (
        raw_name.strip()
        if isinstance(raw_name, str) and raw_name.strip()
        else fallback_name
    )
    raw_entries = book.get("entries")
    if isinstance(raw_entries, dict):
        entries_list = list(raw_entries.values())
    elif isinstance(raw_entries, list):
        entries_list = raw_entries
    else:
        entries_list = []
    normalized: List[Dict[str, Any]] = []
    skipped = 0
    for i, entry in enumerate(entries_list):
        try:
            normalized.append(_normalize_entry(entry, i))
        except ValueError as exc:
            skipped += 1
            logger.warning(f"character_book entry skipped on import: {exc}")
    block = {
        "name": name,
        "description": str(book.get("description") or ""),
        "scan_depth": _coerce_int(book.get("scan_depth"), 3),
        "token_budget": _coerce_int(book.get("token_budget"), 500),
        "recursive_scanning": _coerce_bool(book.get("recursive_scanning"), False),
        "enabled": _coerce_bool(book.get("enabled"), True),
        "entries": normalized,
    }
    return block, len(normalized), skipped
```

(`List`/`Dict`/`Any` are already imported in this module — confirm; add to the `typing` import if not.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Character_Chat/test_world_book_import.py -v`
Expected: PASS (new + existing).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Character_Chat/world_book_import.py Tests/Character_Chat/test_world_book_import.py
git commit -m "feat(lore): lenient character_book -> world-book snapshot converter (task-429)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 2: Convert-and-replace in `parse_v2_card` (import wiring)

**Files:**
- Modify: `tldw_chatbook/Character_Chat/Character_Chat_Lib.py` (`parse_v2_card`, the `character_book` block ~1272-1279)
- Test: `Tests/Character_Chat/test_character_file_operations.py`

**Interfaces:**
- Consumes: `character_book_to_world_book_block` (Task 1).
- Produces: after `import_and_save_character_from_file`, a saved character with `extensions['character_world_books']` = `[block]` and **no** `extensions['character_book']` (when a book with ≥1 salvaged entry was present).

- [ ] **Step 1: Write the failing tests**

Add to `Tests/Character_Chat/test_character_file_operations.py` (reuse `db_instance`; build a V2 card dict with a `character_book` and import it via `import_and_save_character_from_file` from a JSON file/bytes — mirror how the existing tests feed the importer; if they use a temp `.json` path, write one with `json.dump`).

```python
def _v2_card_with_book(top_level_book=True):
    book = {"name": "Second Chance Lore", "description": "d",
            "entries": [{"keys": ["coffee"], "content": "explodes",
                         "enabled": True, "insertion_order": 1, "position": 0}]}
    data = {"name": "Elara", "description": "cap", "first_mes": "hi",
            "extensions": {}}
    if top_level_book:
        data["character_book"] = book
    else:
        data["extensions"]["character_book"] = book   # legacy nested shape
    return {"spec": "chara_card_v2", "spec_version": "2.0", "data": data}


def test_import_converts_top_level_character_book(db_instance, tmp_path):
    p = tmp_path / "elara.json"
    p.write_text(json.dumps(_v2_card_with_book(top_level_book=True)))
    cid = import_and_save_character_from_file(db_instance, str(p))
    rec = db_instance.get_character_card_by_id(cid)
    ext = rec["extensions"]
    ext = json.loads(ext) if isinstance(ext, str) else ext
    assert "character_book" not in ext           # no double-inject
    books = ext["character_world_books"]
    assert len(books) == 1 and books[0]["name"] == "Second Chance Lore"
    assert books[0]["entries"][0]["keys"] == ["coffee"]


def test_import_converts_nested_legacy_character_book(db_instance, tmp_path):
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps(_v2_card_with_book(top_level_book=False)))
    cid = import_and_save_character_from_file(db_instance, str(p))
    rec = db_instance.get_character_card_by_id(cid)
    ext = rec["extensions"]
    ext = json.loads(ext) if isinstance(ext, str) else ext
    assert "character_book" not in ext
    assert ext["character_world_books"][0]["name"] == "Second Chance Lore"


def test_import_book_with_no_salvageable_entries_keeps_legacy_key(db_instance, tmp_path):
    card = _v2_card_with_book(top_level_book=False)
    card["data"]["extensions"]["character_book"]["entries"] = [
        {"content": "no keys", "enabled": True, "insertion_order": 1}
    ]
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(card))
    cid = import_and_save_character_from_file(db_instance, str(p))
    rec = db_instance.get_character_card_by_id(cid)
    ext = rec["extensions"]
    ext = json.loads(ext) if isinstance(ext, str) else ext
    # nothing salvaged -> legacy key untouched, no empty block written
    assert "character_world_books" not in ext
    assert "character_book" in ext
```

Match the exact importer-call convention the existing tests use (grep `import_and_save_character_from_file(` in the file; it may take a path string, bytes, or a stream — feed it the same way). If the existing round-trip test (`test_reimport_exported_character`) shows the PNG/JSON feeding pattern, reuse it.

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Character_Chat/test_character_file_operations.py -k "character_book" -v`
Expected: FAIL — the saved character has `extensions['character_book']` and no `character_world_books`.

- [ ] **Step 3: Rewrite the `parse_v2_card` character_book handling**

In `Character_Chat_Lib.py`, add the import near the top (or locally inside `parse_v2_card` if a module-level import risks a cycle — grep whether `world_book_import` imports `Character_Chat_Lib`; it does not, so a top-level import is fine): `from .world_book_import import character_book_to_world_book_block`.

Replace the current block (`~1272-1279`):

```python
        if "character_book" in data_node and isinstance(
            data_node["character_book"], dict
        ):
            if not isinstance(parsed_data["extensions"], dict):
                parsed_data["extensions"] = {}
            parsed_data["extensions"]["character_book"] = parse_character_book(
                data_node["character_book"]
            )
```

with:

```python
        # TASK-429: convert an embedded V2 character_book into the app's managed
        # character_world_books snapshot so it is visible/attached and injects
        # ONCE. Handle the top-level V2 field AND the nested legacy key (cards
        # exported by this app before the fix carry the book under extensions).
        # Only drop character_book after a block with >=1 entry is built, so a
        # book that yields nothing keeps its legacy key (still injects).
        if not isinstance(parsed_data["extensions"], dict):
            parsed_data["extensions"] = {}
        _source_book = None
        if isinstance(data_node.get("character_book"), dict):
            _source_book = data_node["character_book"]
        elif isinstance(parsed_data["extensions"].get("character_book"), dict):
            _source_book = parsed_data["extensions"]["character_book"]
        if _source_book is not None:
            _fallback = f"{parsed_data.get('name') or 'Character'} Lorebook"
            _block, _imported, _skipped = character_book_to_world_book_block(
                _source_book, _fallback
            )
            if _block is not None and _imported > 0:
                _existing = parsed_data["extensions"].get("character_world_books")
                if not isinstance(_existing, list):
                    _existing = []
                if not any(
                    isinstance(b, dict)
                    and str(b.get("name")) == str(_block.get("name"))
                    for b in _existing
                ):
                    _existing = _existing + [_block]
                parsed_data["extensions"]["character_world_books"] = _existing
                parsed_data["extensions"].pop("character_book", None)
                logger.info(
                    "Imported character lorebook '%s': %d entries (%d skipped)",
                    _block["name"], _imported, _skipped,
                )
            else:
                logger.warning(
                    "character_book on import yielded no usable entries "
                    "(%d skipped); leaving any legacy key intact.", _skipped,
                )
```

Wrap the whole block in a `try/except Exception:` that logs and continues, so malformed imported content never raises out of `parse_v2_card`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Character_Chat/test_character_file_operations.py -v`
Expected: PASS (new + existing round-trip tests). Existing `test_reimport_exported_character` must stay green — verifies AC#3 (a converted block round-trips because export copies extensions verbatim).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Character_Chat/Character_Chat_Lib.py Tests/Character_Chat/test_character_file_operations.py
git commit -m "feat(lore): import converts character_book to managed world-book, drops legacy key (task-429)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 3: Import toast — name the lorebook / honest conflict copy

**Files:**
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py` (`_import_character_from_path` ~3762-3806)
- Test: `Tests/UI/test_personas_workbench.py` (or the personas import test module — grep for an existing `_import_character_from_path` test)

**Interfaces:**
- Consumes: Task 2's saved `extensions['character_world_books']`; `ccp_character_handler.fetch_character_by_id` (`ccp_character_handler.py:81`, returns the card dict with parsed `extensions`).

- [ ] **Step 1: Write the failing test**

Add a test that imports a card with a book through the Personas flow (or unit-tests a small helper). Prefer a helper so the toast string is assertable without a full mounted screen. Add a helper `_imported_lorebook_note(self, character_id) -> str` (Step 3) and test it directly with a fake `ccp_character_handler.fetch_character_by_id` (monkeypatch) returning a record whose `extensions['character_world_books']` has one 2-entry block:

```python
async def test_imported_lorebook_note_names_book(monkeypatch):
    from tldw_chatbook.UI.Screens import personas_screen as ps
    monkeypatch.setattr(
        ps.ccp_character_handler, "fetch_character_by_id",
        lambda cid: {"extensions": {"character_world_books": [
            {"name": "Second Chance Lore", "entries": [{}, {}]}]}},
    )
    note = await ps.PersonasScreen._imported_lorebook_note.__get__(
        _MinimalScreenStub())(  # or construct via the test's existing screen harness
        "7")
    assert "Second Chance Lore" in note and "2 entries" in note
```

If the file has no async-helper test harness, instead assert via the existing Personas screen test harness (grep how `test_*import*` mounts the screen) that after importing a book-carrying card the notify text contains the book name. Match the real `_notify`/notification capture the file already uses.

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_personas_workbench.py -k lorebook_note -v`
Expected: FAIL — `_imported_lorebook_note` does not exist.

- [ ] **Step 3: Implement the toast helper + wire it**

In `personas_screen.py` confirm `json` is imported (add `import json` if absent; `asyncio` is already used at `:3769`). Add the helper:

```python
    async def _imported_lorebook_note(self, character_id: str) -> str:
        """Return a ' Lorebook 'X' attached (N entries).' suffix, or '' (task-429)."""
        try:
            record = await asyncio.to_thread(
                ccp_character_handler.fetch_character_by_id, character_id
            )
        except Exception:
            return ""
        ext = (record or {}).get("extensions")
        if isinstance(ext, str):
            try:
                ext = json.loads(ext)
            except (ValueError, TypeError):
                ext = {}
        if not isinstance(ext, dict):
            return ""
        books = ext.get("character_world_books")
        if not isinstance(books, list) or not books:
            return ""
        first = books[0] if isinstance(books[0], dict) else {}
        name = str(first.get("name") or "lorebook")
        entries = first.get("entries")
        n = len(entries) if isinstance(entries, list) else 0
        return f" Lorebook '{name}' attached ({n} entries)."
```

Then update the two notify branches (`:3803-3806`):

```python
        if imported_id in pre_import_ids:
            self._notify(
                "Character already existed; selected it. "
                "Re-importing does not update an existing character.",
                "information",
            )
        else:
            lore_note = await self._imported_lorebook_note(imported_id)
            self._notify(f"Character imported.{lore_note}", "information")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_personas_workbench.py -k "import or lorebook" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/personas_screen.py Tests/UI/test_personas_workbench.py
git commit -m "feat(lore): name the imported lorebook in the toast; honest re-import copy (task-429)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 4: Regression + live verification

**Files:** none (verification only).

- [ ] **Step 1: Run the affected suites**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Character_Chat/test_world_book_import.py Tests/Character_Chat/test_character_file_operations.py Tests/Character_Chat/test_resolve_character_world_books.py Tests/Character_Chat/test_character_world_book_send_path.py -q`
Expected: all green (the send-path/resolve tests confirm no double-injection and the block reads correctly).

- [ ] **Step 2: Live-verify in the real TUI**

Using the `verify` recipe (scratch `TLDW_CONFIG_PATH` profile): import the review's SillyTavern-style card (a V2 PNG/JSON carrying a `character_book`). Confirm:
- the import toast names the lorebook and its entry count;
- the character's "World Books (embedded copies)" panel lists the book with its entries;
- a chat that mentions a keyword injects the lore **once** (not twice);
- export the character to JSON and re-import → the book still appears (round-trip).

- [ ] **Step 3: Mark ACs + notes**

```bash
backlog task edit 429 --check-ac 1 --check-ac 2 --check-ac 3
backlog task edit 429 --notes "<implementation summary>"
```

---

## Self-Review

- **Spec coverage:** §1 converter→Task 1; §2 convert-and-replace (two shapes, convert-before-pop, merge/dedup)→Task 2; §3 toast (book name + honest conflict)→Task 3; AC#3 round-trip→Task 2 existing round-trip test + Task 4 live; no-double-inject invariant→Task 2 test + Task 4 send-path suite. All covered.
- **Placeholder scan:** test bodies note where to match the real importer-call convention / notification capture (grep instructions, not TBDs). No functional placeholders.
- **Type consistency:** `character_book_to_world_book_block(book, fallback_name) -> (block|None, int, int)` used identically in Tasks 1/2; block/entry key names match `export_world_book` + `_normalize_entry` throughout.
- **Ordering:** 1 (converter) → 2 (wiring, consumes converter) → 3 (toast, consumes saved block) → 4 (verify).
