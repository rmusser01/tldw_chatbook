# Roleplay P2d-2 — world-book (Lore) import/export UI

**Status:** Design.

**Program:** Roleplay (Personas) redesign — P2 (Lore mode). Second half of P2d, after P2d-1 (matching-controls editor, merged #694). Schema stays v21.

## Why

A user can create, edit, duplicate, and delete world books in the Roleplay Lore mode, but cannot import a world book from a file or export the selected one. `WorldBookManager` already has `export_world_book`/`import_world_book` (a full SillyTavern-character-book-compatible round-trip, carrying priority + selective/secondary_keys/case_sensitive), and the internal Duplicate already uses them — but there is no file I/O UI. P2d-2 adds import-from-file and export-to-file, plus an adapter so real SillyTavern *World Info* exports (a different field shape) import correctly.

## Scope

**In scope:** import a world book from a `.json` file (with a normalization adapter covering tldw exports, character-book array form, and SillyTavern World Info object-form); export the selected world book to a `.json` file; a pure, unit-tested normalization/validation function.

**Explicitly NOT in scope (deferred):**
- Retiring the legacy `ccp_dictionary_handler.py` "Import Dictionary/World Book" TODO stub → P2g.
- Extracting an embedded `character_book` from a full character-card JSON → later (overlaps Characters-mode card import).
- Markdown export (world books have no Markdown equivalent — JSON only).
- Any change to `world_book_manager.py` (manager) or the schema (stays v21).

## Behavior-change framing

Pure additive UI + one new pure helper module. No schema migration, no manager change, no matcher change. Existing flows (create/edit/duplicate/delete/Try-it) are untouched.

## Ground truths (verified at dev tip, post-#694)

- **Manager:** `export_world_book(id) -> dict` emits `{name, description, scan_depth, token_budget, recursive_scanning, entries: [ {keys, content, enabled, position, insertion_order, selective, secondary_keys, case_sensitive, extensions, priority} ]}`. `import_world_book(data, name_override=None) -> int` calls `create_world_book(name)` then, for each `entry` in `data.get('entries', [])` (a **list**), `create_world_book_entry(keys=entry.get('keys',[]), content=entry.get('content',''), ...)`.
  - `import_world_book` is **NOT atomic** (book insert, then per-entry inserts in separate transactions) and iterates `entries` as a **list**; feeding a dict (`enumerate` yields string keys) or an entry with empty `keys`/`content` (`create_world_book_entry` raises `InputError`) crashes mid-loop and leaves a **partial book**. → The UI must fully validate/normalize before calling it.
  - `world_books.name` is `UNIQUE NOT NULL`; `create_world_book` raises `ConflictError` on a duplicate name; `import_world_book` accepts `name_override` for exactly this.
- **Screen (`personas_screen.py`) — reusable pieces already present:**
  - `_unique_lore_name(base)` (`:2547`) disambiguates against `self._lore_books_cache`.
  - `_render_lore_rows(query="")` (`:993`) reloads the lore library list.
  - `_select_lore_entry(entity_id, name)` (`:1286`) selects a book.
  - `_duplicate_selected_lore` (`:2581`) is already an `export_world_book` → `import_world_book(name_override=unique)` → `_render_lore_rows("")` template (the exact collision + refresh pattern).
  - `_io_dialog_active` (`:397`) re-entrancy guard; I/O workers run in `group="personas-io"`; user feedback via `self._notify(msg, level)`.
  - `_run_guarded(continuation)` (`:3727`) wraps guarded actions.
  - `validate_path_simple` imported (`:29`); `PERSONAS_DICTIONARY_IMPORT_MAX_BYTES = 10*1024*1024` (`:154`).
  - Dictionary import template: `_open_dictionary_import_dialog`/`_dictionary_import_dialog_worker` (`:2953`/`:2961`) + `_import_dictionary_from_path` (`:2989`); export template: `_handle_dictionary_export` (`:1405`) + `_dictionary_export_worker` (`:1431`) using `EnhancedFileSave`.
  - `_handle_action_requested` (`:2387`) `import` branch handles characters/dictionaries — **no `lore` branch** yet.
- **Library pane (`personas_library_pane.py`):** `set_mode` gates the Import button visible for `("characters", "dictionaries")` (`:148`) — needs `"lore"` added; the Import button posts `PersonaActionRequested(action="import")` (`:323`). `PersonaAction` literal already includes `"import"`/`"export"`.
- **Lore detail (`personas_lore_detail.py`):** Settings tab composes name/description/scan-depth/token-budget/recursive/enabled + a "Save settings" button (`:131-142`); it already defines typed messages (e.g. `LoreBookSettingsSaveRequested`) that the screen handles — the pattern a new `LoreBookExportRequested` follows.
- **File dialogs:** `EnhancedFileOpen`/`EnhancedFileSave` + `Filters` from `Widgets/enhanced_file_picker.py`, via `await self.app.push_screen_wait(picker)` → `Path | None`. `Filters` entries are `(label, callable)`.
- **SillyTavern World Info format** (object-form export) differs from tldw/character-book: entries is an **object** `{"0": {...}}`, and fields are `key` (not `keys`), `keysecondary` (not `secondary_keys`), `order` (not `insertion_order`), integer `position` (0/1/…), `disable` (inverted `enabled`), `comment` (a label, no tldw equivalent). The character-book array form (and tldw's own export) already use tldw's field names.

## Architecture

### 1. Normalization adapter — new pure module `tldw_chatbook/Character_Chat/world_book_import.py`

A pure, DB-free, unit-tested function that maps *any* supported input shape to tldw's field names **and** validates every entry, so the screen can reject a bad file before any DB write (no partial import).

```
def normalize_world_book_import(data: dict) -> dict:
    # raise ValueError(<user message>) on any invalid shape
```

- If `data` is not a dict → `ValueError("World book file must be a JSON object.")`
- Resolve entries container: `entries = data.get('entries')`; `None` → `[]`; a dict → `list(entries.values())` (World Info object-form); a list → as-is; anything else → `ValueError("'entries' must be a list or object.")`
- Map + validate each entry via `_normalize_entry(entry, index)`; return `{**data, 'entries': [<normalized>]}` (metadata keys — name/description/scan_depth/token_budget/recursive_scanning — pass through untouched for `import_world_book`).

`_normalize_entry(entry, index) -> dict` (raises `ValueError` with the 1-based index on invalid data):
- Not a dict → `ValueError(f"Entry {index+1} is not an object.")`
- `keys = _as_str_list(entry.get('keys') if entry.get('keys') is not None else entry.get('key'))`; if no non-blank key → `ValueError(f"Entry {index+1} has no keys.")`
- `content = str(entry.get('content', ''))`; if blank → `ValueError(f"Entry {index+1} has no content.")`
- `secondary_keys = _as_str_list(entry.get('secondary_keys') if entry.get('secondary_keys') is not None else entry.get('keysecondary'))`
- `insertion_order = _coerce_int(entry.get('insertion_order', entry.get('order', index)), index)`
- `position = _normalize_position(entry.get('position'))`
- `selective = bool(entry.get('selective', False))`
- `case_sensitive = bool(entry.get('case_sensitive', entry.get('caseSensitive', False)))`
- `enabled = bool(entry.get('enabled', not entry.get('disable', False)))`
- `priority = _coerce_int(entry.get('priority', 0), 0)`
- `extensions = entry.get('extensions') if isinstance(entry.get('extensions'), dict) else {}` (pass-through for round-trip fidelity; `import_world_book` reads it)
- returns exactly the field set `import_world_book` consumes: `{keys, secondary_keys, content, insertion_order, position, selective, case_sensitive, enabled, priority, extensions}`

Helpers (same module):
- `_as_str_list(v)`: `None`→`[]`; `str`→`[v]` (if non-blank); `list`→`[str(x) for x in v if str(x).strip()]`; else `[]`.
- `_coerce_int(v, default)`: `try: int(v) except (TypeError, ValueError): default` (mirrors the P2c/P2d-0 helper).
- `_normalize_position(pos)`: string in `{before_char, after_char, at_start, at_end}` → itself; `int` → `{0:"before_char", 1:"after_char"}.get(pos, "before_char")`; else `"before_char"`.

### 2. Import flow (`personas_screen.py` + `personas_library_pane.py`)

- **`personas_library_pane.py::set_mode`** (`:148`): add `"lore"` → `self._import_visible = mode in ("characters", "dictionaries", "lore")`.
- **`_handle_action_requested`** (`:2399` import branch): add `elif self.state.active_mode == "lore": await self._run_guarded(self._open_lore_import_dialog)`.
- **`_open_lore_import_dialog`** (mirror `_open_dictionary_import_dialog`): re-entrancy guard on `_io_dialog_active`; `self.run_worker(self._lore_import_dialog_worker(), group="personas-io")`.
- **`_lore_import_dialog_worker`** (mirror the dictionary worker): `EnhancedFileOpen(title="Import World Book", filters=Filters(("World books (JSON)", lambda p: p.suffix.lower()==".json"), ("All Files", lambda p: True)), context="lore_import")` → `push_screen_wait` → `_import_world_book_from_path(str(path))`; `finally: self._io_dialog_active = False`.
- **`_import_world_book_from_path(path)`**:
  1. `manager = self._lore_manager()`; if None → notify + return.
  2. `validate_path_simple(path, require_exists=True)` (reject `ValueError`/`OSError` → notify).
  3. Size cap: `source.stat().st_size > PERSONAS_WORLDBOOK_IMPORT_MAX_BYTES` (a new `10*1024*1024` constant beside the dictionary one) → notify + return.
  4. `text = await asyncio.to_thread(source.read_text, "utf-8")` (reject `OSError`/`UnicodeDecodeError`).
  5. `parsed = json.loads(text)` (reject `JSONDecodeError`/`ValueError`/`RecursionError` → "not valid JSON").
  6. `try: data = normalize_world_book_import(parsed) except ValueError as exc: self._notify(f"Import failed: {exc}", "error"); return` — **all validation up front; no partial import.**
  7. Collision + import (mirror `_duplicate_selected_lore`): `base = str(data.get("name") or "Imported world book")`; `name = self._unique_lore_name(base)`; `new_id = await asyncio.to_thread(manager.import_world_book, data, name_override=name)`; on `ConflictError` (race) notify + return; other `Exception` → notify + return.
  8. On success: `await self._render_lore_rows(query="")`; `await self._select_lore_entry(str(new_id), name)`; `self._notify(f"Imported '{name}'." + (" Renamed to avoid a name clash." if name != base else ""), "information")`.

### 3. Export flow (`personas_lore_detail.py` + `personas_screen.py`)

- **`personas_lore_detail.py`:** add a new typed message `class LoreBookExportRequested(Message)` (no payload — JSON only) and an "Export" `Button(id="personas-lore-export")` in the **Settings tab** next to "Save settings"; a `@on(Button.Pressed, "#personas-lore-export")` handler posts `LoreBookExportRequested()`.
- **`_handle_lore_export`** (`@on(LoreBookExportRequested)`, mirror `_handle_dictionary_export`): `message.stop()`; return unless `self.state.selected_entity_kind == "lore" and self.state.selected_entity_id`; re-entrancy guard; `self.run_worker(self._lore_export_worker(), group="personas-io")`.
- **`_lore_export_worker`**: read `book_id = int(self.state.selected_entity_id)`; `data = await asyncio.to_thread(manager.export_world_book, book_id)`; default filename = `_safe_filename(data.get("name") or "world_book") + ".json"`; `EnhancedFileSave(title="Export World Book", filters=Filters(("JSON Files", lambda p: p.suffix.lower()==".json"), ("All Files", lambda p: True)), default_filename=..., context="lore_export")` → `push_screen_wait` → if a path: `await asyncio.to_thread(target.write_text, json.dumps(data, indent=2, ensure_ascii=False), "utf-8")`; notify success; wrap write in try/except → notify on `OSError`; `finally: self._io_dialog_active = False`. (`_safe_filename` strips path separators/control chars — reuse the character-export sanitizer if one exists, else a small local `re.sub(r'[^\w.\- ]', "_", name).strip()`.)

## Data flow

Import: file → `push_screen_wait` path → validate path/size → read → `json.loads` → `normalize_world_book_import` (map + validate, or `ValueError`) → `_unique_lore_name` → `import_world_book(data, name_override)` → `_render_lore_rows("")` + `_select_lore_entry` + notify.

Export: selected book → `export_world_book(id)` → `EnhancedFileSave` path → `json.dumps` → write → notify.

## Error handling

- Import never crashes the UI: path, size, decode, JSON, and shape/field errors are each caught and surfaced via `_notify`; `normalize_world_book_import` validates the entire file before any DB write, so a bad file yields **no partial book**. A late `ConflictError` (name race) is caught.
- Export: `export_world_book` and the file write are wrapped; write failures notify and don't crash.
- Both paths run inside `group="personas-io"` workers with the `_io_dialog_active` re-entrancy guard.

## Testing

- **Adapter (pure unit tests, `Tests/Character_Chat/test_world_book_import.py`):** tldw export passes through unchanged; character-book array form passes through; a SillyTavern World Info object-form (`entries` as `{"0": {...}}` with `key`/`keysecondary`/`order`/int-`position`/`disable`) maps to tldw fields (keys, secondary_keys, insertion_order, string position, `enabled = not disable`); non-dict top-level → `ValueError`; `entries` neither list nor dict → `ValueError`; an entry with empty keys → `ValueError` naming the entry; an entry with empty content → `ValueError`; a non-dict entry → `ValueError`; `caseSensitive`→`case_sensitive`; priority preserved / defaulted.
- **Import (real-DB, `Tests/UI/test_personas_lore.py` with `LorePersonasTestApp` + `lore_db`):** exporting a seeded book to a temp file then importing it recreates the book + entries (priority + selective/secondary_keys/case_sensitive preserved) = round-trip; importing a World Info object-form file creates a book with correctly-mapped entries; importing a malformed file (bad entry) notifies an error and creates **no** book; a name collision imports under a `_unique_lore_name` and the imported book is selected.
- **Export (real-DB):** the Export button on a selected book writes a JSON file whose parsed content re-imports to an equivalent book.
- Full gate: `Tests/Character_Chat/test_world_book_import.py`, `Tests/UI/test_personas_lore.py`, `Tests/Character_Chat/test_world_book_manager.py`, `Tests/Character_Chat/test_world_info.py`, plus `import tldw_chatbook.app`.

## Acceptance criteria

- [ ] `normalize_world_book_import` maps tldw / character-book / SillyTavern World Info shapes to tldw's field set and raises `ValueError` (with a user-facing message) on any invalid shape or empty-keys/empty-content entry — validating the whole file before import (no partial book).
- [ ] The library Import button is visible in Lore mode and imports a `.json` world book from a file picker; a name clash imports under a unique name; the imported book is selected and the list refreshes.
- [ ] Import rejects oversized files, non-JSON, and malformed shapes gracefully (a notification, no crash, no partial book).
- [ ] The Lore detail Settings tab has an Export button that writes the selected book to a chosen `.json` path; export→import is a faithful round-trip (priority + matching fields preserved).
- [ ] No manager change, no schema migration (stays v21); no Markdown export; no legacy-stub retirement; no character-card `character_book` extraction.
- [ ] Full gate green; `import tldw_chatbook.app` OK.
