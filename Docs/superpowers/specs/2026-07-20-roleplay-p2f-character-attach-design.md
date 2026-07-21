# Roleplay P2f — character-level world-book (Lore) attach

**Status:** Design.

**Program:** Roleplay (Personas) redesign — P2 (Lore mode), sixth sub-project. Mirrors the merged character-level **dictionary** attach (P1f, PR #645). Follows P2a/P2c/P2d-1/P2d-2/P2d-regex/P2e (all merged); schema **v22**.

## Why

A user can build a standalone Lore book in the Roleplay Lore mode and attach it to a **conversation** (P2e). There is no way to attach one to a **character**, so its entries apply whenever that character is active — the world-book parallel to the merged character-level dictionary attach. This is distinct from a character's *native* embedded `character_book` (a single book baked in at card import): P2f lets a user layer *additional* standalone books onto a character as portable snapshots.

## What already exists (verified at dev tip `516461b1`)

- **Native character world book:** `extensions['character_book']` (CCv3 field remapped at import in `Character_Chat_Lib.parse_v2_card` :1272-1279; shape from `parse_character_book` :1009-1071). It is **already** fed to the send path — `chat_events.py` :867-916 reads `active_char_data['extensions']['character_book']` and passes `character_data=active_char_data` to `WorldInfoProcessor`.
- **Runtime merge already unifies both shapes:** `WorldInfoProcessor` (`world_info_processor.py`) processes a character book (`_process_character_book` :113-144) and standalone `world_books` list entries (`_process_world_books` :146-189) through the **same** `_process_entry` (:191-237) and `_make_candidate` (:239-260). It accepts `character_data=` and `world_books=` simultaneously and needs **no change** for P2f. The P2d-regex load-time backstop lives in `_process_entry`, so it protects character-attached entries automatically.
- **Send-path sources today:** (a) conversation-attached books via `WorldBookManager.get_world_books_for_conversation` (the `conversation_world_books` junction, P2e), (b) the native `character_book`. There is **no** global/always-on source, and the current union is **not deduped by name**.
- **`update_character_card` accepts arbitrary `extensions`:** `extensions` is in `_CHARACTER_CARD_JSON_FIELDS` (`ChaChaNotes_DB.py` :4130), serialized whole and optimistic-locked (version check :4592-4604 + `WHERE ... version = ?` :4663). **No schema migration needed.**
- **`WorldBookManager` has ZERO character methods** — only standalone CRUD + the P2e conversation junction (`world_book_manager.py`; `export_world_book` :690, `import_world_book` :735, `list_world_books` :191).

## The P1f dictionary precedent to mirror (verified)

- **Storage:** `extensions['chat_dictionaries']` = a list of embedded content snapshots (each an `export_json` `data` block), written only by explicit attach. `LocalChatDictionaryService.attach_to_character` :939-973 / `_write_embedded_dictionaries` :930-937 (read-modify-write `extensions` under `expected_version`) / `detach_from_character` :975-1007 / `list_character_dictionaries` :1009-1038; reader `load_character_dictionaries` :1228-1298 (dedup by name, first-wins :1272-1281).
- **Send-path union:** `collect_active_chatdict_entries` :1367-1400 + `_resolve_active_dictionaries` :1301-1364 — additive union of conversation dicts + character-embedded dicts, dedup by name, **conversation wins**, enabled-only, never raises; a same-named character dict is `shadowed` only by an **enabled** conversation dict.
- **Editor coherence:** `sync_attached_dictionaries(blocks, new_version)` on the editor (`personas_character_editor_widget.py` :349-367) + `_sync_character_editor_dictionaries` (`personas_screen.py` :1990-2015) — patch only `extensions['chat_dictionaries']` + `version` on the editor's Save-base so an out-of-band attach isn't clobbered by a later Save.
- **UI:** `PersonasCharacterDictionariesWidget` (`Widgets/Persona_Widgets/personas_character_dictionaries.py` :37-149) — docked-bottom in `#personas-detail-stack` (`personas_screen.py` :405-416, mounted :535), list + "Attach dictionary…"/"Detach" posting `CharacterDictionaryAttachRequested`/`CharacterDictionaryDetachRequested(name)`. Handlers `_handle_character_dictionary_attach`/`_character_dictionary_attach_worker` :1910-1966 (uses `DictionaryPicker` + `_list_attachable_dictionaries`), `_handle_character_dictionary_detach` :2017-2047.

## Design decisions (locked with user)

1. **Storage = embedded snapshots in a NEW key** `extensions['character_world_books']` (a list, idempotent by name). The native `extensions['character_book']` is left untouched.
2. **Precedence = dedup by name, conversation wins.** A character-attached snapshot is dropped if an **enabled** conversation-attached book has the same name.
3. **UI = a sibling docked widget** (`PersonasCharacterWorldBooksWidget`) mirroring the dict widget, **with a mandatory geometry check** that the two stacked bottom-docked panels do not clip the character card/editor.
4. **Native `character_book` stays additive** (never newly deduped) — existing send behavior preserved byte-identical.

## Behavior-change framing

Additive: a new `extensions` key, a new service surface, a new docked UI panel, and one small legacy-send-path addition (read the new key + dedup vs conversation books). For a character with no `character_world_books` key, the send path is **byte-identical** to today. No schema change. No `WorldInfoProcessor` change. Console "what's in play" + native-Console send + legacy-UI retirement → P2g.

## Ground truths

- `extensions` is arbitrary JSON, whitelisted + optimistic-locked; caller owns read-modify-write coherence (read full `extensions`, mutate one key, write the whole dict). Character ids are **ints** (no P2e string/int hint mismatch).
- The snapshot is `export_world_book(id)` output — a self-contained book+entries block. **Load-bearing seam:** the resolver must map this snapshot to the exact book-dict shape `_process_world_books` consumes, carrying every matcher field — `keys`, `content`, `secondary_keys`, `position`, `insertion_order`, `case_sensitive`, `selective`, `priority`, **`regex`**, `enabled` — and the book-level `name`/`enabled`. Character-attached books have no conversation-junction association priority, so the resolver maps them with book-level `priority` 0 (`_process_world_books` offsets `insertion_order` by `book.get("priority", 0) * 1000`; per-entry priority from P2c rides on each entry and is honored separately). The plan verifies `export_world_book`'s shape against `get_world_books_for_conversation`/`_process_world_books` and converts if they differ.
- **Untrusted-content is the crash-class** (P1f Critical, recurring): embedded snapshots may come from imported cards. Every read needs guards; a crafted card with two same-named blocks must never reach `DataTable.add_row(key=name)` twice (that is `DuplicateKey` → worker `exit_on_error` → **app exit**).

## Architecture

### 1. Storage (`extensions['character_world_books']`)
A list of `export_world_book` snapshot blocks. Idempotent by name (attach is a no-op if a block with that name is already present). Written only by the attach service via `update_character_card({"extensions": ext}, expected_version=...)`.

### 2. Service (`WorldBookManager`, net-new methods)
- `attach_world_book_to_character(self, world_book_id: int, character_id: int) -> Dict[str, Any]` — snapshot the book via `export_world_book`, read the character record, append the block iff no same-named block exists, write back under `expected_version`. Returns `{"character_id": int, "name": str, "attached": bool}` (`attached=False` when it was already present). Mirrors `LocalChatDictionaryService.attach_to_character`.
- `detach_world_book_from_character(self, character_id: int, name: str) -> Dict[str, Any]` — remove the block whose (normalized) name matches; write back under `expected_version`. Returns `{"character_id": int, "name": str, "detached": bool}`.
- `get_world_books_for_character(self, character_id: int) -> List[Dict[str, Any]]` — return `[{"name": str, "entry_count": int}]` summaries for the panel (name normalized, entries `isinstance(list)`-guarded, **deduped by name** at read so a hostile card can't produce duplicate rows). I/O (reads the character record).
- All three read/normalize `extensions` defensively (`isinstance(dict)`); never raise on malformed embedded data (worst case: attach/detach no-op, list returns `[]`).

### 3. Send-path resolver (pure, testable) + union
- New pure function `resolve_character_world_books(char_data: Optional[dict], exclude_names: Set[str]) -> List[Dict[str, Any]]` (home: `world_book_manager.py` module-level, no DB): read `char_data['extensions']['character_world_books']`, **dedup by name** (first-wins), drop blocks whose book-level `enabled` is false or whose name is in `exclude_names`, and return each survivor mapped to the `_process_world_books` book-dict shape (per the §Ground-truths seam). Never raises; malformed block → skipped.
- `chat_events.py` change (only new send-path code): the conversation `world_books` list is already fetched `enabled_only=True`, so `exclude_names = {str(b.get("name")) for b in world_books}` is exactly the set of enabled conversation-book names. Then `world_books += resolve_character_world_books(active_char_data, exclude_names)`. The native `character_book` continues to flow via `character_data=` unchanged. Net effect: character-attached books apply, a name-collision with an enabled conversation book is won by the conversation, the native book is untouched.

### 4. Picker (`WorldBookPicker`, net-new)
A `WorldBookPicker(ModalScreen[int | None])` mirroring `DictionaryPicker`: lists standalone books via `WorldBookManager.list_world_books`, **excludes books already attached to this character by name**, search-filterable, returns the picked `world_book_id` (int) or `None`. (The P2e `ConversationAttachPicker` is conversation-specific and is NOT reused.)

### 5. UI (`PersonasCharacterWorldBooksWidget`, net-new)
Mirror `PersonasCharacterDictionariesWidget`: a `Container` with a header Static ("World Books (embedded copies)"), an empty-state Static, a `DataTable(id="personas-char-worldbooks-table")` (columns `world book`, `entries`), and "Attach world book…"/"Detach" buttons posting `CharacterWorldBookAttachRequested` / `CharacterWorldBookDetachRequested(name: str)`. **I/O-free**: `load_world_books(rows)` renders given `[{"name", "entry_count"}]`; `DataTable` row `key=name` (rows already deduped by the service, but the widget also guards against a duplicate key defensively). Mounted docked-bottom in `#personas-detail-stack` alongside the dict widget.
- **Geometry check (mandatory):** a plan task verifies at 80×24 and a larger size that both stacked bottom-docked panels (dict + world-books, each `max-height` capped) render without clipping the character card/editor; adjust the `max-height`s if needed.

### 6. Screen handlers + editor coherence
Mirror P1f: `@on(CharacterWorldBookAttachRequested) _handle_character_worldbook_attach` (+ `_character_worldbook_attach_worker` using `WorldBookPicker` + a `_list_attachable_world_books` that excludes already-attached), `@on(CharacterWorldBookDetachRequested) _handle_character_worldbook_detach`, and `_refresh_character_worldbooks` (re-list into the widget). Each attach/detach also calls `_sync_character_editor_worldbooks` → `sync_attached_world_books(blocks, new_version)` on the editor (re-fetch the record, patch only `extensions['character_world_books']` + `version` on the editor's Save-base). All DB I/O off-thread (`asyncio.to_thread`), wrapped → `_notify`, re-entrancy-guarded (`_io_dialog_active` / `group="personas-io"`), guarded on a selected character.

### 7. Send path / processor — no change beyond §3
`WorldInfoProcessor` is untouched; character-attached books ride the existing `world_books` list.

## Error handling / untrusted-content guards

- **Dedup-by-name at BOTH panel render and resolver parse** (the DuplicateKey app-exit crash-class). Service list + resolver both dedup first-wins.
- Every read of embedded content is guarded: name via `str(x or "(untitled)")`; entries via `isinstance(list)`; per-entry/book booleans (`enabled`/`case_sensitive`/`selective`/`regex`) via the existing `_coerce_bool`; ids/priority via `_coerce_int`.
- Attach is idempotent by name (no duplicate). Detach on an absent name is a harmless no-op. Missing/unsaved selected character → handlers no-op gracefully. Optimistic-lock `ConflictError` on write → `_notify`, never crash.
- Resolver and service **never raise** on malformed embedded data.

## Testing (real-DB / real-integration, no fakes)

- **Service:** attach embeds a snapshot into `extensions['character_world_books']`; idempotent by name (second attach of same name is a no-op); detach removes by name; `get_world_books_for_character` summarizes; snapshot carries all matcher fields incl. `regex`/`secondary_keys`/`priority`; a hostile record with two same-named blocks → list returns a single deduped row (no crash).
- **Resolver (pure):** character-attached enabled book is returned in processor shape; a name-collision with an enabled conversation book (via `exclude_names`) is dropped (conversation wins); a **disabled** conversation book does NOT exclude (name not in the enabled set); a book-level-disabled snapshot is dropped; malformed/duplicate blocks never raise and are deduped.
- **Send-path integration:** with a character-attached book and no conversation book, `WorldInfoProcessor` (built the way `chat_events` builds it) fires the attached book's entries; with a same-named enabled conversation book, only one fires (conversation's).
- **Widget (I/O-free):** empty-state vs render toggle; Attach posts `CharacterWorldBookAttachRequested`; select + Detach posts `CharacterWorldBookDetachRequested(name)`; a duplicate-key row set does not crash `add_row`.
- **Screen (real-DB):** attach via monkeypatched picker persists a snapshot + refreshes the panel + patches the editor base (no clobber on a subsequent Save, no `ConflictError`); detach removes it.
- **Geometry:** the layout check above.
- **Full gate:** `test_world_book_manager`, the new resolver/widget/screen tests, plus `import tldw_chatbook.app`.

## Decomposition

Single sub-project mirroring P1f (#645), one PR. No migration (schema stays v22). Console "what's in play" + native-Console world-info send + legacy world-book UI retirement → P2g.

## Acceptance criteria

- [ ] `WorldBookManager` gains `attach_world_book_to_character` / `detach_world_book_from_character` / `get_world_books_for_character`, storing snapshots in `extensions['character_world_books']`, idempotent by name, under optimistic lock; no schema change.
- [ ] A pure `resolve_character_world_books(char_data, exclude_names)` returns deduped, enabled, processor-shaped character-attached books; drops names present among enabled conversation books; never raises on malformed data.
- [ ] `chat_events.py` unions character-attached books into the send path with **conversation-wins** name dedup; the native `character_book` and the no-attachment case are byte-identical to today.
- [ ] A `WorldBookPicker` returns a picked `world_book_id` and excludes already-attached books; a `PersonasCharacterWorldBooksWidget` (I/O-free) lists/attaches/detaches and posts the two messages; both dedup by name so a hostile card cannot crash the DataTable.
- [ ] Screen attach/detach persist to `extensions` and keep the editor Save-base coherent (no clobber, no `ConflictError`); all DB I/O off-thread, guarded, never crashes.
- [ ] The two stacked bottom-docked character panels render without clipping the card/editor (geometry-verified).
- [ ] Snapshots carry all matcher fields incl. `regex`; the P2d-regex load-time backstop protects character-attached entries.
- [ ] Full gate green; `import tldw_chatbook.app` OK. Console side deferred to P2g.
