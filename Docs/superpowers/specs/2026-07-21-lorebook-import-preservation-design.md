# TASK-429 — Character-card import preserves the embedded lorebook

- **Date:** 2026-07-21
- **Task:** TASK-429 (from the RP/character-card UX review `Docs/superpowers/qa/rp-ux-review-2026-07-21/report.md`)
- **Branch base:** origin/dev (schema v22)

## Problem

Importing a V2 character card with an embedded `character_book` leaves it **invisible and unmanaged** in the app. The card's lorebook is parsed and stored under `extensions['character_book']` (`Character_Chat_Lib.py:1272-1279`), a *different* key from the app's managed world-book format `extensions['character_world_books']`. Consequences observed in the review: the character's "attached world books" panel shows "No world books attached" and Lore mode shows "No lore books yet" — the user can't see, toggle, or edit the imported book.

(Note: the book *does* inject at send-time today via `WorldInfoProcessor._extract_character_book` reading `extensions['character_book']` — so this is a UI-visibility + manageability gap, not a total runtime loss. That existing send-path read is why the double-injection guard below matters.)

## Goal (acceptance criteria)

1. Importing a V2 card with a `character_book` results in that lorebook existing in the app and **attached to the imported character** (visible/editable in the character's world-books panel).
2. Import **surfaces what happened** — a toast naming the imported book and its entry count.
3. **Round-trip:** exporting the character and re-importing preserves the book.

## Scope

**In scope:** at import time, convert a V2 card's `data.character_book` into one embedded `character_world_books` snapshot block on the saved character, and surface a toast. Import-only.

**Out of scope (settled during design):**
- A standalone Lore-mode `world_books` DB row (the `world_books.name` UNIQUE constraint makes multi-card / re-import conflict-prone; the P2f character-level architecture is embedded snapshots). Chosen: embedded snapshot only.
- Export changes — `export_character_card_to_json` copies `extensions` verbatim (`Character_Chat_Lib.py:3342-3344`) and `parse_v2_card` copies it back, so `character_world_books` round-trips automatically. **AC#3 needs no export work.**
- A migration/backfill of characters imported before this fix (their `extensions['character_book']` still injects at runtime as it does today; they simply remain UI-unmanaged). No DB migration (extensions is a JSON blob, schema v22).

## Design

### 1. Lenient converter (new, in `world_book_import.py`)

Add `character_book_to_world_book_block(book: Any, fallback_name: str) -> tuple[dict | None, int, int]` returning `(block_or_None, imported_count, skipped_count)`:

- If `book` is not a dict → return `(None, 0, 0)`.
- Book-level fields (coerced, reusing this module's `_coerce_int`/`_coerce_bool`): `name` (use `book["name"]` if a non-empty string, else `fallback_name` — the resolver silently skips empty-named blocks, `world_book_manager.py:113`), `description` (str), `scan_depth` (`_coerce_int`, default 3), `token_budget` (`_coerce_int`, default 500), `recursive_scanning` (`_coerce_bool`, default False), `enabled` (default True — V2 `character_book` has no book-level `enabled`).
- Entries: iterate `book.get("entries")` (list, or `.values()` if a dict, else `[]`). For each, call the existing `_normalize_entry(entry, i)` in a `try/except ValueError`: on success append the normalized entry (already in snapshot shape: keys/secondary_keys/content/insertion_order/position/selective/case_sensitive/enabled/priority/extensions/regex); on `ValueError` (entry not a dict, no keys, or no content — genuinely inert) increment `skipped_count` and log a warning naming the entry index. `imported_count` = len of salvaged entries.
- The block's `entries` = salvaged list. **Never raises.**

This reuses the tested normalization (int→string `position`, `disable`→`enabled`, `keysecondary`/`caseSensitive` aliases, regex validation) but is salvage-and-count rather than all-or-nothing, so one bad entry doesn't sink the book.

### 2. Convert-and-replace at parse time (`Character_Chat_Lib.parse_v2_card`, ~:1272-1279)

Replace the current `extensions['character_book'] = parse_character_book(...)` handling with:

- If `data_node['character_book']` is a dict: build the block via `character_book_to_world_book_block(data_node['character_book'], fallback_name=f"{parsed_data.get('name') or 'Character'} Lorebook")`.
- If a block is produced, **merge into `extensions['character_world_books']`**: coerce the existing value to a list, append the block **only if no existing block shares its name** (string-compared, matching the manager's dedup), and store back.
- **Do NOT set `extensions['character_book']`.** Additionally `pop('character_book', None)` from `extensions` (defensive: the verbatim `extensions` copy at `:1268` could carry a nested `extensions.character_book` from a hand-crafted source; leaving it alongside the new `character_world_books` would double-inject at send-time since `_collect_active_world_books` unions both with no cross-dedup — `world_info_resolver.py:47-63`).
- Log the imported/skipped counts. Guard the whole block so malformed input never raises out of `parse_v2_card`.

`parse_character_book` (`:1009`) stays (still used by `validate_character_book`); its output is simply no longer written into the saved extensions.

### 3. Feedback (AC#2) — Personas import toast

In `_import_character_from_path` (`UI/Screens/personas_screen.py:3762`, success toast at `:3806`): after a successful import (character id in hand), re-read the saved character's `extensions['character_world_books']` (via the same handler/db used for import). If a block is present, append to the toast, e.g. `"Character imported. Lorebook '{name}' attached ({N} entries)."` (N = len of the block's entries). When no book is present, the toast is unchanged. Per the design decision, the toast reports the book name + imported entry count (the converter also logs any skipped inert entries; explicit skipped-count in the toast is out of scope).

## Data flow

```
Import V2 card (SillyTavern-style, data.character_book present)
  → parse_v2_card: character_book_to_world_book_block(...) → 1 snapshot block
      → extensions['character_world_books'] += [block]  (dedup by name)
      → extensions.pop('character_book')                (no double-inject)
  → db.add_character_card(extensions=...)
Personas import handler
  → re-read saved character → toast "Lorebook '{name}' attached ({N} entries)"
Send-time
  → resolve_character_world_books reads character_world_books → injects ONCE
Export → JSON/PNG copies extensions verbatim → re-import passes the block through (AC#3)
```

## Testing

- **Converter unit** (`Tests/Character_Chat/test_world_book_import.py`): `character_book_to_world_book_block` on a V2 `character_book` with mixed entries — produces a block whose entry shape matches the snapshot fields; int `position` → string; a no-keys and a no-content entry are skipped and counted; empty book name → fallback; non-dict input → `(None, 0, 0)`; never raises.
- **Import integration** (`Tests/Character_Chat/test_character_file_operations.py`, extend `sample_character` with a V2-with-`character_book` fixture): after `import_and_save_character_from_file`, the saved character's `extensions['character_world_books']` has the block with the salvaged entries, and `extensions` has **no** `character_book` key (no double-inject).
- **Round-trip** (same file, mirror `test_reimport_exported_character`): import a card with a book → export to JSON/PNG → re-import → the `character_world_books` block survives unchanged.
- **No double-injection** (a send-path or resolver test): a character carrying the converted block injects each entry once (only `character_world_books`, no `character_book`).
- **Toast** (`Tests/UI/test_personas_*` import test, if a harness exists; else assert the handler computes the book-name+count string): a successful import with a book yields a toast naming the book and its entry count.
- **Live-verify:** import the review's SillyTavern-style card in the real TUI; confirm the book appears in the character's attached-world-books panel with its entries, the toast names it, and a chat still injects the lore once.

## Risks / mitigations

- **Double-injection** (the main correctness risk) → convert-and-replace: write `character_world_books`, strip `character_book` (top-level not stored + `extensions.pop`).
- **Silently-dropped entries** → the converter salvages leniently (defaults everything but keys+content) and logs skips; only genuinely-inert entries are dropped.
- **Empty book name silently skipped by the resolver** → fallback name `"{character} Lorebook"`.
- **Malformed imported content** → converter never raises; guarded read in `parse_v2_card` (the recurring "guard every read of imported card content" lesson).
- **Re-export/re-import duplication** → dedup by name when merging into `character_world_books`.

## Non-goals

- Standalone Lore-mode book / `world_books` DB row.
- Export changes; schema migration; backfill of previously-imported characters.
- Surfacing an explicit skipped-entry count in the toast (logged only).
