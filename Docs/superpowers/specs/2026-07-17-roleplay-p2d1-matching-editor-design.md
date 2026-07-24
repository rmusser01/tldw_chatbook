# Roleplay P2d-1 — surface selective / secondary_keys / case_sensitive in the Lore entry editor

**Status:** Design.

**Program:** Roleplay (Personas) redesign — P2 (Lore mode). First half of P2d (the "richer Lore entry editor + import/export" sub-project), split into P2d-1 (matching-controls editor) and P2d-2 (import/export UI). Follows P2c (entry priority + priority-aware budget, PR #682 merged; schema now v21).

## Why

The Lore entry editor (`PersonasLoreDetailWidget`) lets a user edit an entry's keys, position, priority, enabled flag, and content — but **not** its three matching controls (`selective`, `secondary_keys`, `case_sensitive`), even though those fields already exist and work end-to-end in the schema, the processor, and the manager. A user can't configure case-sensitive matching or selective (require-a-secondary-key) matching from the redesigned editor; they can only be set by import or by the legacy UI. P2d-1 closes that gap.

## Scope

**In scope:** editing controls for `selective`, `secondary_keys`, and `case_sensitive` in the entry form, threaded through `entry_form_payload()`, `_fill_form_from_entry()`, and the screen's add handler; a visual "secondary keys inactive" hint that never destroys stored data.

**Explicitly NOT in scope (deferred):**
- **Regex matching** — the only genuinely new matching capability; deferred to its own sub-project (needs storage + a ReDoS-safe branch in the live matcher). Not touched here.
- **Import/export UI** — P2d-2.
- **No new DataTable column** for matching flags (keeps the entries table narrow; the form is the source of truth).
- Conversation attach (P2e), character attach (P2f), Console "what's in play" + native-send (P2g).

## Behavior-change framing

This is a **pure UI surfacing** change over capability that already exists and is already exercised on the live send path. It adds **no** schema migration (stays v21), **no** processor/matcher code, and **no** manager code. The only behavior change is that the editor can now write three fields it previously left at their defaults; entries created/edited before P2d-1 are unaffected (their stored `selective`/`secondary_keys`/`case_sensitive` values are already honored by the matcher and are simply now visible/editable).

## Ground truths (verified at dev tip, post-#682)

- **Schema** (`ChaChaNotes_DB.py:1194`, `world_book_entries`, v21): already has `selective BOOLEAN DEFAULT 0`, `secondary_keys TEXT` (JSON array), `case_sensitive BOOLEAN DEFAULT 0`. No `regex` column.
- **Processor** (`world_info_processor.py`): `_entry_matches` already honors `case_sensitive` (via `_keyword_in_text`, word-boundary `\b…\b` matching), `selective`, and `secondary_keys` (a selective entry with secondary keys requires at least one secondary hit; a selective entry with **no** secondary keys falls back to primary-only). No change needed.
- **Manager** (`world_book_manager.py`):
  - `create_world_book_entry(..., selective: bool = False, secondary_keys: Optional[List[str]] = None, case_sensitive: bool = False, ...)` — already accepts all three.
  - `update_world_book_entry(**kwargs)` — its field loop already includes `'selective'`, `'secondary_keys'`, `'case_sensitive'` (`secondary_keys` is JSON-encoded in the `['keys','secondary_keys','extensions']` group; empty list → `NULL`).
  - `get_world_book_entries` — already returns `selective` (bool), `secondary_keys` (list), `case_sensitive` (bool) in each row dict.
- **Widget** (`personas_lore_detail.py`): entry form composed at `:114-124` (one Horizontal row: keys / position / priority / enabled, then a content `TextArea`); `entry_form_payload()` at `:271`; `_fill_form_from_entry()` at `:337`; `on_mount` at `:145` (registers DataTable columns). None of the three matching fields are exposed.
- **Textual `Switch` behavior** (verified against installed Textual): `Switch.Changed` carries `.value` (bool) and `.switch`. Textual does **not** fire `Switch.Changed` when a switch is set to the value it already holds — the widget already compensates for this for the book-enabled switch via `_set_enabled_switch` (`:221-237`, comment at `:231`), and has a scoped `@on(Switch.Changed, "#personas-lore-enabled")` handler at `:426`.
- **Screen** (`personas_screen.py`): `_handle_lore_entry_add` calls `create_world_book_entry` with an **explicit kwarg list** (keys, content, enabled, position, insertion_order, priority) — so any field not named is silently dropped (this is the exact bug fixed for `priority` in P2c). `_handle_lore_entry_update` calls `update_world_book_entry(int(entry_id), **payload)`, so it picks up any new payload key automatically.

## Architecture

### 1. Form controls (`personas_lore_detail.py`, compose)

Add **two rows** immediately after the existing keys/position/priority/enabled row and **before** the content `TextArea` (grouping all key/matching config above the content area), to avoid crowding at the QA resolution (2050×1240) and give secondary keys room:

- **Row 2** (`Horizontal`, `classes="personas-lore-form-row"`): a labeled Case-sensitive switch and a labeled Selective switch. The Settings tab already uses `Static + Switch` inside `personas-lore-form-row` (`:136-141`), so this reuses a proven layout.
  - `Static("Case-sensitive", markup=False)` + `Switch(value=False, id="personas-lore-entry-case-sensitive")`
  - `Static("Selective", markup=False)` + `Switch(value=False, id="personas-lore-entry-selective")`
- **Secondary-keys input**: a full-width `Input(placeholder="Secondary keys (comma-separated)", id="personas-lore-entry-secondary-keys")` yielded as a **direct child** of the entries `VerticalScroll` (like the content `TextArea` and the Name input at `:132`), **not** wrapped in a single-child Horizontal — a lone child in a horizontal form-row does not fill the row cleanly.

Resulting entries-tab child order: DataTable → Row 1 (keys/position/priority/enabled) → Row 2 (case-sensitive/selective) → secondary-keys `Input` → content `TextArea` → button row.

Both switches default `False` (matching the schema defaults `selective`/`case_sensitive` DEFAULT 0).

### 2. `entry_form_payload()` (`personas_lore_detail.py:271`)

After the existing keys/content guard, read the three controls and add them to the returned dict:

- `case_sensitive = bool(query_one("#personas-lore-entry-case-sensitive", Switch).value)`
- `selective = bool(query_one("#personas-lore-entry-selective", Switch).value)`
- `secondary_keys = [k.strip() for k in query_one("#personas-lore-entry-secondary-keys", Input).value.split(",") if k.strip()]` (same never-raise comma-split as `keys`)

**Data fidelity:** `secondary_keys` is read and stored **regardless** of the Selective switch state, so toggling Selective off does not erase typed secondary keys (they persist in the DB and are simply inactive in matching until Selective is on). Update the docstring's field list to include `case_sensitive`, `selective`, `secondary_keys`.

### 3. `_fill_form_from_entry()` (`personas_lore_detail.py:337`)

Populate the three controls from the entry dict (never raises; missing → default):

- `#personas-lore-entry-case-sensitive`.value = `bool(entry.get("case_sensitive", False))`
- `#personas-lore-entry-selective`.value = `bool(entry.get("selective", False))`
- `#personas-lore-entry-secondary-keys`.value = `", ".join(str(k) for k in (entry.get("secondary_keys") or []))`

Setting the Selective switch value here re-fires `Switch.Changed`, which re-syncs the secondary-keys disabled hint (see §5).

### 4. Add handler threads the three fields (`personas_screen.py::_handle_lore_entry_add`)

Add three explicit kwargs to the `create_world_book_entry` call (the P2c-priority lesson — the explicit kwarg list drops anything not named):

- `selective=payload.get("selective", False)`
- `secondary_keys=payload.get("secondary_keys", [])`
- `case_sensitive=payload.get("case_sensitive", False)`

`_handle_lore_entry_update` needs **no change**: it already forwards `**payload`, and the manager's update field-loop already handles the three fields.

### 5. Secondary-keys "inactive" visual hint (`personas_lore_detail.py`)

`secondary_keys` only affects matching when `selective` is on. Disable the secondary-keys `Input` (grays it) when Selective is off, as a pure visual hint — the stored value is preserved either way (§2).

**Use a helper, not the event alone.** Textual does **not** fire `Switch.Changed` when a switch is programmatically set to the value it already holds — the widget already works around this exact hazard for the book-enabled switch via `_set_enabled_switch` (`:221-237`). So relying only on a `Switch.Changed` handler would leave the disabled state stale when `_fill_form_from_entry` sets Selective to a value it already had. Instead:

- Add `_sync_secondary_keys_disabled()`: sets `#personas-lore-entry-secondary-keys`.disabled = `not #personas-lore-entry-selective`.value.
- Call it from **`on_mount`** (initial state — Selective defaults off → input starts disabled), from **`_fill_form_from_entry`** (after setting the Selective switch, so the state is correct even when `Changed` doesn't fire), and from the **`@on(Switch.Changed, "#personas-lore-entry-selective")`** handler (`event.stop()`; handles live user toggles). The widget already has a scoped `@on(Switch.Changed, "#personas-lore-enabled")` handler (`:426`) for the book-enabled switch, so a second scoped handler for a different id follows the established pattern and cannot collide.
- A disabled `Input` still returns its `.value` on query, so `entry_form_payload()` reads it correctly even while disabled — the stored value is never lost.

## Data flow

Editor form → `entry_form_payload()` (adds `case_sensitive`/`selective`/`secondary_keys`) → `LoreEntryAddRequested` / `LoreEntryUpdateRequested` → screen handler → `WorldBookManager.create_world_book_entry(...)` (add, explicit kwargs) or `update_world_book_entry(**payload)` (update) → DB. On selection, `get_world_book_entries` → `_fill_form_from_entry()` repopulates the three controls. The live matcher (`_entry_matches`) already reads these fields — no processor change.

## Error handling

- Switches yield booleans; the secondary-keys parse is a never-raising comma-split (empty → `[]`). The `entry_form_payload` never-raise contract is preserved.
- The disabled-input hint cannot lose data (value is read regardless of disabled state).
- No new failure surface on the send path (no matcher change).

## Testing

Real-DB + real-widget, per program rules (`HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=…/.local/share .venv/bin/python -m pytest … -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread`):

1. **Widget payload round-trip** (`Tests/UI/test_personas_lore.py`): setting the three controls makes `entry_form_payload()` return `case_sensitive`/`selective`/`secondary_keys` correctly (incl. comma-split and empty → `[]`).
2. **Fill-from-entry**: `_fill_form_from_entry` populates all three from a persisted entry (incl. `secondary_keys` join, and an entry with `selective=False` but stored secondary keys — the data-fidelity case).
3. **Real add-handler regression** (the P2c-lesson guard): driving `_handle_lore_entry_add` with a payload carrying the three fields persists them via the real screen handler + real `WorldBookManager` (would fail if the explicit kwargs were dropped).
4. **Update round-trip**: `_handle_lore_entry_update` persists changes to the three fields via `**payload`.
5. **Secondary-keys hint**: the input is disabled when Selective is off and enabled when on — verified on mount, after a live toggle, and after selecting an entry via `_fill_form_from_entry` (the case where `Switch.Changed` does not fire because the value is unchanged, which the `_sync_secondary_keys_disabled()` helper covers). Its stored value survives a toggle-off (fidelity).
6. **One integration test** (spans widget→DB→matcher, not a broad matcher re-test): an entry created through the real add-handler path with `selective=True` + secondary keys gates matching in `WorldInfoProcessor` (fires only when a secondary key is present in scan text), proving the surfaced config actually changes matching behavior.

Full gate: `Tests/UI/test_personas_lore.py`, `Tests/Character_Chat/test_world_book_manager.py`, `Tests/Character_Chat/test_world_info_diagnostics.py`, `Tests/Character_Chat/test_world_info.py`, plus `import tldw_chatbook.app`.

## Acceptance criteria

- [ ] The Lore entry editor exposes editable Case-sensitive and Selective switches and a Secondary-keys input.
- [ ] `entry_form_payload()` returns `case_sensitive` (bool), `selective` (bool), and `secondary_keys` (list); `secondary_keys` is stored as typed regardless of the Selective switch state.
- [ ] `_fill_form_from_entry()` repopulates all three controls from a persisted entry, including an entry whose `selective=False` but has stored secondary keys.
- [ ] Creating an entry through the real screen add-handler persists all three fields (regression against a dropped kwarg); updating an entry persists changes to all three.
- [ ] The Secondary-keys input is disabled (visual hint) when Selective is off and enabled when on, without ever losing its stored value.
- [ ] A `selective=True` entry created through the editor path gates matching in `WorldInfoProcessor` as expected.
- [ ] No schema migration (stays v21), no processor/matcher change, no manager change, no new DataTable column.
- [ ] Full gate green; `import tldw_chatbook.app` OK.
