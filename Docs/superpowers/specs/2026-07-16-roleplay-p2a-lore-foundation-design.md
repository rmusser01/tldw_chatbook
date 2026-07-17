# Roleplay P2a — Lore mode foundation + Try-it diagnostics

**Status:** Design approved (brainstorm), pending spec review.

**Program:** Roleplay (Personas) redesign — P2 (Lore mode), first sub-project. Mirrors the completed Dictionaries mode (P1a foundation + P1b Try-it diagnostics, folded here).

## Why

Lore ("world books") is the next Roleplay mode. Today the Personas "lore" mode is a bare coming-soon placeholder; the only world-book UIs are 2–3 disconnected legacy surfaces (chat sidebar, old CCP tab). The activation engine (`world_info_processor`) matches on keywords and injects content, but exposes **no diagnostics** — you cannot see *which* entries fired, *why*, at what token cost, or which *near-missed*. P2a delivers a local authoring surface (List → Detail → Try-it) and, crucially, the **diagnostics plumbing** that a meaningful Try-it (and every later Lore sub-project) depends on.

## Scope

**In scope (P2a):**
- **Diagnostics engine** in `world_info_processor.py`: `process_messages_with_diagnostics(...)` returning per-entry activation records (fired / near-miss with reason, token cost, injection order, source book), plus a byte-compatible refactor that keeps the live `process_messages` output unchanged.
- **List** (two-panel): all local world books (name, entry count, enabled), New / Duplicate / Delete, enable-toggle.
- **Detail**: Entries tab (per entry: keys, content, position, enabled; reorder via `insertion_order`) and Settings tab (name, description, scan_depth, token_budget, recursive_scanning, enabled).
- **Try-it**: paste sample text (+ optional pull-last-N conversation turns) → run the open book's entries through the diagnostics engine → show injected content by position + the fired/near-miss diagnostics.
- Wire the "lore" mode in `personas_screen.py` to the new widgets; remove "lore" from `_COMING_SOON_MODES`.
- Local-only, over `WorldBookManager` + `world_info_processor` (no new service facade).

**Deferred (later P2 sub-projects — do NOT build here):**
- Entry-level `priority` + priority-aware budget trim (fixes the FIFO hard-break correctness gap) → **P2c**.
- Richer entry editor (selective / secondary_keys / case_sensitive), regex + ReDoS safety, JSON/SillyTavern import-export → **P2d**. *(The diagnostics engine still READS selective/secondary_keys/case_sensitive so near-misses display for existing/imported entries; P2a's editor just doesn't expose editing them.)*
- Conversation attach → **P2e**; character attach → **P2f**.
- Console "what's in play" + native-Console send-path application; fixing the conversation-only legacy-send bug (`world_info_processor` built only when a character is active, `chat_events.py:612`) → **P2g**.
- Retiring the legacy world-book UIs (chat sidebar `chat_right_sidebar.py` + `chat_events_worldbooks.py`; old CCP tab) → later cleanup.
- Constant/always-on and sticky entries → **never** (per the digest; server never really built them).

## Ground truths (verified at dev `9de2a1c5`)

- **Schema (ChaChaNotes_DB, v9–10, stable; DB is at v20):** `world_books` (id, name UNIQUE, description, scan_depth=3, token_budget=500, recursive_scanning, enabled, version, soft-delete); `world_book_entries` (id, world_book_id FK CASCADE, keys JSON, content, enabled, position DEFAULT 'before_char', insertion_order, selective, secondary_keys JSON, case_sensitive, extensions JSON — **no priority, no regex, no group**); `conversation_world_books` junction (conversation_id, world_book_id, priority) — **no character junction** (character lore is embedded `extensions.character_book`).
- **`WorldBookManager(db)`** (`Character_Chat/world_book_manager.py`): `create/get/get_by_name/list(include_disabled)/update/delete_world_book`; `create/get(enabled_only)/update/delete_world_book_entry`; `associate/disassociate_world_book_with_conversation`, `get_world_books_for_conversation`; `export_world_book`, `import_world_book(data, name_override)`. **No reorder API** (update `insertion_order` directly). **No clone** (Duplicate = export + import with `name_override`).
- **`WorldInfoProcessor(character_data=None, world_books=None)`** (`world_info_processor.py`): loads entries at construction, **filtering `enabled=False` at load** (`:63,:89`); merges book settings via `max()` scan_depth/token_budget, OR recursive; applies a book-priority offset to `insertion_order` (`priority*1000`, `:86`).
  - `process_messages(current_message, conversation_history, scan_depth=None, apply_token_budget=True) -> {injections, matched_entries, tokens_used}`. Pipeline: `_build_scan_text` (current + last `scan_depth` history turns) → `_find_matching_entries` (`_entry_matches`: primary key word-boundary match → if `selective` and has secondary_keys, require a secondary match; then recursion re-scanning matched content, `_recursion_depth < 3`, **dedup by `match not in matched` dict-equality**) → `_apply_token_budget` (**FIFO walk, HARD `break` at first over-budget entry** — everything after is dropped) → `_organize_by_position` (groups content by position, **skips empty content**) → `_estimate_tokens` (`len(content)//4`).
  - `format_injections(injections) -> {position: joined_str}`.
  - **`matched_entries` today carries no `id`, `world_book_id`/name, activation_reason, or token_cost** — the diagnostics gap this cycle fills.
  - **Live on the legacy send** at `chat_events.py:1000` (`process_messages` → `format_injections` → positional assembly into the outgoing message). This is why the refactor must be byte-compatible.
- **Dictionaries mode to parallel:** `Widgets/Persona_Widgets/personas_dictionary_detail.py` (List/Detail tabs), `personas_dictionary_tryit.py` (`word_diff` + `_render_diagnostics`: summary strip / fired list / near-miss list — the diagnostics story to mirror, **not** the word-diff), `Chat_Dictionary_Lib.process_user_input_with_diagnostics` → `DictionaryProcessDiagnostics`/`DictionaryEntryDiagnostic` (the data-model template). `personas_screen.py`: `MODE_CHIP_ORDER` (:115), `_COMING_SOON_MODES` (:128), `_MODE_PLACEHOLDER_BODY` (:132), `_CENTER_VIEW_IDS` (:151), `_show_center`, Ctrl+1..4 + `[`/`]` mode cycle.

## Architecture

Three units, each independently testable.

### 1. Diagnostics engine — `world_info_processor.py`

Add a diagnostics data model and an instrumented processing entrypoint; keep `process_messages` byte-identical.

- **Data model** (mirrors `DictionaryProcessDiagnostics`):
  - `LoreEntryDiagnostic(entry_id, source_book_id, source_book_name, keys, activation_reason, status, token_cost, injection_order, position, content_preview, depth_level)` where `status ∈ {"fired", "skipped:disabled", "skipped:secondary", "skipped:budget"}` (an entry whose primary key never matched is simply not reported). `activation_reason` = the primary key (and, for selective, the secondary key) that matched, or the reason it didn't (e.g. `"secondary key not found"`).
  - `LoreProcessDiagnostics(entries: list[LoreEntryDiagnostic], matched, fired, skipped, tokens_used, token_budget, budget_exceeded, books_scanned)`, with `.to_dict()` producing the shape the Try-it widget renders.
- **`process_messages_with_diagnostics(current_message, conversation_history, scan_depth=None, apply_token_budget=True) -> (result_dict, LoreProcessDiagnostics)`** — over an entry set that additionally carries `id`, `world_book_id`, `world_book_name`, and `enabled`, classifies **every candidate**: primary-key match but disabled → `skipped:disabled`; primary match, selective, secondary missing → `skipped:secondary`; matched but past the budget break → `skipped:budget`; matched + within budget → `fired`. `result_dict` is the same `{injections, matched_entries, tokens_used}` the plain path returns.
  - **Decomposed matcher (not the boolean predicate).** `_entry_matches` returns only a bool, so it can't distinguish primary-no-match from secondary-fail or name the matched key. Add a finer classifier — e.g. `_classify_entry_match(entry, scan_text, scan_text_lower) -> (primary_hit: bool, primary_key: str|None, secondary_hit: bool|None, secondary_key: str|None)` built on the existing `_keyword_in_text` — and reimplement `_entry_matches` on top of it (so the plain path's boolean result is provably unchanged). `activation_reason` is derived from this (which key matched; or "secondary key not found").
  - **Empty-content matched entry.** `_organize_by_position` skips empty content (`if content:`), so a matched entry with empty/whitespace content injects nothing but still consumes its (≈0) budget and counts in `matched_entries`. Classify it as `fired` with an empty `content_preview` — do not invent a new status.
  - **Recursion.** Track `depth_level` (0 = direct scan, ≥1 = matched in a prior entry's content); do not track the full triggering-entry chain (YAGNI).
- **Refactor for byte-compatibility (the load-bearing constraint):**
  - **Relocate the enabled filter.** `__init__`/`_process_*_book` must retain *all* entries (tagged `enabled` + source book id/name), rather than dropping disabled ones at load. `process_messages` (plain) filters `enabled` before matching, so its matched set / injections are unchanged; the diagnostics path keeps disabled entries to classify them.
  - **Preserve the plain path's entry-dict shape + recursion dedup.** The plain `process_messages` recursion dedup is dict-equality (`match not in matched`); adding diagnostic keys (`id`/source/enabled) to the dicts the plain path compares would change that. Thread the diagnostic fields so the plain path's matched set is provably unchanged (e.g. the diagnostic fields are ignored by the plain dedup, or the plain path derives from the diagnostics classification while replicating today's dedup exactly).
  - **Regression pin (mirrors P1b):** a corpus of scenarios (disabled entry whose key matches; selective entry with/without secondary; entries that overflow the budget with the hard-break; recursive scanning) asserting the diagnostics path's **fired set + `injections` + `tokens_used` are byte-identical to today's `process_messages`** output. This is the "shown = applied" invariant: the Try-it must show exactly what the send injects.

### 2. Data access — direct `WorldBookManager` + `world_info_processor`

No new service facade in P2a (YAGNI; introduced when scope-dispatch/attach needs it, P2e). The screen calls `WorldBookManager` for CRUD and constructs a `WorldInfoProcessor(world_books=[open_book_with_entries])` for Try-it. Duplicate = `export_world_book(id)` → `import_world_book(data, name_override="<name> (copy)")`. Reorder = `update_world_book_entry(entry_id, insertion_order=…)` per moved entry.

### 3. UI — new Lore widgets, `personas_screen.py` wiring

- **`Widgets/Persona_Widgets/personas_lore_detail.py`** — `PersonasLoreDetailWidget` (I/O-free; emits message classes `LoreBookCreateRequested`, `LoreBookDuplicateRequested`, `LoreBookDeleteRequested`, `LoreBookEnableToggled`, `LoreEntryAddRequested/UpdateRequested/DeleteRequested`, `LoreEntriesReorderRequested`, `LoreBookSettingsSaveRequested`). Two-panel List (DataTable of books) + tabbed Detail (Entries · Settings), paralleling `PersonasDictionaryDetailWidget`.
- **`Widgets/Persona_Widgets/personas_lore_tryit.py`** — `PersonasLoreTryItWidget` (I/O-free). Main result view: **injected content grouped by position** (before_char / after_char / at_start / at_end) — NOT a word-diff. Diagnostics story (mirroring `personas_dictionary_tryit._render_diagnostics`): summary strip (`"N fired · M near-miss · X/Y tokens" [+ "· over budget"]`), fired list (key → content_preview · N tok, in injection order), near-miss list (key — reason: disabled / secondary key not found / dropped by budget). Input: sample text + a "pull last N turns" toggle; scan_depth/token_budget/recursive prefilled from the open book's settings.
- **`personas_screen.py`** — the "lore" mode resolves to the new detail widget (added to `_CENTER_VIEW_IDS`, shown via `_show_center`) with the Try-it mounted alongside (`display=False` until invoked), exactly like Dictionaries. Remove `"lore"` from `_COMING_SOON_MODES`; drop the lore placeholder Static. The screen owns DB I/O in workers (personas-io pattern) and drives the widgets from message handlers. If a CCP glue handler is the established seam (as `ccp_dictionary_handler.py` is for dictionaries), add the Lore analog.

## Data flow (Try-it)

open book → paste sample (+ optional last-N turns) → screen builds `WorldInfoProcessor(world_books=[{**book, "entries": get_world_book_entries(book_id)}])` → `process_messages_with_diagnostics(sample, history)` → widget renders injections-by-position + `diagnostics.to_dict()`.

## Error handling

- `process_messages_with_diagnostics` never raises on bad entry data (mirrors the processor's defensive `_process_entry`): a malformed entry is skipped, never crashes a Try-it or (via the shared predicates) the send.
- Widgets are I/O-free and render defensively (escape user text in DataTable/labels — the P1 `escape_markup` lesson).
- Screen DB workers guard `chachanotes_db` presence and catch per-book errors so one bad book never breaks the List.

## Testing

- **Byte-compat regression pin** (load-bearing): diagnostics fired-set + injections + tokens == plain `process_messages`, over a corpus covering the disabled-relocation and recursion-dedup edges AND the **multi-book + book-priority `insertion_order` offset** case (`priority*1000`) — that is what the *legacy send* actually feeds the engine (conversation world books + embedded character book), so single-book coverage alone would not pin the live behavior. Also pin that plain `matched_entries` keeps the same **count and per-entry shape** the legacy send reads (`chat_events.py:1005`, `current_world_info_count`).
- **Diagnostics classification** (real data): `skipped:disabled`, `skipped:secondary`, `skipped:budget` (hard-break: an entry after the break is budget-skipped even if it would fit), `fired` ordering, source-book attribution, recursion `depth_level`, `content_preview`.
- **Widget tests**: List CRUD (create/duplicate/delete/enable-toggle), Detail Entries add/edit/delete/reorder, Settings save; Try-it rendering (injection-by-position + fired/near-miss lists), including empty state and a hostile-content (markup) entry.
- **Screen wiring**: "lore" mode shows the detail widget (not the placeholder); no longer in coming-soon.

## Acceptance criteria

- [ ] `process_messages` output (injections, matched_entries, tokens_used) is byte-identical before/after the refactor across the regression corpus (disabled, selective/secondary, budget hard-break, recursion).
- [ ] `process_messages_with_diagnostics` classifies every candidate as fired / skipped:disabled / skipped:secondary / skipped:budget with source book, activation_reason, token_cost, injection_order, and depth_level; its fired set equals the plain matched set.
- [ ] Lore List supports create, duplicate (export+import), delete, enable-toggle over `WorldBookManager`.
- [ ] Lore Detail edits entries (keys, content, position, enabled, reorder) and book settings (name, description, scan_depth, token_budget, recursive_scanning, enabled).
- [ ] Try-it shows injected content by position AND the fired/near-miss diagnostics for the open book against sample text (+ optional last-N turns).
- [ ] The Personas "lore" mode shows the real widget, not the coming-soon placeholder; `"lore"` is removed from `_COMING_SOON_MODES`.
- [ ] No entry priority, no richer-field editing, no attach, no Console/native-send wiring, no legacy-UI retirement (all deferred).
- [ ] Full gate suite green; `import tldw_chatbook.app` OK.
