# Roleplay P3a — Characters/Personas Library: pagination + sort + tag filter

**Program:** Personas → "Roleplay" workbench redesign. Sub-project **P3a** — the first of the **P3 (Characters/Personas flow polish)** cycle, which is the final phase of `P0 → P1 → P2 → P3`. P1 (Dictionaries) and P2 (Lore) are complete and merged. See the northstar `Docs/superpowers/specs/2026-07-13-roleplay-p0-reframe-northstar-design.md`.

**Date:** 2026-07-21
**Status:** Design (awaiting user review before writing-plans)
**Schema:** ChaChaNotes v22 — **P3a adds NO migration** (no new columns; reads existing `character_cards`).

---

## Goal

Make the shared Roleplay library list usable at scale and easier to search. Today the character list mounts every row at once (no virtualization), caps at 1,000 rows, has a hardcoded `ORDER BY name`, a denominator-less count, and a tag-blind capped-50 search. P3a introduces **page-at-a-time pagination**, a **sort control**, and a **tag filter** across the two list backends behind the shared `PersonasLibraryPane`.

## Non-goals (explicitly out of scope)

- **No architectural convergence** — the Characters mode keeps its card-view ⇄ editor-view swap + live-chat preview. We are NOT rebuilding it to the Dict/Lore unified-tabbed-detail shape (user decision: "targeted polish, keep shape").
- **No editor changes** — avatar preview, alternate-greetings UX, validation, dirty-tracking re-arm are **P3b** (the next cycle), not P3a.
- **No Dictionaries/Lore behavior change** — their lists are tiny; the new controls are gated OFF for those modes.
- **No new schema, no migration.**
- **No deterministic Try-it preview** (a separate deferred item; user did not select it).

## Success criteria

- A user with thousands of characters sees a fast, paginated list ("◄ 1–50 of N ►") that mounts only one page of rows at a time.
- The list can be sorted by Name (A–Z), Recently modified, or Recently created; while searching, results default to relevance and an explicit sort overrides it.
- The list can be filtered to characters carrying a chosen tag; the count reflects the filtered total.
- Search preserves the current multi-field FTS matching (name/description/personality/scenario/system_prompt), and now composes with sort + tag + pagination + an accurate count.
- The Personas (who-you-are) mode gets pagination + sort + search over its file-backed backend (no tag filter — persona profiles have no tags).
- Dictionaries/Lore modes are visually and behaviorally unchanged.

---

## Background — current state (verified via source scout, file:line approximate; re-verify at plan time)

**Shared pane** — `tldw_chatbook/Widgets/Persona_Widgets/personas_library_pane.py`
- The list is a Textual **`ListView`** (`#personas-library-rows`, ~:143). `update_rows(rows, total, noun)` (~:159–259) does a **full rebuild** every render: `await list_view.clear()` then `await list_view.extend(items)`. No virtualization — all `ListItem`s mount at once.
- A `LibraryRow` dataclass (~:37–45) = `item_id, kind, name, is_unsaved, meta`.
- Controls today: a search `Input` (`#personas-library-search`, ~:123), toolbar buttons New/Import/Duplicate (~:124–142), a count `Static` (`#personas-library-count`, ~:144). **No sort or filter control exists.**
- `set_mode(mode)` (~:146–157) toggles which toolbar buttons show per mode. **One pane, one `ListView`, fed different rows per mode** — changing the pane contract affects all four modes, so mode-specific behavior is driven from the screen's per-mode render methods.

**Screen** — `tldw_chatbook/UI/Screens/personas_screen.py`
- Characters feed: `refresh_character_library_list` (~:625) → cached `self._characters` (id+name only). `_render_library_rows` (~:667–722): `total = len(self._characters)`; small libraries filter in-memory by name substring; at `total >= LIBRARY_FTS_THRESHOLD` (=1000, ~:188) it runs `search_characters_fts` (top-50, `ORDER BY rank`, no offset, count denominator-less).
- Personas feed: `_refresh_profile_rows_worker` (~:752) → `_render_profile_rows` (~:776–821), in-memory name substring only.
- Search debounce: `PERSONAS_SEARCH_DEBOUNCE_SECONDS = 0.2` (~:153); `_handle_search_changed` (~:823) → exclusive worker (group `personas-library-search`) → per-mode render, with a stale-snapshot guard (`_library_render_snapshot_is_current`, ~:650). **Every render is a full ListView rebuild.**

**Characters backend**
- `Character_Chat/Character_Chat_Lib.py`: `fetch_character_names(limit=1000)` (~:457) → `get_character_list_for_ui(db, limit=1000)` (~:415) returns `[{"id","name"}]` sorted by `name.lower()` (~:437). Drops `created_at`/`last_modified`/`tags`.
- `DB/ChaChaNotes_DB.py`: `list_character_cards(limit=100, offset=0)` (~:3913), query `SELECT * FROM character_cards WHERE deleted = 0 ORDER BY name LIMIT ? OFFSET ?` (~:3932). **No `order_by` param, no include-deleted flag.**
- Columns on `character_cards` (schema ~:176–198): `name` (UNIQUE), `created_at`, `last_modified`, `version`, `tags` (TEXT, JSON array, in `_CHARACTER_CARD_JSON_FIELDS` ~:3651, deserialized to a Python list), plus creator fields.
- FTS: `character_cards_fts` covers `name, description, personality, scenario, system_prompt` (schema ~:212–217). `tags` is NOT in FTS. `search_character_cards(search_term, limit=10)` (~:4455) = `... MATCH ? ORDER BY rank LIMIT ?`, no offset.
- **No COUNT(*) helper for `character_cards` exists.** No query-by-tag helper.

**Handler seam** — `tldw_chatbook/UI/CCP_Modules/ccp_character_handler.py`: `fetch_all_characters` (~:67), `search_characters_fts` (~:45, wraps term as `"term"*` prefix, `db.search_character_cards(..., limit=50)`), `fetch_character_by_id` (~:85), CRUD create/update/delete (~:93/98/115). `CCPCharacterHandler.refresh_character_list` (~:329).

**Personas backend** — separate, **file-backed**: `local_character_persona_service.py` `LocalCharacterPersonaService.list_persona_profiles(mode, limit=100, offset=0)` (~:651) over an in-memory `self._persona_profiles` JSON list (~:36, loaded/persisted as JSON), sorts by `created_at` desc in Python (~:665), slices `[offset:offset+limit]` (~:666). Reached via `app.character_persona_scope_service.list_persona_profiles` (`character_persona_scope_service.py` ~:596). A server backend variant exists (`server_character_persona_service.py`).

**Design-relevant constraints**
1. The list mounts every row at once → the scale lever is **only page-a-worth of `ListItem`s mounted** (pagination), not virtualization.
2. No COUNT for `character_cards` → accurate "X of N" needs a new count query.
3. `list_character_cards` is `ORDER BY name` fixed → sort needs an `order_by` param; non-name sorts also need the extra columns fetched.
4. Large-library search is FTS (rank, cap 50, no offset, tag-blind) → the FTS path must gain a composable/paged form.
5. Tags are a JSON array in `character_cards.tags`, unindexed, un-queried, and not loaded into `self._characters`.
6. Two very different backends (SQL characters, file-backed personas) sit behind one pane → the pane needs one backend-agnostic contract.

---

## Design decisions (resolved with the user)

1. **Direction:** targeted polish, keep the current Characters shape.
2. **P3 scope:** two sub-projects — **P3a = scale + find** (this doc), **P3b = editor polish** (next cycle).
3. **Scale mechanism:** **pagination** (page-at-a-time), not virtualization, not raise-the-cap.
4. **Feature set:** pagination **+ sort + tag filter** (all three).
5. **Search model:** **preserve FTS** (multi-field relevance), made composable so sort + tag + pagination + count all apply.

---

## Architecture

### A. Characters query seam (`DB/ChaChaNotes_DB.py` + `Character_Chat_Lib.py`)

Introduce one paginated, composable query path plus a count, replacing the split (in-memory substring / capped-FTS) the screen uses today.

**Sort whitelist (no SQL injection).** A fixed mapping from a UI sort key to an exact ORDER BY clause — never interpolate user input into SQL:

```
CHARACTER_SORT_CLAUSES = {
    "name_asc":       "ORDER BY name COLLATE NOCASE ASC",
    "modified_desc":  "ORDER BY last_modified DESC, name COLLATE NOCASE ASC",
    "created_desc":   "ORDER BY created_at DESC, name COLLATE NOCASE ASC",
    "relevance":      "ORDER BY rank",   # search mode only (FTS)
}
```

**New DB methods** (names indicative; finalize in plan):

- `list_character_cards_page(*, limit, offset, order_by="name_asc", search_term=None, tag=None) -> list[dict]`
  - **Browse (no `search_term`):** `SELECT <needed cols> FROM character_cards WHERE deleted = 0 [AND <tag predicate>] <order_by clause> LIMIT ? OFFSET ?`.
  - **Search (`search_term` present):** join FTS — **verified: this mirrors the existing `search_character_cards` exactly** (`ChaChaNotes_DB.py` ~:5136 does `FROM character_cards_fts fts JOIN character_cards cc ON fts.rowid = cc.id WHERE fts.character_cards_fts MATCH ? AND cc.deleted = 0 ORDER BY rank LIMIT ?`). The FTS table is FTS5 **external-content** (`content='character_cards', content_rowid='id'`, ~:223–228), and the AI/AU/AD sync triggers exclude soft-deleted rows — so FTS already contains only non-deleted rows (the `AND c.deleted=0` is redundant-but-harmless in search mode, required in browse mode).
    `SELECT c.<cols> FROM character_cards c JOIN character_cards_fts f ON f.rowid = c.id WHERE f.character_cards_fts MATCH ? AND c.deleted = 0 [AND <tag predicate>] <order_by clause | ORDER BY rank> LIMIT ? OFFSET ?`.
    `order_by="relevance"` → `ORDER BY rank`; any explicit sort overrides it. The MATCH term is wrapped `"<term>"*` (prefix) — this wrapping lives in the **handler** (`ccp_character_handler.py` `search_characters_fts` ~:45), NOT the DB method, which passes the term through raw; mirror the handler's wrapping and wrap the MATCH call so an invalid FTS query string degrades to no-results rather than raising.
  - Returns the columns the UI needs: at least `id, name, last_modified, created_at, tags` (so rows can carry a meta line and sorting/tag work without a second fetch).
- `count_character_cards(*, search_term=None, tag=None) -> int`
  - Same WHERE (and FTS join when searching) as the list query, `SELECT COUNT(*)`. Drives the accurate "X–Y of N".
- `list_distinct_character_tags() -> list[str]`
  - `SELECT DISTINCT je.value FROM character_cards c, json_each(CASE WHEN json_valid(c.tags) THEN c.tags ELSE '[]' END) je WHERE c.deleted = 0 ORDER BY je.value COLLATE NOCASE`. Populates the tag-filter control.

**Tag predicate.** **⚠ `json_each` must be guarded** — `character_cards.tags` is nullable and the writer (`_ensure_json_string_from_mixed`, `ChaChaNotes_DB.py` ~:4155–4167) **passes a non-JSON string through unchanged** while reads tolerate bad JSON (`_deserialize_row_fields` ~:4116–4127), so a real row can hold `NULL` or an invalid-JSON `tags` value. Bare `json_each(tags)` **RAISES** on those and would crash the entire list/count/distinct query. Use the always-valid form:
`EXISTS (SELECT 1 FROM json_each(CASE WHEN json_valid(character_cards.tags) THEN character_cards.tags ELSE '[]' END) WHERE value = ?)` — exact tag membership; the `CASE` guarantees `json_each` always receives valid JSON (`'[]'` for NULL/invalid). `tag=None` omits the clause. Requires SQLite JSON1 (default-compiled since 3.9 / 2015; the DB already relies on JSON storage here) — if ever absent, `list_distinct_character_tags` and the predicate fall back to a Python pass over loaded rows.

**Keep existing methods intact.** `list_character_cards`, `search_character_cards`, `fetch_character_names`, `get_character_list_for_ui` stay for their other callers; P3a adds new methods and points the *pane's* character path at them. (Plan verifies no other caller depends on the old capped behavior.)

**Lib wrapper.** A thin `Character_Chat_Lib` helper (e.g. `get_character_page_for_ui(db, *, limit, offset, order_by, search_term, tag)` + `count_character_page`) so the screen never talks raw SQL and rows arrive UI-shaped.

### B. Personas: filter/sort/page in the screen layer (no backend edits)

**Verified:** `app.character_persona_scope_service` is a `CharacterPersonaScopeService` that wraps BOTH a local (file-backed) and a server backend behind a policy enforcer (`app.py` ~:3788–3796) — modifying `list_persona_profiles` on both backends for parity is avoidable. Persona lists are small (local default cap 100), so P3a **applies sort/search/pagination in the screen (or a thin scope-layer helper) over the profile list already returned by `list_persona_profiles(mode)`** — backend-agnostic, works for whichever backend the policy routes to, and needs no dual-backend change.

- **Search:** case-insensitive substring over the profile `name` (and `description`) — **verified: persona profiles carry `name` + `description`** (`local_character_persona_service.py` ~:512–534).
- **Sort:** the same sort keys mapped to Python `sorted()` (Name A–Z; Recently modified/created over the profile timestamps). "Relevance" is not meaningful without FTS, so **personas default to Name A–Z while searching** and never expose the Relevance option.
- **Tag filter: NOT applicable to personas — verified persona profiles have NO `tags` field.** The tag control is therefore **characters-only** (gated off in personas mode; see §C).
- **Pagination:** filter → sort → `[offset:offset+page_size]`; total = length of the filtered (pre-slice) list, so "X–Y of N" is accurate.

### C. Pane UI (`personas_library_pane.py`)

Add backend-agnostic controls the screen drives, gated by mode via `set_mode` (all hidden for dict/lore):
- **Sort control** — characters **and** personas. A compact cycle button ("Sort: Name ▾" → Recently modified → Recently created → [Relevance, only while a character search is active]) posting a `PersonaSortChanged(sort_key)` message. (A `Select` is the alternative; the plan picks whichever fits the 2fr rail cleanly — cycle button preferred for width.)
- **Tag filter** — **characters only** (personas have no tags — §B). A "Tag: All ▾" control opening a tag picker (reuse the existing modal-picker pattern from the Lore/Dict attach pickers) populated from `list_distinct_character_tags()`; selecting posts `PersonaTagFilterChanged(tag | None)`. "All" clears it. Hidden in personas/dict/lore modes.
- **Page bar** — characters **and** personas. At the list foot: "◄ 1–50 of N ►" with prev/next (disabled at ends), posting `PersonaPageChanged(delta | page)`. Hidden when `N <= page_size`.

**Page size:** `PERSONAS_LIBRARY_PAGE_SIZE = 50` (one constant).

**`update_rows` contract extension:** today `update_rows(rows, total, noun)`; extend to also convey the page window (offset, page_size, filtered_total) so the pane can render the page bar and an accurate count. Keep the signature backward-compatible for dict/lore callers (default page params → no page bar, current behavior).

The pane stays a `ListView` with the same `clear()`+`extend()` rebuild — but each rebuild now receives only one page (≤50) of rows, so mounted-widget count is bounded regardless of library size.

### D. Screen wiring & data flow (`personas_screen.py`)

- Hold the query state per mode: `sort_key`, `tag_filter`, `page_offset`, alongside the existing `search_query` (in `PersonasWorkbenchState`).
- A single **`_load_character_page` worker** (off-thread via `asyncio.to_thread`, exclusive, group reused from the current search group) that: calls the lib page with `(limit=PAGE_SIZE, offset, order_by=sort_key, search_term, tag)`, builds `LibraryRow`s (now with a meta line, e.g. last-modified date), and calls `library.update_rows(rows, filtered_total, page window, noun)`. The existing stale-snapshot guard extends to include `(sort_key, tag, offset)` so a slow page load can't overwrite a newer request.
- **Count caching:** the total `N` depends only on `(search_term, tag)` — not on sort or offset. Run `count_character_cards` only when `(search_term, tag)` changes (cache it keyed by that pair); a prev/next page click or a sort change reuses the cached count and issues just the page query. Avoids a redundant COUNT on every page navigation.
- **Reset offset to 0** whenever search / sort / tag changes (a filter change invalidates the current page). Page prev/next only changes offset.
- **Relevance is search-only:** `sort_key="relevance"` is reachable only while a search term is present. When the search clears while `sort_key=="relevance"`, the screen falls back to the default `name_asc` (and the sort control re-labels to "Name"). Personas never expose Relevance (no FTS).
- Personas: a parallel `_load_persona_page` path calling the persona service with the same params.
- Handlers: `@on(PersonaSortChanged)`, `@on(PersonaTagFilterChanged)`, `@on(PersonaPageChanged)` → update state → trigger the load worker for the active mode. All wrapped so a backend error notifies rather than crashing the worker (`run_worker(exit_on_error=…)` discipline — see constraints).
- `set_mode`/`_apply_mode`: show sort + page controls for characters + personas and the tag control for characters only; on entering a mode, reset offset and load page 0.

**Cache-assumption fixes (verified — `self._characters` becomes one page, not the capped-full library).** Selection is safe (resolved by id via `character_handler.load_character`, not a cache scan), but five sites assume the cache holds the whole (capped) library and must be corrected in this task:
1. `_render_library_rows` count/filter (~:778 `total = len(self._characters)`, ~:796/801 in-memory substring) → replaced by the paged query + `count_character_cards` (the core change).
2. Status string (~:1329 `f"Characters: {len(self._characters)} ..."`) → use the COUNT total, not the page length.
3. Import name-conflict detection (~:3766 `pre_import_ids = {ids in self._characters}` → ~:3803 `if imported_id in pre_import_ids`) → a page-only id set would misclassify an import that collides with an **off-page** existing character as "new". Derive the conflict signal without the full-library cache — e.g. an existence check on `imported_id` before/after, or a return signal from the importer. (Cosmetic: notification wording; no data effect — but a real regression to prevent.)
4. `_character_record` name resolution after import (~:3797) and after save (~:4565) — the cache scan returns `None` for an off-page character → generic fallback name. Resolve the name by id (handler-loaded record / by-id fetch) instead of scanning the page cache. (Selection itself already uses the id and is unaffected.)

### Error handling / edge cases

- **Empty results** (search/tag matches nothing): page bar hidden, count "0 of 0", empty ListView (existing empty-state).
- **Offset past the end** (e.g. tag change shrinks N below current offset): clamp offset into range before fetch; if the page is empty but N>0, snap to the last valid page.
- **Deleted-during-paging / stale snapshot:** the extended snapshot guard drops results whose `(mode, search, sort, tag, offset)` no longer matches current state (mirrors the P2e/P2g freshness-guard lesson — re-check state after the await before writing to the pane).
- **Malformed / NULL tags JSON** on a row: handled by the `CASE WHEN json_valid(tags) THEN tags ELSE '[]' END` guard (§A) so `json_each` never sees invalid JSON and never raises; such a row simply never matches a tag filter and contributes no distinct tags. The distinct-tags/count/list queries all use the same `deleted = 0` + guarded predicate, so they stay consistent with each other. **A test must seed a row with NULL and a non-JSON tags value and assert the list/count/distinct queries don't raise.**
- **FTS unavailable / MATCH syntax error** on odd input: the search path must degrade to no-results-with-notice rather than raise (wrap the FTS query; the current `search_characters_fts` already tolerates this pattern).
- **`json_each` absence** (hypothetical old SQLite): `list_distinct_character_tags` and the tag predicate fall back to a Python-side pass over loaded rows (documented; not expected to trigger).

### Testing strategy

- **Real-DB tests** (`Tests/…` mirroring existing character-DB tests) for the new DB methods: COUNT matches the list length across pages; `order_by` whitelist produces the three orders + relevance; `json_each` tag predicate returns exactly the tagged rows; offset paging returns disjoint consecutive windows covering the full set; search + tag + sort compose (FTS join returns only matched∧tagged rows in the chosen order); `list_distinct_character_tags` returns sorted uniques; malformed/NULL tags never raise. Seed >2 pages of characters with varied tags/timestamps.
- **Personas service tests:** pure-Python sort/tag/search/paging parity over a seeded in-memory profile list; accurate filtered total; empty/short-list cases.
- **Pane widget tests** (host-App mount): page bar renders "X–Y of N", prev/next enable/disable at bounds and post `PersonaPageChanged`; sort cycle posts `PersonaSortChanged` cycling the keys (and includes Relevance only while a search is active); tag control posts `PersonaTagFilterChanged`; controls hidden for dict/lore via `set_mode`; `update_rows` with `N <= page_size` hides the page bar.
- **Screen integration tests** (real screen + real DB): seed >page-size characters → first page shows PAGE_SIZE rows and "1–50 of N"; next → second page (disjoint); pick a tag → count drops to the tagged subset and rows carry the tag; type a search → FTS-matched paginated subset; change sort → order changes and offset resets to 0; switch to Personas mode → same controls drive the file-backed list. A stale-snapshot regression test (slow page load superseded by a newer sort/tag change does not overwrite).

---

## Task decomposition (for writing-plans → subagent-driven-development)

1. **Characters query + count + tags seam** — `list_character_cards_page`, `count_character_cards`, `list_distinct_character_tags` in `ChaChaNotes_DB.py` + the `Character_Chat_Lib` UI-shaped wrappers, with the sort whitelist and tag predicate. Real-DB tests. (Deliverable: backend can page/sort/tag/count/search characters.)
2. **Personas filter/sort/page helper** — a thin screen/scope-layer function that takes the profiles from `list_persona_profiles(mode)` and applies search (name+description substring), sort (Name/Recent), and `[offset:offset+page_size]`, returning `(page_rows, filtered_total)`. **No tag path** (personas have no tags), **no backend edits** (works for whichever backend the scope service routes to). Pure-Python tests over a seeded profile list.
3. **Pane controls** — sort cycle button, tag filter control + picker, page bar; new `Persona*Changed` messages; `update_rows` page-window extension; `set_mode` gating. Widget tests.
4. **Screen wiring + integration + cache-assumption fixes** — query state in `PersonasWorkbenchState`; `_load_character_page` / `_load_persona_page` workers (off-thread, extended stale-snapshot guard incl. `(sort, tag, offset)`, offset reset on filter change, offset clamp); the three `@on` handlers; `_apply_mode` control gating (tag = characters only) + page-0 load; **plus the five `self._characters` cache-assumption fixes in §D** (count/filter, status string, import-conflict detection, name resolution after import/save). Screen integration tests including a paged >page-size library, tag-narrows-count, sort resets offset, mode-switch, a stale-snapshot regression, and an import-of-off-page-name-conflict messaging test.

Four focused tasks → one PR.

---

## Global constraints

- **NO schema migration** — P3a only reads existing `character_cards` columns; ChaChaNotes stays **v22** (NEXT migration v22→v23 reserved for a future change).
- **Parameterized SQL only** — the ONLY dynamic SQL is the ORDER BY clause, which comes from the fixed `CHARACTER_SORT_CLAUSES` whitelist keyed by an enum; user values (search term, tag) are bound parameters. No string interpolation of user input.
- **All screen DB I/O off-thread** via `asyncio.to_thread`, wrapped so a backend error notifies (never crashes the worker); reuse the existing exclusive search-worker group; extend the stale-snapshot freshness guard to `(mode, search, sort, tag, offset)`.
- **Dict/Lore modes unchanged** — new controls gated off; `update_rows` stays backward-compatible for them.
- **Branch off the LATEST `origin/dev`** (`8f1d2cae8` at spec time — re-verify at plan/merge time). Feature branch `claude/roleplay-p3a-library-scale-find`.
- **Implementers stage ONLY their task's files** — never `git add -A`, never stage `.superpowers/`.
- **No background/broad test sweeps; never broad-`pkill` pytest** in this multi-session environment — scope commands to this worktree.
- **Test env:** `HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest … -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread` (venv in the MAIN checkout; imports resolve to worktree source via cwd; UI tests are slow — generous timeouts).

## Deferred / follow-ups

- **P3b** — editor polish (avatar preview, alternate-greetings UX, validation surface, dirty-tracking re-arm), the next P3 cycle.
- Consistency-of-feel pass and deterministic Try-it preview (not selected for P3).
- Adding `tags` to the FTS index (would let search match tags directly) — out of scope; tag filtering is explicit here.
- Server persona-backend paging parity if/when that path becomes the live workbench backend.
