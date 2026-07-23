# Roleplay P3a — Library Pagination + Sort + Tag Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the shared Roleplay library list scale (page-at-a-time instead of mounting every row) and add a sort control + a characters-only tag filter, across the SQL-backed Characters mode and the file-backed Personas mode.

**Architecture:** Add three new read-only DB methods (paged list + count + distinct-tags) that compose search (FTS-join) + tag (`json_each`, json-valid-guarded) + a whitelisted `ORDER BY` + `LIMIT/OFFSET`; thin `Character_Chat_Lib` wrappers shape them for the UI. Personas filter/sort/page in a pure helper over the ≤100 returned profiles (no backend edit, no tags). The shared `PersonasLibraryPane` gains a sort button, a tag button, and a page bar (all gated by mode); the screen owns the paging state, a `(search,tag)`-keyed count cache, and the five `self._characters` cache-assumption fixes.

**Tech Stack:** Python 3.11+, Textual, SQLite (FTS5 + JSON1), pytest.

**Spec:** `Docs/superpowers/specs/2026-07-21-roleplay-p3a-library-scale-find-design.md` (committed `290249b1e`).

## Global Constraints

- **NO schema migration** — reads existing `character_cards` columns only; ChaChaNotes stays **v22** (`_CURRENT_SCHEMA_VERSION = 22`, ChaChaNotes_DB.py:154). Do not touch the DDL or `migration_steps`.
- **Parameterized SQL only.** The ONLY dynamic SQL fragment is the `ORDER BY` clause, taken from the fixed `_CHARACTER_SORT_CLAUSES` whitelist keyed by an enum string; `search_term` and `tag` are always bound `?` parameters. Never interpolate user input into SQL.
- **`json_each` MUST be json-valid-guarded:** `json_each(CASE WHEN json_valid(<col>) THEN <col> ELSE '[]' END)` everywhere — bare `json_each` raises on NULL/invalid, and rows can hold NULL or non-JSON tags.
- **`PERSONAS_LIBRARY_PAGE_SIZE = 50`** (one constant).
- **Tag filter is characters-only** — persona profiles have no tags. Sort + page apply to characters and personas; **dict/lore modes are unchanged** (new controls gated off; `update_rows` stays backward-compatible when page kwargs are omitted).
- **All screen DB I/O off-thread** via `asyncio.to_thread`, wrapped so a backend error notifies (never crashes the worker); reuse the existing exclusive `personas-library-search` worker group; extend the stale-snapshot guard to include `(mode, search, sort, tag, offset)`; **reset offset to 0 on any filter change**; clamp offset into range.
- **Count is cached keyed by `(search_term, tag)`** — a prev/next page click or sort change reuses it and issues only the page query.
- **Selection is by id** (`character_handler.load_character`) — never resolve a selected character by scanning `self._characters`.
- **Real-DB tests** (fakes mask bugs). Character DB tests seed >2 pages (≥120 rows) with varied tags/timestamps **plus one NULL-tags row and one non-JSON-tags row** and assert the list/count/distinct queries never raise.
- **Implementers stage ONLY their task's files** — never `git add -A`, never stage `.superpowers/`.
- **No background/broad test sweeps; never broad-`pkill` pytest** (multi-session env) — scope commands to this worktree.
- **Branch:** `claude/roleplay-p3a-library-scale-find` off `origin/dev` (`8f1d2cae8`). **Test command** (prefix every pytest run):
  ```bash
  HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <targets> \
  -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
  ```
  (venv lives in the MAIN checkout; imports resolve to worktree source via cwd. UI tests are slow — the 300s timeout is deliberate.)

---

## File Structure

**Modified:**
- `tldw_chatbook/DB/ChaChaNotes_DB.py` — add `_CHARACTER_SORT_CLAUSES`, `list_character_cards_page`, `count_character_cards`, `list_distinct_character_tags` (Task 1). `list_character_cards` / `search_character_cards` left intact.
- `tldw_chatbook/Character_Chat/Character_Chat_Lib.py` — add `get_character_page_for_ui`, `count_character_page`, `list_character_tags` (Task 1).
- `tldw_chatbook/Widgets/Persona_Widgets/personas_messages.py` — add `PersonaSortCycleRequested`, `PersonaTagFilterRequested`, `PersonaPageChanged` (Task 3).
- `tldw_chatbook/Widgets/Persona_Widgets/personas_library_pane.py` — sort button, tag button, page bar, `update_rows` page kwargs, `set_sort_label`/`set_tag_label`, `set_mode` gating (Task 3).
- `tldw_chatbook/Widgets/Persona_Widgets/personas_state.py` — add `sort_key`, `tag_filter`, `page_offset` fields (Task 4).
- `tldw_chatbook/UI/Screens/personas_screen.py` — paged character/persona render, count cache, sort/tag/page handlers, cache-assumption fixes, `_apply_mode` gating (Task 4).

**Created:**
- `tldw_chatbook/Character_Chat/persona_list_paging.py` — pure `page_persona_profiles(...)` (Task 2).
- `tldw_chatbook/Widgets/Persona_Widgets/tag_filter_picker.py` — `TagFilterPicker(ModalScreen[str | None])` (Task 3).
- Tests: `Tests/DB/test_character_cards_paging.py` (Task 1), `Tests/Character_Chat/test_persona_list_paging.py` (Task 2), `Tests/UI/test_personas_library_pane_paging.py` (Task 3), extend `Tests/UI/test_personas_screen_*` or a new `Tests/UI/test_personas_library_scale.py` (Task 4). Confirm exact existing test paths/fixtures at plan time by grepping `Tests/` for `character_cards`, `PersonasLibraryPane`, and personas-screen host apps.

---

## Task 1: Characters query + count + tags DB seam

**Files:**
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py` (add methods near `list_character_cards`, ~:4460, and `search_character_cards`, ~:5113)
- Modify: `tldw_chatbook/Character_Chat/Character_Chat_Lib.py` (add wrappers near `get_character_list_for_ui`, ~:490)
- Test: `Tests/DB/test_character_cards_paging.py`

**Interfaces:**
- Produces (DB, on `CharactersRAGDB`):
  - `_CHARACTER_SORT_CLAUSES: dict[str, str]` — class constant.
  - `list_character_cards_page(self, *, limit: int, offset: int, order_by: str = "name_asc", search_term: str | None = None, tag: str | None = None) -> list[dict]` (full deserialized rows).
  - `count_character_cards(self, *, search_term: str | None = None, tag: str | None = None) -> int`.
  - `list_distinct_character_tags(self) -> list[str]`.
- Produces (lib):
  - `get_character_page_for_ui(db, *, limit, offset, order_by="name_asc", search_term=None, tag=None) -> list[dict]` → each `{"id","name","last_modified","created_at","tags"}`.
  - `count_character_page(db, *, search_term=None, tag=None) -> int`.
  - `list_character_tags(db) -> list[str]`.

**Verified facts to build on:**
- `character_cards.id INTEGER PRIMARY KEY AUTOINCREMENT` (rowid alias); `character_cards_fts` is FTS5 external-content (`content='character_cards', content_rowid='id'`); triggers exclude soft-deleted so FTS holds only non-deleted rows.
- Existing search SQL to mirror (~:5136): `SELECT cc.* FROM character_cards_fts fts JOIN character_cards cc ON fts.rowid = cc.id WHERE fts.character_cards_fts MATCH ? AND cc.deleted = 0 ORDER BY rank LIMIT ?`.
- The `"term"*` FTS prefix wrap lives in the handler (`ccp_character_handler.py:56-57`: `escaped = term.replace('"','""'); match_query = f'"{escaped}"*'`). Reuse this exact wrapping in the DB layer's search branch.
- `_deserialize_row_fields(row, self._CHARACTER_CARD_JSON_FIELDS)` (~:4094) deserializes `tags`/`alternate_greetings`/`extensions`.

- [ ] **Step 1: Write the failing tests**

Create `Tests/DB/test_character_cards_paging.py`. Grep an existing DB test (e.g. `Tests/DB/test_chachanotes_db*.py`) for the in-memory `CharactersRAGDB` fixture and mirror it; if none, construct `CharactersRAGDB(db_path=":memory:", client_id="test")` (confirm the constructor signature at plan time).

```python
import pytest
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def db(tmp_path):
    return CharactersRAGDB(tmp_path / "p3a.db", client_id="test")


def _add(db, name, tags, desc=""):
    return db.add_character_card({"name": name, "description": desc, "tags": tags})


def test_page_returns_window_and_disjoint_pages(db):
    for i in range(120):
        _add(db, f"char{i:03d}", ["even"] if i % 2 == 0 else ["odd"])
    page1 = db.list_character_cards_page(limit=50, offset=0, order_by="name_asc")
    page2 = db.list_character_cards_page(limit=50, offset=50, order_by="name_asc")
    assert len(page1) == 50 and len(page2) == 50
    ids1 = {c["id"] for c in page1}
    ids2 = {c["id"] for c in page2}
    assert ids1.isdisjoint(ids2)
    # name_asc order
    assert [c["name"] for c in page1] == sorted(c["name"] for c in page1)


def test_count_matches_full_set(db):
    for i in range(120):
        _add(db, f"c{i:03d}", [])
    # +1 for the pre-seeded Default Assistant (id=1) — assert relative, not absolute:
    assert db.count_character_cards() == len(
        db.list_character_cards_page(limit=1000, offset=0)
    )


def test_tag_filter_narrows_list_and_count(db):
    for i in range(10):
        _add(db, f"t{i}", ["hero"] if i < 3 else ["villain"])
    heroes = db.list_character_cards_page(limit=100, offset=0, tag="hero")
    assert {c["name"] for c in heroes} == {"t0", "t1", "t2"}
    assert db.count_character_cards(tag="hero") == 3


def test_search_composes_with_tag_and_sort(db):
    _add(db, "Dragon Knight", ["hero"])
    _add(db, "Dragon Fiend", ["villain"])
    _add(db, "Wolf", ["hero"])
    hits = db.list_character_cards_page(
        limit=100, offset=0, search_term='"Dragon"*', tag="hero", order_by="name_asc"
    )
    assert [c["name"] for c in hits] == ["Dragon Knight"]
    assert db.count_character_cards(search_term='"Dragon"*', tag="hero") == 1


def test_sort_by_modified_desc(db):
    a = _add(db, "alpha", [])
    b = _add(db, "beta", [])
    # bump beta's last_modified by updating it
    rec = db.get_character_card_by_id(b)
    db.update_character_card(b, {"description": "x"}, int(rec["version"]))
    rows = db.list_character_cards_page(limit=100, offset=0, order_by="modified_desc")
    names = [c["name"] for c in rows]
    assert names.index("beta") < names.index("alpha")


def test_distinct_tags_sorted_unique(db):
    _add(db, "a", ["Zed", "amber"])
    _add(db, "b", ["amber"])
    tags = db.list_distinct_character_tags()
    assert tags == sorted(set(tags), key=str.lower)
    assert "amber" in tags and "Zed" in tags
    assert tags.count("amber") == 1


def test_malformed_and_null_tags_never_raise(db):
    good = _add(db, "good", ["ok"])
    # Force a NULL tags row and a non-JSON tags row directly.
    conn = db.get_connection()
    conn.execute("UPDATE character_cards SET tags = NULL WHERE id = ?", (good,))
    bad = _add(db, "bad", [])
    conn.execute("UPDATE character_cards SET tags = 'not,json' WHERE id = ?", (bad,))
    conn.commit()
    # None of these may raise:
    assert isinstance(db.list_character_cards_page(limit=100, offset=0), list)
    assert isinstance(db.count_character_cards(), int)
    assert isinstance(db.list_distinct_character_tags(), list)
    assert isinstance(db.count_character_cards(tag="ok"), int)


def test_unknown_order_by_falls_back_to_name(db):
    _add(db, "b", [])
    _add(db, "a", [])
    rows = db.list_character_cards_page(limit=10, offset=0, order_by="bogus")
    assert [c["name"] for c in rows][:2] == ["a", "b"]
```

- [ ] **Step 2: Run the tests — verify they fail**

Run (from Global Constraints test command) `Tests/DB/test_character_cards_paging.py`.
Expected: FAIL — `AttributeError: 'CharactersRAGDB' object has no attribute 'list_character_cards_page'`.

- [ ] **Step 3: Add the sort whitelist + the three DB methods**

In `ChaChaNotes_DB.py`, add the class constant near `_CHARACTER_CARD_JSON_FIELDS` (~:4130):

```python
    # P3a: whitelist of UI sort keys → exact ORDER BY clauses. The ONLY dynamic
    # SQL fragment; search_term/tag are always bound parameters. "relevance"
    # is valid only in the search (FTS) branch.
    _CHARACTER_SORT_CLAUSES = {
        "name_asc": "ORDER BY name COLLATE NOCASE ASC",
        "modified_desc": "ORDER BY last_modified DESC, name COLLATE NOCASE ASC",
        "created_desc": "ORDER BY created_at DESC, name COLLATE NOCASE ASC",
        "relevance": "ORDER BY rank",
    }
```

Add the methods after `list_character_cards` (after ~:4527). Note the shared `json_each` guard and the FTS-join branch mirroring `search_character_cards`:

```python
    # P3a: json-valid guard so json_each never sees NULL / non-JSON tags.
    _TAGS_JSON_EACH = "json_each(CASE WHEN json_valid({t}.tags) THEN {t}.tags ELSE '[]' END)"

    def _resolve_sort_clause(self, order_by: str, *, searching: bool) -> str:
        clause = self._CHARACTER_SORT_CLAUSES.get(order_by)
        if clause is None or (order_by == "relevance" and not searching):
            clause = self._CHARACTER_SORT_CLAUSES["name_asc"]
        return clause

    def list_character_cards_page(
        self,
        *,
        limit: int,
        offset: int,
        order_by: str = "name_asc",
        search_term: str | None = None,
        tag: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Paged, sortable, tag- and search-filterable character list.

        Args:
            limit: Page size.
            offset: Rows to skip.
            order_by: A key of ``_CHARACTER_SORT_CLAUSES``; unknown keys (and
                "relevance" without a search term) fall back to "name_asc".
            search_term: FTS5 MATCH query (already prefix-wrapped by the caller,
                e.g. ``'"dragon"*'``) or None for a browse query.
            tag: Exact tag membership filter, or None.

        Returns:
            Deserialized character-card dicts for the page. Never raises on
            NULL/invalid tags (json_each is json-valid-guarded).
        """
        searching = bool(search_term)
        sort_clause = self._resolve_sort_clause(order_by, searching=searching)
        params: list[Any] = []
        where = ["cc.deleted = 0"] if searching else ["deleted = 0"]
        alias = "cc" if searching else "character_cards"
        if searching:
            head = (
                "SELECT cc.* FROM character_cards_fts fts "
                "JOIN character_cards cc ON fts.rowid = cc.id"
            )
            where.insert(0, "fts.character_cards_fts MATCH ?")
            params.append(search_term)
        else:
            head = "SELECT * FROM character_cards"
        if tag is not None:
            where.append(
                f"EXISTS (SELECT 1 FROM {self._TAGS_JSON_EACH.format(t=alias)} WHERE value = ?)"
            )
            params.append(tag)
        query = f"{head} WHERE {' AND '.join(where)} {sort_clause} LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        cursor = self.execute_query(query, tuple(params))
        return [
            self._deserialize_row_fields(row, self._CHARACTER_CARD_JSON_FIELDS)
            for row in cursor.fetchall()
            if row
        ]

    def count_character_cards(
        self, *, search_term: str | None = None, tag: str | None = None
    ) -> int:
        """Count non-deleted character cards matching the same search+tag filter."""
        searching = bool(search_term)
        params: list[Any] = []
        alias = "cc" if searching else "character_cards"
        where = ["cc.deleted = 0"] if searching else ["deleted = 0"]
        if searching:
            head = (
                "SELECT COUNT(*) FROM character_cards_fts fts "
                "JOIN character_cards cc ON fts.rowid = cc.id"
            )
            where.insert(0, "fts.character_cards_fts MATCH ?")
            params.append(search_term)
        else:
            head = "SELECT COUNT(*) FROM character_cards"
        if tag is not None:
            where.append(
                f"EXISTS (SELECT 1 FROM {self._TAGS_JSON_EACH.format(t=alias)} WHERE value = ?)"
            )
            params.append(tag)
        query = f"{head} WHERE {' AND '.join(where)}"
        cursor = self.execute_query(query, tuple(params))
        row = cursor.fetchone()
        return int(row[0]) if row else 0

    def list_distinct_character_tags(self) -> List[str]:
        """Distinct tag values across non-deleted cards, case-insensitively sorted."""
        query = (
            "SELECT DISTINCT je.value "
            "FROM character_cards cc, "
            + self._TAGS_JSON_EACH.format(t="cc")
            + " je WHERE cc.deleted = 0 ORDER BY je.value COLLATE NOCASE"
        )
        cursor = self.execute_query(query, ())
        return [str(r[0]) for r in cursor.fetchall() if r and r[0] is not None]
```

(Confirm `self.execute_query` / `self.get_connection` are the right accessors by matching `list_character_cards` and `search_character_cards` in the same file.)

- [ ] **Step 4: Run the DB tests — verify they pass**

Run `Tests/DB/test_character_cards_paging.py`. Expected: PASS (all 8).
If `execute_query` isn't the right helper, mirror whatever `list_character_cards` uses.

- [ ] **Step 5: Add the lib wrappers**

In `Character_Chat_Lib.py`, after `get_character_list_for_ui` (~:525), add:

```python
def get_character_page_for_ui(
    db: CharactersRAGDB,
    *,
    limit: int,
    offset: int,
    order_by: str = "name_asc",
    search_term: str | None = None,
    tag: str | None = None,
) -> List[Dict[str, Any]]:
    """UI-shaped page of characters: id/name/last_modified/created_at/tags."""
    try:
        rows = db.list_character_cards_page(
            limit=limit, offset=offset, order_by=order_by,
            search_term=search_term, tag=tag,
        )
    except Exception as exc:
        logger.opt(exception=True).error(f"Character page fetch failed: {exc}")
        return []
    return [
        {
            "id": c.get("id"),
            "name": c.get("name"),
            "last_modified": c.get("last_modified"),
            "created_at": c.get("created_at"),
            "tags": c.get("tags") if isinstance(c.get("tags"), list) else [],
        }
        for c in rows
        if c.get("id") is not None
    ]


def count_character_page(
    db: CharactersRAGDB, *, search_term: str | None = None, tag: str | None = None
) -> int:
    try:
        return db.count_character_cards(search_term=search_term, tag=tag)
    except Exception as exc:
        logger.opt(exception=True).error(f"Character count failed: {exc}")
        return 0


def list_character_tags(db: CharactersRAGDB) -> List[str]:
    try:
        return db.list_distinct_character_tags()
    except Exception as exc:
        logger.opt(exception=True).error(f"Character tag list failed: {exc}")
        return []
```

- [ ] **Step 6: Add a lib wrapper test + run it**

Append to `Tests/DB/test_character_cards_paging.py` (or a lib test file if that's the local convention):

```python
def test_lib_wrapper_shapes_rows(db):
    from tldw_chatbook.Character_Chat.Character_Chat_Lib import (
        get_character_page_for_ui, count_character_page, list_character_tags,
    )
    _add(db, "Zeta", ["x"])
    rows = get_character_page_for_ui(db, limit=10, offset=0)
    assert rows and set(rows[0]) == {"id", "name", "last_modified", "created_at", "tags"}
    assert count_character_page(db) == len(get_character_page_for_ui(db, limit=1000, offset=0))
    assert "x" in list_character_tags(db)
```

Run the whole `Tests/DB/test_character_cards_paging.py`. Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/DB/ChaChaNotes_DB.py tldw_chatbook/Character_Chat/Character_Chat_Lib.py Tests/DB/test_character_cards_paging.py
git commit -m "feat(personas): P3a Task 1 — paged/sorted/tagged character query + count + distinct tags"
```

---

## Task 2: Personas in-screen filter/sort/page helper

**Files:**
- Create: `tldw_chatbook/Character_Chat/persona_list_paging.py`
- Test: `Tests/Character_Chat/test_persona_list_paging.py`

**Interfaces:**
- Produces: `page_persona_profiles(profiles: list[dict], *, search_term: str | None, sort_key: str, offset: int, page_size: int) -> tuple[list[dict], int]` — returns `(page_rows, filtered_total)`. `sort_key` ∈ `{"name_asc","modified_desc","created_desc"}` (NO "relevance" — personas have no FTS). Search = case-insensitive substring over `name` + `description`. No tag path (persona profiles have no tags).

**Verified facts:** persona profiles carry `name` + `description` + `created_at` (+ a modified timestamp — confirm the exact key: check `_persona_profile_view`, ~:197, and the create payload, ~:512-534; likely `updated_at`/`last_modified`/`modified_at`). Use `created_at` for `created_desc`; use the modified key for `modified_desc`, falling back to `created_at` if absent.

- [ ] **Step 1: Write the failing test**

```python
from tldw_chatbook.Character_Chat.persona_list_paging import page_persona_profiles

PROFILES = [
    {"id": "1", "name": "Alice", "description": "hero", "created_at": "2026-01-01", "updated_at": "2026-01-03"},
    {"id": "2", "name": "bob", "description": "villain", "created_at": "2026-01-02", "updated_at": "2026-01-02"},
    {"id": "3", "name": "Carol", "description": "hero mage", "created_at": "2026-01-03", "updated_at": "2026-01-01"},
]


def test_search_matches_name_and_description_case_insensitive():
    rows, total = page_persona_profiles(PROFILES, search_term="HERO", sort_key="name_asc", offset=0, page_size=50)
    assert {r["name"] for r in rows} == {"Alice", "Carol"}
    assert total == 2


def test_sort_name_asc_case_insensitive():
    rows, total = page_persona_profiles(PROFILES, search_term=None, sort_key="name_asc", offset=0, page_size=50)
    assert [r["name"] for r in rows] == ["Alice", "bob", "Carol"]
    assert total == 3


def test_sort_created_desc():
    rows, _ = page_persona_profiles(PROFILES, search_term=None, sort_key="created_desc", offset=0, page_size=50)
    assert [r["name"] for r in rows] == ["Carol", "bob", "Alice"]


def test_sort_modified_desc():
    rows, _ = page_persona_profiles(PROFILES, search_term=None, sort_key="modified_desc", offset=0, page_size=50)
    assert [r["name"] for r in rows] == ["Alice", "bob", "Carol"]


def test_pagination_window_and_total():
    profiles = [{"id": str(i), "name": f"p{i:03d}", "description": "", "created_at": "x", "updated_at": "x"} for i in range(120)]
    page2, total = page_persona_profiles(profiles, search_term=None, sort_key="name_asc", offset=50, page_size=50)
    assert total == 120 and len(page2) == 50 and page2[0]["name"] == "p050"


def test_unknown_sort_key_falls_back_to_name():
    rows, _ = page_persona_profiles(PROFILES, search_term=None, sort_key="relevance", offset=0, page_size=50)
    assert [r["name"] for r in rows] == ["Alice", "bob", "Carol"]
```

- [ ] **Step 2: Run — verify it fails**

Run `Tests/Character_Chat/test_persona_list_paging.py`. Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement the pure helper**

Create `tldw_chatbook/Character_Chat/persona_list_paging.py`:

```python
"""Pure filter/sort/paginate helper for the file-backed persona profile list (P3a).

Personas have no FTS and no tags, so this is a straight in-memory pass over the
(≤100) profiles the scope service returns — no DB, no backend edit.
"""

from __future__ import annotations

from typing import Any


def _modified_key(p: dict[str, Any]) -> str:
    for key in ("updated_at", "last_modified", "modified_at"):
        if p.get(key):
            return str(p[key])
    return str(p.get("created_at") or "")


_SORTS = {
    "name_asc": (lambda p: str(p.get("name") or "").lower(), False),
    "created_desc": (lambda p: str(p.get("created_at") or ""), True),
    "modified_desc": (_modified_key, True),
}


def page_persona_profiles(
    profiles: list[dict[str, Any]],
    *,
    search_term: str | None,
    sort_key: str,
    offset: int,
    page_size: int,
) -> tuple[list[dict[str, Any]], int]:
    """Return ``(page_rows, filtered_total)`` for the persona list.

    Search is a case-insensitive substring over name + description. ``sort_key``
    outside ``_SORTS`` (e.g. "relevance") falls back to "name_asc".
    """
    rows = list(profiles or [])
    term = (search_term or "").strip().lower()
    if term:
        rows = [
            p for p in rows
            if term in str(p.get("name") or "").lower()
            or term in str(p.get("description") or "").lower()
        ]
    filtered_total = len(rows)
    key, reverse = _SORTS.get(sort_key, _SORTS["name_asc"])
    rows = sorted(rows, key=key, reverse=reverse)
    start = max(0, offset)
    return rows[start:start + page_size], filtered_total
```

- [ ] **Step 4: Run — verify it passes**

Run `Tests/Character_Chat/test_persona_list_paging.py`. Expected: PASS (6). If the persona modified-timestamp key differs from `updated_at`/`last_modified`/`modified_at`, add it to `_modified_key` and the test payload.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Character_Chat/persona_list_paging.py Tests/Character_Chat/test_persona_list_paging.py
git commit -m "feat(personas): P3a Task 2 — pure persona list filter/sort/paginate helper"
```

---

## Task 3: Pane controls (sort, tag, page bar) + messages + tag picker

**Files:**
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_messages.py`
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_library_pane.py`
- Create: `tldw_chatbook/Widgets/Persona_Widgets/tag_filter_picker.py`
- Test: `Tests/UI/test_personas_library_pane_paging.py`

**Interfaces:**
- Consumes: `LibraryRow` (existing).
- Produces (messages): `PersonaSortCycleRequested()` (bare intent), `PersonaTagFilterRequested()` (bare intent), `PersonaPageChanged(delta: int)` (−1 / +1).
- Produces (pane methods): `set_sort_label(text: str)`, `set_tag_label(text: str)`; `update_rows(...)` gains `page_offset: int | None = None, page_size: int | None = None`; `set_mode` gates the sort button (characters+personas), tag button (characters only), and the page bar is auto-managed by `update_rows`.
- Produces: `TagFilterPicker(ModalScreen[str | None])` — `__init__(self, tags: list[str], current: str | None)`; dismisses with the chosen tag string, `None` for "All (clear)", and does not dismiss with a sentinel on cancel/escape (Escape dismisses with a distinct `_CANCEL` sentinel — see below).

**Design note (refinement over the spec's message names):** the pane posts *bare intents*; the screen owns all search-awareness, the tag list, and the sort cycle/labels (it has the DB). This keeps the pane a dumb view, matching the existing `PersonaActionRequested` pattern.

- [ ] **Step 1: Add the messages**

In `personas_messages.py`, mirror the existing `PersonaSearchChanged`/`PersonaActionRequested` `Message` subclasses and add:

```python
class PersonaSortCycleRequested(Message):
    """The user asked to advance the library sort (screen decides the next key)."""


class PersonaTagFilterRequested(Message):
    """The user asked to open the tag filter (characters only)."""


class PersonaPageChanged(Message):
    """The user asked to move the library page window."""

    def __init__(self, delta: int) -> None:
        super().__init__()
        self.delta = delta
```

Export them in that module's `__all__` if it has one.

- [ ] **Step 2: Write the failing pane tests**

Create `Tests/UI/test_personas_library_pane_paging.py`. Mirror any existing `PersonasLibraryPane` host-app test (grep `Tests/UI` for `PersonasLibraryPane`); if none, use this harness:

```python
import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Static
from tldw_chatbook.Widgets.Persona_Widgets.personas_library_pane import (
    PersonasLibraryPane, LibraryRow,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_messages import (
    PersonaSortCycleRequested, PersonaTagFilterRequested, PersonaPageChanged,
)


class _Host(App):
    def __init__(self):
        super().__init__()
        self.events = []

    def compose(self) -> ComposeResult:
        yield PersonasLibraryPane(id="pane")

    def on_persona_sort_cycle_requested(self, m): self.events.append(("sort", None))
    def on_persona_tag_filter_requested(self, m): self.events.append(("tag", None))
    def on_persona_page_changed(self, m): self.events.append(("page", m.delta))


def _rows(n):
    return tuple(LibraryRow(item_id=str(i), kind="character", name=f"c{i:03d}") for i in range(n))


@pytest.mark.asyncio
async def test_page_bar_shows_when_total_exceeds_page_size():
    app = _Host()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasLibraryPane)
        await pane.update_rows(_rows(50), total=130, noun="characters", page_offset=0, page_size=50)
        info = app.query_one("#personas-library-page-info", Static)
        assert "1-50 of 130" in str(info.render())
        assert app.query_one("#personas-library-prev", Button).disabled is True
        assert app.query_one("#personas-library-next", Button).disabled is False
        assert app.query_one("#personas-library-pagebar").display is True


@pytest.mark.asyncio
async def test_page_bar_hidden_when_fits_one_page():
    app = _Host()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasLibraryPane)
        await pane.update_rows(_rows(5), total=5, noun="characters", page_offset=0, page_size=50)
        assert app.query_one("#personas-library-pagebar").display is False


@pytest.mark.asyncio
async def test_next_prev_post_page_changed():
    app = _Host()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasLibraryPane)
        await pane.update_rows(_rows(50), total=130, noun="characters", page_offset=50, page_size=50)
        await pilot.click("#personas-library-next")
        await pilot.click("#personas-library-prev")
        assert ("page", 1) in app.events and ("page", -1) in app.events


@pytest.mark.asyncio
async def test_sort_and_tag_buttons_post_intents():
    app = _Host()
    async with app.run_test() as pilot:
        app.query_one(PersonasLibraryPane).set_mode("characters")
        await pilot.click("#personas-library-sort")
        await pilot.click("#personas-library-tag")
        assert ("sort", None) in app.events and ("tag", None) in app.events


@pytest.mark.asyncio
async def test_tag_button_hidden_for_personas_visible_for_characters():
    app = _Host()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasLibraryPane)
        pane.set_mode("personas")
        assert app.query_one("#personas-library-tag", Button).display is False
        assert app.query_one("#personas-library-sort", Button).display is True
        pane.set_mode("characters")
        assert app.query_one("#personas-library-tag", Button).display is True


@pytest.mark.asyncio
async def test_sort_page_hidden_for_lore():
    app = _Host()
    async with app.run_test() as pilot:
        pane = app.query_one(PersonasLibraryPane)
        pane.set_mode("lore")
        assert app.query_one("#personas-library-sort", Button).display is False
        assert app.query_one("#personas-library-tag", Button).display is False
        # dict/lore call update_rows WITHOUT page kwargs → no page bar, plain count
        await pane.update_rows((), total=0, noun="world books")
        assert app.query_one("#personas-library-pagebar").display is False
```

- [ ] **Step 3: Run — verify it fails**

Run `Tests/UI/test_personas_library_pane_paging.py`. Expected: FAIL — no `#personas-library-sort` / `page-info` widgets.

- [ ] **Step 4: Add the controls to the pane**

In `personas_library_pane.py` `compose` (after the `#personas-library-toolbar` Horizontal, ~:142, before `yield ListView(...)`), add a filter bar:

```python
        with Horizontal(id="personas-library-filterbar", classes="ds-toolbar"):
            yield Button(
                "Sort: Name", id="personas-library-sort",
                tooltip="Cycle the list sort order.",
                classes="console-action-secondary",
            )
            yield Button(
                "Tag: All", id="personas-library-tag",
                tooltip="Filter characters by tag.",
                classes="console-action-secondary",
            )
```

After `yield ListView(id="personas-library-rows")` and before the count Static, add the page bar (hidden by default):

```python
        with Horizontal(id="personas-library-pagebar", classes="ds-toolbar"):
            yield Button("<", id="personas-library-prev", compact=True,
                         classes="console-action-secondary")
            yield Static("", id="personas-library-page-info",
                         classes="destination-purpose")
            yield Button(">", id="personas-library-next", compact=True,
                         classes="console-action-secondary")
```

In `on_mount`, hide the page bar initially and default sort/tag button visibility for characters mode:

```python
    def on_mount(self) -> None:
        """Initialize control visibility for default characters mode."""
        self.query_one("#personas-library-duplicate", Button).display = False
        self.query_one("#personas-library-pagebar").display = False
```

Add label setters + intent handlers (near the other `@on(Button.Pressed, ...)` handlers):

```python
    def set_sort_label(self, text: str) -> None:
        self.query_one("#personas-library-sort", Button).label = text

    def set_tag_label(self, text: str) -> None:
        self.query_one("#personas-library-tag", Button).label = text

    @on(Button.Pressed, "#personas-library-sort")
    def _sort_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(PersonaSortCycleRequested())

    @on(Button.Pressed, "#personas-library-tag")
    def _tag_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(PersonaTagFilterRequested())

    @on(Button.Pressed, "#personas-library-prev")
    def _prev_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(PersonaPageChanged(-1))

    @on(Button.Pressed, "#personas-library-next")
    def _next_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(PersonaPageChanged(1))
```

Import the three messages at the top (`from .personas_messages import (... PersonaSortCycleRequested, PersonaTagFilterRequested, PersonaPageChanged)`).

- [ ] **Step 5: Gate the new controls in `set_mode`**

Extend `set_mode` (~:146) — sort + page apply to characters+personas, tag characters-only:

```python
        sort_visible = mode in ("characters", "personas")
        self.query_one("#personas-library-sort", Button).display = sort_visible
        self.query_one("#personas-library-tag", Button).display = mode == "characters"
        if not sort_visible:
            # dict/lore never paginate — keep the page bar hidden.
            self.query_one("#personas-library-pagebar").display = False
```

- [ ] **Step 6: Extend `update_rows` with the page window**

Add the kwargs to the signature (after `recovery_id`):

```python
        page_offset: int | None = None,
        page_size: int | None = None,
```

Replace the trailing count block (the `if recovery_copy: ... else: ...` at ~:252-262) with page-aware logic:

```python
        pagebar = self.query_one("#personas-library-pagebar")
        count_static = self.query_one("#personas-library-count", Static)
        paginated = page_offset is not None and page_size is not None
        if recovery_copy:
            pagebar.display = False
            count_static.update(f"{noun.capitalize()} unavailable")
        elif paginated and total > page_size:
            start = page_offset + 1 if total else 0
            end = page_offset + len(rows)
            self.query_one("#personas-library-page-info", Static).update(
                f"{start}-{end} of {total} {noun}"
            )
            self.query_one("#personas-library-prev", Button).disabled = page_offset <= 0
            self.query_one("#personas-library-next", Button).disabled = (
                page_offset + page_size >= total
            )
            pagebar.display = True
            count_static.update("")
        else:
            pagebar.display = False
            if filtered and filtered_total_unbounded:
                match_word = "match" if len(rows) == 1 else "matches"
                count_static.update(
                    f"Showing {len(rows)} {_singular_noun(noun)} {match_word} from full library"
                )
            elif filtered:
                count_static.update(f"{len(rows)} of {total} {noun}")
            else:
                count_static.update(f"{total} {noun}")
```

- [ ] **Step 7: Create the tag picker**

Create `tldw_chatbook/Widgets/Persona_Widgets/tag_filter_picker.py`. Read the existing `ConversationAttachPicker` (`tldw_chatbook/Widgets/Persona_Widgets/conversation_attach_picker.py`, from P2e) and mirror its structure — a `ModalScreen[str | None]` with a search `Input` + a filterable `ListView`. First row is "All (clear filter)" → dismiss `None`; each tag row dismisses that tag string; Escape dismisses the sentinel `TagFilterPicker.CANCEL`. Keep ids generic (`#tag-filter-list`, `#tag-filter-search`).

```python
"""Modal tag picker for the characters library tag filter (P3a)."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, ListItem, ListView, Static


class TagFilterPicker(ModalScreen[str | None]):
    CANCEL = object()  # distinct from None ("All") so cancel != clear-filter

    def __init__(self, tags: list[str], current: str | None) -> None:
        super().__init__()
        self._tags = list(tags)
        self._current = current

    def compose(self) -> ComposeResult:
        with Vertical(id="tag-filter-dialog"):
            yield Label("Filter by tag")
            yield Input(placeholder="Filter tags...", id="tag-filter-search")
            yield ListView(id="tag-filter-list")
            yield Static("Enter to pick · Esc to cancel", classes="destination-purpose")

    def on_mount(self) -> None:
        self._populate(self._tags)

    def _populate(self, tags: list[str]) -> None:
        lv = self.query_one("#tag-filter-list", ListView)
        lv.clear()
        lv.append(ListItem(Static("All (clear filter)", markup=False), id="tag-all"))
        for t in tags:
            safe = t.replace(" ", "-")
            lv.append(ListItem(Static(t, markup=False), id=f"tag-{safe}"))

    def on_input_changed(self, event: Input.Changed) -> None:
        q = event.value.strip().lower()
        self._populate([t for t in self._tags if q in t.lower()] if q else self._tags)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item_id = event.item.id or ""
        if item_id == "tag-all":
            self.dismiss(None)
            return
        # Recover the exact tag from the visible label (ids are lossy).
        label = str(event.item.query_one(Static).renderable)
        self.dismiss(label)

    def on_key(self, event) -> None:
        if event.key == "escape":
            event.stop()
            self.dismiss(self.CANCEL)
```

(If `ConversationAttachPicker` uses a different dismiss/label-recovery idiom, follow it instead — the goal is a working picker consistent with the codebase.)

- [ ] **Step 8: Run the pane tests — verify they pass**

Run `Tests/UI/test_personas_library_pane_paging.py`. Expected: PASS (6). Fix any DEFAULT_CSS needed so the filter bar / page bar don't collapse (mirror `.ds-toolbar` usage already in the pane).

- [ ] **Step 9: Commit**

```bash
git add tldw_chatbook/Widgets/Persona_Widgets/personas_messages.py tldw_chatbook/Widgets/Persona_Widgets/personas_library_pane.py tldw_chatbook/Widgets/Persona_Widgets/tag_filter_picker.py Tests/UI/test_personas_library_pane_paging.py
git commit -m "feat(personas): P3a Task 3 — library sort/tag/page controls + tag picker"
```

---

## Task 4: Screen wiring + cache-assumption fixes + integration

**Files:**
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_state.py` (add `sort_key`, `tag_filter`, `page_offset`)
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Test: `Tests/UI/test_personas_library_scale.py` (new; also confirm/extend the existing personas-screen test module — grep `Tests/UI` for `PersonasScreen` host apps and mirror the fixture)

**Interfaces:**
- Consumes: Task 1 (`get_character_page_for_ui`, `count_character_page`, `list_character_tags`), Task 2 (`page_persona_profiles`), Task 3 (pane page kwargs, `set_sort_label`/`set_tag_label`, `PersonaSortCycleRequested`/`PersonaTagFilterRequested`/`PersonaPageChanged`, `TagFilterPicker`).

**Verified facts / fix sites (re-verify line numbers fresh):**
- `PersonasWorkbenchState` (personas_state.py:36) has `active_mode`, `search_query`, `selected_entity_kind`, `selected_entity_id`.
- `_render_library_rows` (personas_screen.py:763) — replace the `total=len(self._characters)` + in-memory/FTS branch (~:778-802) with the paged query.
- `_render_profile_rows` (~:885) — replace its `total=len` + in-memory filter (~:900-909) with `page_persona_profiles`.
- `_build_library_rows` (~:733) — add a `meta` line for characters.
- `refresh_character_library_list` (~:717) — `self._characters` is now the CURRENT PAGE only.
- `_handle_search_changed` (~:942) — resets and re-renders; must reset `page_offset` and re-fetch count.
- `_character_record` cache-scan (~:829) → callers at import (~:3797) and save (~:4565): switch name resolution to `_full_character_record`/by-id.
- Status string (~:1329) `len(self._characters)` → use the cached total.
- `pre_import_ids` (~:3766 → used ~:3803) — replace page-cache membership with a by-id existence check.
- `PERSONAS_LIBRARY_PAGE_SIZE` — add near `PERSONAS_SEARCH_DEBOUNCE_SECONDS` (~:178).
- Sort cycle order + labels: `[("name_asc","Name"), ("modified_desc","Recent edit"), ("created_desc","Recent add")]`; when a **character** search is active, prepend `("relevance","Relevance")` and make it the default the first time a search begins.

- [ ] **Step 1: Add state fields**

In `personas_state.py` (dataclass at :36), add fields (with the module's existing default style):

```python
    sort_key: str = "name_asc"
    tag_filter: str | None = None
    page_offset: int = 0
```

If `switch_mode`/`clear_selection` reset search, also reset `page_offset = 0` and `tag_filter = None` there (read the methods first).

- [ ] **Step 2: Write the failing integration tests**

Create `Tests/UI/test_personas_library_scale.py`. Mirror the existing personas-screen host-app fixture (it wires a real `CharactersRAGDB` + the scope service; find it by grepping `Tests/UI` for `PersonasScreen`). Sketch (adapt names to the real fixture):

```python
import pytest

# Uses the project's PersonasScreen host-app fixture: `personas_app` yields a
# running app whose screen has a real chachanotes_db. Seed >2 pages, drive the
# library, assert paging/sort/tag.

@pytest.mark.asyncio
async def test_first_page_and_next(personas_app):
    app, screen, db = personas_app
    for i in range(130):
        db.add_character_card({"name": f"c{i:03d}", "tags": []})
    await screen._reload_character_page()          # helper added in Step 4
    pane = screen.query_one_pane()                 # PersonasLibraryPane
    assert screen.state.page_offset == 0
    # page bar visible, 50 rows
    assert len(pane_visible_rows(pane)) == 50
    screen.state.page_offset = 0
    await screen._on_page_changed_delta(1)         # simulate next
    assert screen.state.page_offset == 50


@pytest.mark.asyncio
async def test_tag_filter_narrows_count(personas_app):
    app, screen, db = personas_app
    for i in range(10):
        db.add_character_card({"name": f"h{i}", "tags": ["hero"] if i < 3 else ["v"]})
    await screen._apply_tag_filter("hero")
    assert screen._character_total == 3
    assert screen.state.page_offset == 0


@pytest.mark.asyncio
async def test_sort_change_resets_offset(personas_app):
    app, screen, db = personas_app
    for i in range(130):
        db.add_character_card({"name": f"c{i:03d}", "tags": []})
    await screen._reload_character_page()
    screen.state.page_offset = 50
    await screen._cycle_sort()
    assert screen.state.page_offset == 0


@pytest.mark.asyncio
async def test_import_offpage_name_conflict_message(personas_app, tmp_path):
    # Seed a name-conflict character that will NOT be on page 1, import a card
    # with the same name, assert the notification says "already exists"/updated,
    # not "imported new". (Drives the pre_import_ids fix.)
    ...
```

Because the exact host-app fixture and helper names depend on the existing test module, **the implementer's first action is to read the current personas-screen tests and shape these to match** — keep the assertions (page window, offset reset on filter change, tag count, import-conflict wording).

- [ ] **Step 3: Run — verify it fails**

Run `Tests/UI/test_personas_library_scale.py`. Expected: FAIL (missing helpers/behaviour).

- [ ] **Step 4: Rewire the character render path**

Add the constant near `PERSONAS_SEARCH_DEBOUNCE_SECONDS`:

```python
PERSONAS_LIBRARY_PAGE_SIZE = 50
```

Add instance state in `__init__` (near where `self._characters` is initialized, ~:475): `self._character_total = 0`, `self._count_cache_key: tuple | None = None`, `self._character_tags: list[str] = []`.

Add a paged loader (new method) that the screen uses instead of the len/FTS branch. The DB reads go off-thread; count is cached by `(search, tag)`:

```python
    def _character_sort_cycle(self) -> list[tuple[str, str]]:
        base = [("name_asc", "Name"), ("modified_desc", "Recent edit"), ("created_desc", "Recent add")]
        if self.state.search_query:
            return [("relevance", "Relevance"), *base]
        return base

    def _fts_match_query(self) -> str | None:
        term = (self.state.search_query or "").strip()
        if not term:
            return None
        escaped = term.replace('"', '""')
        return f'"{escaped}"*'

    async def _reload_character_page(self, *, reset_offset: bool = False) -> None:
        if reset_offset:
            self.state.page_offset = 0
        mode, query = self.state.active_mode, self.state.search_query
        sort_key, tag = self.state.sort_key, self.state.tag_filter
        offset = self.state.page_offset
        search = self._fts_match_query()
        db = self.app.chachanotes_db  # confirm the accessor used elsewhere in this screen
        cache_key = (search, tag)
        try:
            if self._count_cache_key != cache_key:
                self._character_total = await asyncio.to_thread(
                    ccl.count_character_page, db, search_term=search, tag=tag
                )
                self._count_cache_key = cache_key
            # clamp offset into range
            if offset > 0 and offset >= self._character_total:
                offset = max(0, ((self._character_total - 1) // PERSONAS_LIBRARY_PAGE_SIZE)
                             * PERSONAS_LIBRARY_PAGE_SIZE)
                self.state.page_offset = offset
            records = await asyncio.to_thread(
                ccl.get_character_page_for_ui, db,
                limit=PERSONAS_LIBRARY_PAGE_SIZE, offset=offset,
                order_by=sort_key, search_term=search, tag=tag,
            )
        except Exception as exc:
            logger.opt(exception=True).warning("Character page load failed.")
            self._notify(f"Could not load characters: {exc}", "error")
            return
        # freshness guard: state may have moved while off-thread
        if not self.is_mounted or self.state.active_mode != mode \
           or self.state.search_query != query or self.state.sort_key != sort_key \
           or self.state.tag_filter != tag or self.state.page_offset != offset:
            return
        self._characters = records
        rows = self._build_library_rows(records, "character")
        library = self.query_one(PersonasLibraryPane)
        async with self._render_lock:
            await library.update_rows(
                rows, total=self._character_total, noun="characters",
                page_offset=offset, page_size=PERSONAS_LIBRARY_PAGE_SIZE,
            )
            library.set_sort_label(f"Sort: {dict(self._character_sort_cycle())[sort_key]}"
                                   if sort_key in dict(self._character_sort_cycle())
                                   else "Sort: Name")
            library.set_tag_label(f"Tag: {tag}" if tag else "Tag: All")
            if self.state.selected_entity_kind == "character" and self.state.selected_entity_id:
                library.mark_active_row("character", self.state.selected_entity_id)
```

(`ccl` = the `Character_Chat_Lib` import alias already used in the screen; confirm it, else import the three functions.) Replace `_render_library_rows`'s body so `refresh_character_library_list` and the debounced search path call `_reload_character_page`. Keep `_render_library_rows` as a thin `await self._reload_character_page()` (so existing callers/args still resolve) OR redirect callers — read the call sites first and pick the lower-churn option; the freshness guard here replaces `_library_render_snapshot_is_current` for the character path.

Enrich `_build_library_rows` to add a meta line for characters (last-modified date). Since it's shared with personas, branch on kind:

```python
    @staticmethod
    def _build_library_rows(records: list[dict], kind: str) -> tuple[LibraryRow, ...]:
        rows = []
        for record in records:
            if record.get("id") is None:
                continue
            meta = None
            if kind == "character":
                lm = str(record.get("last_modified") or "")
                meta = lm[:10] if lm else None  # YYYY-MM-DD
            rows.append(LibraryRow(
                item_id=str(record.get("id")), kind=kind,
                name=str(record.get("name") or "Unnamed"), meta=meta,
            ))
        return tuple(rows)
```

- [ ] **Step 5: Rewire the persona render path**

In `_render_profile_rows` (~:885), replace the `total=len` + substring filter with the Task 2 helper + page window (personas paginate, no tag, relevance N/A):

```python
        from tldw_chatbook.Character_Chat.persona_list_paging import page_persona_profiles
        sort_key = self.state.sort_key if self.state.sort_key != "relevance" else "name_asc"
        page_rows, total = page_persona_profiles(
            self._profiles, search_term=self.state.search_query,
            sort_key=sort_key, offset=self.state.page_offset,
            page_size=PERSONAS_LIBRARY_PAGE_SIZE,
        )
        rows = self._build_library_rows(page_rows, "persona_profile")
        # ... update_rows with page_offset=self.state.page_offset,
        #     page_size=PERSONAS_LIBRARY_PAGE_SIZE, noun="persona profiles",
        #     plus the existing recovery_copy handling.
```

Keep the existing recovery-copy path and `mark_active_row`. (Personas load ≤100 into `self._profiles` already, so no off-thread change is needed — filtering is in-memory.)

- [ ] **Step 6: Add the sort / tag / page handlers**

Add `@on` handlers for the three new messages:

```python
    @on(PersonaSortCycleRequested)
    async def _handle_sort_cycle(self, message) -> None:
        message.stop()
        await self._cycle_sort()

    async def _cycle_sort(self) -> None:
        cycle = [k for k, _ in self._character_sort_cycle()]
        cur = self.state.sort_key if self.state.sort_key in cycle else cycle[0]
        self.state.sort_key = cycle[(cycle.index(cur) + 1) % len(cycle)]
        self.state.page_offset = 0
        await self._reload_active_library()

    @on(PersonaTagFilterRequested)
    async def _handle_tag_filter(self, message) -> None:
        message.stop()
        if self.state.active_mode != "characters" or self._io_dialog_active:
            return
        self._io_dialog_active = True
        self.run_worker(self._tag_filter_worker(), group="personas-io", exit_on_error=False)

    async def _tag_filter_worker(self) -> None:
        try:
            db = self.app.chachanotes_db
            tags = await asyncio.to_thread(ccl.list_character_tags, db)
            picked = await self.app.push_screen_wait(
                TagFilterPicker(tags, self.state.tag_filter)
            )
            if picked is TagFilterPicker.CANCEL:
                return
            await self._apply_tag_filter(picked)  # None clears
        except Exception as exc:
            logger.opt(exception=True).warning("Tag filter failed.")
            self._notify(f"Tag filter failed: {exc}", "error")
        finally:
            self._io_dialog_active = False

    async def _apply_tag_filter(self, tag: str | None) -> None:
        self.state.tag_filter = tag
        self.state.page_offset = 0
        self._count_cache_key = None  # (search,tag) changed → recount
        await self._reload_character_page()

    @on(PersonaPageChanged)
    async def _handle_page_changed(self, message) -> None:
        message.stop()
        await self._on_page_changed_delta(message.delta)

    async def _on_page_changed_delta(self, delta: int) -> None:
        new_offset = self.state.page_offset + delta * PERSONAS_LIBRARY_PAGE_SIZE
        total = self._character_total if self.state.active_mode == "characters" else len(self._profiles)
        if new_offset < 0 or new_offset >= max(1, total):
            return
        self.state.page_offset = new_offset
        await self._reload_active_library()

    async def _reload_active_library(self) -> None:
        if self.state.active_mode == "characters":
            await self._reload_character_page()
        elif self.state.active_mode == "personas":
            await self._render_profile_rows()
```

In `_handle_search_changed` (~:942) and the debounced render path: whenever the search query changes, set `self.state.page_offset = 0` and `self._count_cache_key = None` before re-rendering so the count is recomputed and paging restarts. (Read the debounce flow first — the reset belongs where `state.search_query` is assigned.)

- [ ] **Step 7: Fix the five cache-assumption sites**

1. **Status string (~:1329):** replace `len(self._characters)` with `self._character_total`:
   ```python
   return f"Characters: {self._character_total} | Source: Local | Attachments: Console"
   ```
2. **`_character_record` callers (import ~:3797, save ~:4565):** resolve the name by id instead of scanning the page cache. Use the handler's loaded record after selection, or fetch by id:
   ```python
   # after load/select, the handler holds the record:
   loaded = self._full_character_record(imported_id)
   name = str((loaded or {}).get("name") or "Imported character")
   ```
   (For the save site, mirror with `saved_id`.) You may keep `_character_record` for any remaining page-local use, but these two must not depend on the full library being cached.
3. **`pre_import_ids` (~:3766 / used ~:3803):** replace the page-cache id snapshot with a by-id existence check. Read the importer to see if it already signals "conflict" (it returns the EXISTING id on a name clash). Simplest correct form — snapshot existence of the *result* id before deciding the message:
   ```python
   existed_before = bool(await asyncio.to_thread(ccp_character_handler.fetch_character_by_id, imported_id))
   # ... run import ...
   # message: "Updated existing" if existed_before else "Imported"
   ```
   Confirm ordering (the snapshot must be taken BEFORE the import so a genuine new id reads as not-existing). If the importer already returns a conflict flag, prefer that over an extra query.
4. **`_apply_mode` control gating + page-0 load:** on entering characters, call `library.set_mode("characters")` (already happens), reset `state.page_offset = 0`, `self._count_cache_key = None`, and `await self._reload_character_page()`. On entering personas, reset offset and `await self._render_profile_rows()`. Read `_apply_mode` (~:1064) and add these to the character/persona branches.

- [ ] **Step 8: Run the integration tests — verify they pass**

Run `Tests/UI/test_personas_library_scale.py` and the Task 3 pane tests together. Expected: PASS. Iterate on helper/fixture names to match the real personas-screen test harness.

- [ ] **Step 9: Run the focused suite (no broad sweep)**

Run exactly these (scoped to this feature):
```
Tests/DB/test_character_cards_paging.py
Tests/Character_Chat/test_persona_list_paging.py
Tests/UI/test_personas_library_pane_paging.py
Tests/UI/test_personas_library_scale.py
```
Plus the existing personas-screen + library-pane test modules you mirrored (name them explicitly). Expected: PASS. Also run `python -c "import tldw_chatbook.app"` (with the test-env prefix) to confirm the app imports.

- [ ] **Step 10: Commit**

```bash
git add tldw_chatbook/Widgets/Persona_Widgets/personas_state.py tldw_chatbook/UI/Screens/personas_screen.py Tests/UI/test_personas_library_scale.py
git commit -m "feat(personas): P3a Task 4 — paged library wiring, sort/tag/page handlers, cache-assumption fixes"
```

---

## Self-Review Notes (author)

- **Spec coverage:** pagination (Tasks 1/4), sort whitelist (1/3/4), tag filter chars-only (1/3/4), FTS-preserve search (1/4), count caching (4), personas in-screen (2/4), dict-lore gated off (3), 5 cache-assumption fixes (4), no-migration/parameterized-SQL/off-thread/json-guard constraints — all mapped.
- **Type consistency:** `list_character_cards_page`/`count_character_cards`/`list_distinct_character_tags` (DB) → `get_character_page_for_ui`/`count_character_page`/`list_character_tags` (lib) → `_reload_character_page` (screen); `page_persona_profiles` (Task 2) consumed in Step 5; `PersonaSortCycleRequested`/`PersonaTagFilterRequested`/`PersonaPageChanged(delta)` + `set_sort_label`/`set_tag_label` + `update_rows(page_offset,page_size)` (Task 3) consumed in Task 4; `TagFilterPicker.CANCEL` sentinel handled. Sort keys identical across DB whitelist, persona `_SORTS` (minus relevance), and the screen cycle.
- **Known plan-time confirmations (read fresh):** the `CharactersRAGDB` constructor + `execute_query` accessor; the persona modified-timestamp key; the screen's DB accessor (`self.app.chachanotes_db` vs a handler); the existing personas-screen test host-app fixture; whether `_render_library_rows` is better replaced or redirected; the importer's conflict-return contract.
