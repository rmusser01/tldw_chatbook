# test_character_cards_paging.py
# Description: RED-first coverage for P3a Task 1 — paged/sorted/tag-filtered
# character list + count + distinct tags DB seam on CharactersRAGDB, plus the
# UI-shaping wrappers in Character_Chat_Lib.
"""
Adds ``list_character_cards_page`` / ``count_character_cards`` /
``list_distinct_character_tags`` to ``CharactersRAGDB`` and thin UI wrappers
in ``Character_Chat_Lib``. These are read-only additions consumed by later
Roleplay P3a tasks (pane controls, screen wiring) -- nothing consumes them
yet, so this suite is the sole verification.

Key correctness contracts under test:
- Pagination windows are disjoint and respect LIMIT/OFFSET.
- COUNT matches the full unpaginated result set for the same filter.
- Tag filtering narrows both the list and the count identically.
- Search composes with tag filter and sort (mirrors the FTS-join pattern of
  the existing ``search_character_cards``).
- Sort whitelist: known keys apply the right ORDER BY; unknown keys fall
  back to name_asc.
- Distinct tags are case-insensitively sorted and de-duplicated.
- NULL and non-JSON ``tags`` values never raise (json_each must be
  json-valid-guarded) across list/count/distinct-tags.
"""

import time

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
    _add(db, "alpha", [])
    # `last_modified` has millisecond precision; a real gap avoids a flaky
    # tie against alpha's creation timestamp on fast (e.g. in-memory-like)
    # SQLite round-trips, which would otherwise land beta's update in the
    # same millisecond as alpha's creation.
    time.sleep(0.02)
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
    # Force a NULL tags row and a non-JSON tags row directly. Each raw UPDATE
    # is committed immediately -- the python sqlite3 driver implicitly opens
    # a transaction on the first DML statement, and `_add` below opens its
    # own explicit transaction via `db.transaction()`, which would otherwise
    # collide with the still-open implicit one ("cannot start a transaction
    # within a transaction").
    conn = db.get_connection()
    conn.execute("UPDATE character_cards SET tags = NULL WHERE id = ?", (good,))
    conn.commit()
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


def test_lib_wrapper_shapes_rows(db):
    from tldw_chatbook.Character_Chat.Character_Chat_Lib import (
        get_character_page_for_ui, count_character_page, list_character_tags,
    )
    _add(db, "Zeta", ["x"])
    rows = get_character_page_for_ui(db, limit=10, offset=0)
    # `description` joined the projection so the character library row can show a
    # recognizable snippet instead of a last-modified date identical on every row.
    assert rows and set(rows[0]) == {
        "id",
        "name",
        "description",
        "last_modified",
        "created_at",
        "tags",
    }
    assert count_character_page(db) == len(get_character_page_for_ui(db, limit=1000, offset=0))
    assert "x" in list_character_tags(db)


# --- task-15474: image-free list/picker projections -----------------------
#
# `character_cards.image` is a BLOB (sometimes multi-MB). `list_character_
# cards_page` backs the Library character list and personas paging, neither
# of which renders avatars -- so by default it must not even fetch the
# `image` column. `include_image=True` is the opt-in escape hatch for a
# caller that genuinely needs it.

_SAMPLE_IMAGE = b"\x89PNG\r\n\x1a\n" + b"fake-png-bytes" * 100


def test_page_excludes_image_column_by_default_no_search(db):
    _add(db, "Imaged", [])
    char_id = db.add_character_card({"name": "Imaged2", "image": _SAMPLE_IMAGE})
    rows = db.list_character_cards_page(limit=100, offset=0)
    assert rows  # sanity: rows were actually returned
    for row in rows:
        assert "image" not in row
    # Confirm the row for the image-carrying card is really in the result
    # set (i.e. the projection excluded the column, not the row).
    assert any(r["id"] == char_id for r in rows)


def test_page_excludes_image_column_by_default_when_searching(db):
    db.add_character_card({"name": "Dragon Imaged", "image": _SAMPLE_IMAGE})
    rows = db.list_character_cards_page(
        limit=100, offset=0, search_term='"Dragon"*'
    )
    assert rows
    for row in rows:
        assert "image" not in row


def test_page_include_image_true_returns_image_bytes(db):
    char_id = db.add_character_card({"name": "Imaged3", "image": _SAMPLE_IMAGE})
    rows = db.list_character_cards_page(limit=100, offset=0, include_image=True)
    row = next(r for r in rows if r["id"] == char_id)
    assert row["image"] == _SAMPLE_IMAGE


def test_page_include_image_true_returns_image_bytes_when_searching(db):
    char_id = db.add_character_card({"name": "Dragon Imaged2", "image": _SAMPLE_IMAGE})
    rows = db.list_character_cards_page(
        limit=100, offset=0, search_term='"Dragon"*', include_image=True
    )
    row = next(r for r in rows if r["id"] == char_id)
    assert row["image"] == _SAMPLE_IMAGE


def test_page_image_free_projection_does_not_regress_other_fields(db):
    """The column-list projection must still surface every non-image field
    a consumer relies on (task-15474 must not silently drop e.g. tags)."""
    char_id = db.add_character_card(
        {
            "name": "Full Fields",
            "description": "desc",
            "personality": "kind",
            "scenario": "a scene",
            "system_prompt": "sp",
            "post_history_instructions": "phi",
            "first_message": "hi",
            "message_example": "ex",
            "creator_notes": "notes",
            "alternate_greetings": ["g1", "g2"],
            "tags": ["x", "y"],
            "creator": "me",
            "character_version": "1.0",
            "extensions": {"a": 1},
        }
    )
    row = next(
        r for r in db.list_character_cards_page(limit=100, offset=0) if r["id"] == char_id
    )
    assert row["description"] == "desc"
    assert row["personality"] == "kind"
    assert row["scenario"] == "a scene"
    assert row["system_prompt"] == "sp"
    assert row["post_history_instructions"] == "phi"
    assert row["first_message"] == "hi"
    assert row["message_example"] == "ex"
    assert row["creator_notes"] == "notes"
    assert row["alternate_greetings"] == ["g1", "g2"]
    assert row["tags"] == ["x", "y"]
    assert row["creator"] == "me"
    assert row["character_version"] == "1.0"
    assert row["extensions"] == {"a": 1}
    assert "created_at" in row and "last_modified" in row
    assert "deleted" in row and "client_id" in row and "version" in row
