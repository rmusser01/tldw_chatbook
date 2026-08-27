"""Seek-pagination coverage for local character conversations."""

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, InputError


@pytest.fixture
def db(tmp_path):
    database = CharactersRAGDB(tmp_path / "chacha.db", client_id="seek-test")
    yield database
    database.close_connection()


def _seed_conversation(
    db,
    conversation_id: str,
    last_modified: str,
    *,
    character_id: int = 1,
    scope_type: str = "global",
    deleted: bool = False,
) -> None:
    data = {
        "id": conversation_id,
        "title": conversation_id,
        "character_id": character_id,
        "scope_type": scope_type,
    }
    if scope_type == "workspace":
        data["workspace_id"] = "workspace-1"
    db.add_conversation(data)
    with db.transaction() as conn:
        conn.execute(
            "UPDATE conversations SET last_modified = ?, deleted = ? WHERE id = ?",
            (last_modified, int(deleted), conversation_id),
        )


def _ids(rows) -> list[str]:
    return [row["id"] for row in rows]


def test_seek_pages_order_ties_and_filter_nonmatching_rows(db):
    visible = [
        ("newest", "2026-08-27T05:00:00Z"),
        ("tie-z", "2026-08-27T04:00:00Z"),
        ("tie-a", "2026-08-27T04:00:00Z"),
        ("older-3", "2026-08-27T03:00:00Z"),
        ("older-2", "2026-08-27T02:00:00Z"),
        ("older-1", "2026-08-27T01:00:00Z"),
    ]
    for conversation_id, last_modified in visible:
        _seed_conversation(db, conversation_id, last_modified)

    other_character_id = db.add_character_card({"name": "Other Character"})
    _seed_conversation(
        db,
        "other-character",
        "2026-08-27T09:00:00Z",
        character_id=other_character_id,
    )
    _seed_conversation(
        db,
        "workspace",
        "2026-08-27T08:00:00Z",
        scope_type="workspace",
    )
    _seed_conversation(
        db,
        "deleted",
        "2026-08-27T07:00:00Z",
        deleted=True,
    )

    first_page = db.get_conversations_for_character(1, limit=3)
    assert _ids(first_page) == ["newest", "tie-z", "tie-a"]

    cursor = first_page[-1]
    second_page = db.get_conversations_for_character(
        1,
        limit=3,
        before_last_modified=cursor["last_modified"],
        before_id=cursor["id"],
    )
    assert _ids(second_page) == ["older-3", "older-2", "older-1"]
    assert set(_ids(first_page)).isdisjoint(_ids(second_page))


def test_legacy_positional_limit_and_offset_return_expected_slice(db):
    for conversation_id, last_modified in [
        ("newest", "2026-08-27T05:00:00Z"),
        ("tie-z", "2026-08-27T04:00:00Z"),
        ("tie-a", "2026-08-27T04:00:00Z"),
        ("older", "2026-08-27T03:00:00Z"),
    ]:
        _seed_conversation(db, conversation_id, last_modified)

    assert _ids(db.get_conversations_for_character(1, 2, 1)) == [
        "tie-z",
        "tie-a",
    ]


def test_seek_cursor_ignores_newer_insert_between_reads(db):
    for conversation_id, last_modified in [
        ("seen-2", "2026-08-27T05:00:00Z"),
        ("seen-1", "2026-08-27T04:00:00Z"),
        ("older-2", "2026-08-27T03:00:00Z"),
        ("older-1", "2026-08-27T02:00:00Z"),
    ]:
        _seed_conversation(db, conversation_id, last_modified)

    first_page = db.get_conversations_for_character(1, limit=2)
    cursor = first_page[-1]
    _seed_conversation(db, "inserted-newer", "2026-08-27T06:00:00Z")

    second_page = db.get_conversations_for_character(
        1,
        limit=2,
        before_last_modified=cursor["last_modified"],
        before_id=cursor["id"],
    )
    assert _ids(first_page + second_page) == [
        "seen-2",
        "seen-1",
        "older-2",
        "older-1",
    ]


def test_seek_cursor_does_not_skip_after_traversed_row_is_deleted(db):
    for conversation_id, last_modified in [
        ("seen-2", "2026-08-27T05:00:00Z"),
        ("seen-1", "2026-08-27T04:00:00Z"),
        ("older-2", "2026-08-27T03:00:00Z"),
        ("older-1", "2026-08-27T02:00:00Z"),
    ]:
        _seed_conversation(db, conversation_id, last_modified)

    first_page = db.get_conversations_for_character(1, limit=2)
    cursor = first_page[-1]
    with db.transaction() as conn:
        conn.execute(
            "UPDATE conversations SET deleted = 1 WHERE id = ?",
            (first_page[0]["id"],),
        )

    second_page = db.get_conversations_for_character(
        1,
        limit=2,
        before_last_modified=cursor["last_modified"],
        before_id=cursor["id"],
    )
    assert _ids(second_page) == ["older-2", "older-1"]


@pytest.mark.parametrize(
    "cursor",
    [
        {"before_last_modified": "2026-08-27T04:00:00Z"},
        {"before_id": "seen-1"},
    ],
)
def test_partial_seek_cursor_fails_before_sql(db, monkeypatch, cursor):
    def fail_sql(*_args, **_kwargs):
        raise AssertionError("SQL should not run")

    monkeypatch.setattr(db, "execute_query", fail_sql)

    with pytest.raises(InputError):
        db.get_conversations_for_character(1, **cursor)


def test_seek_cursor_with_nonzero_offset_fails_before_sql(db, monkeypatch):
    def fail_sql(*_args, **_kwargs):
        raise AssertionError("SQL should not run")

    monkeypatch.setattr(db, "execute_query", fail_sql)

    with pytest.raises(InputError):
        db.get_conversations_for_character(
            1,
            50,
            1,
            before_last_modified="2026-08-27T04:00:00Z",
            before_id="seen-1",
        )
