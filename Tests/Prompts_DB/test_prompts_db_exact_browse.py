import sqlite3
import uuid

import pytest

from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
from tldw_chatbook.Prompt_Management.prompt_scope_service import LocalPromptService


@pytest.fixture()
def database(tmp_path):
    db = PromptsDatabase(tmp_path / "prompts.db", client_id="browse-test")
    try:
        yield db
    finally:
        db.close_connection()


def _insert_prompt(
    database,
    *,
    name,
    details="",
    author="Author",
    system_prompt="",
    user_prompt="",
    last_modified="2026-08-09T12:00:00.000Z",
    artifact_type="prompt",
):
    cursor = database.get_connection().execute(
        """
        INSERT INTO Prompts (
            name, author, details, system_prompt, user_prompt, uuid,
            last_modified, version, client_id, deleted, prompt_format,
            prompt_schema_version, prompt_definition, artifact_type
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, 0, 'legacy', NULL, NULL, ?)
        """,
        (
            name,
            author,
            details,
            system_prompt,
            user_prompt,
            str(uuid.uuid4()),
            last_modified,
            database.client_id,
            artifact_type,
        ),
    )
    database.get_connection().commit()
    return int(cursor.lastrowid)


def _mark_seeded_prompt_deleted(database, prompt_id):
    database.get_connection().execute(
        """
        UPDATE Prompts
        SET deleted = 1,
            last_modified = '2026-08-09T13:00:00.000Z',
            version = version + 1,
            client_id = ?
        WHERE id = ?
        """,
        (database.client_id, prompt_id),
    )
    database.get_connection().commit()


def test_browse_prompts_pages_exactly_beyond_one_hundred_mixed_artifacts(database):
    ids = []
    for index in range(105):
        ids.append(
            _insert_prompt(
                database,
                name=f"Prompt {index:03d}",
                details=f"Details {index}",
                system_prompt="system" if index == 104 else "",
                user_prompt="user" if index == 104 else "",
                last_modified=f"2026-08-09T12:{index // 60:02d}:{index % 60:02d}.000Z",
                artifact_type="recipe" if index % 2 else "prompt",
            )
        )

    first, total_pages, current_page, total_items = database.browse_prompts(
        page=1, page_size=100
    )
    second, second_total_pages, second_current_page, second_total = (
        database.browse_prompts(page=2, page_size=100)
    )

    assert (len(first), total_pages, current_page, total_items) == (100, 2, 1, 105)
    assert (len(second), second_total_pages, second_current_page, second_total) == (
        5,
        2,
        2,
        105,
    )
    assert {item["id"] for item in first + second} == set(ids)
    assert {item["artifact_type"] for item in first + second} == {
        "prompt",
        "recipe",
    }
    assert set(first[0]) == {
        "id",
        "name",
        "uuid",
        "author",
        "details",
        "last_modified",
        "version",
        "artifact_type",
        "has_system_prompt",
        "has_user_prompt",
    }
    assert first[0]["has_system_prompt"] == 1
    assert first[0]["has_user_prompt"] == 1


def test_browse_prompts_explicit_twenty_row_pages_apply_scope_before_paging(database):
    matching_ids = [
        _insert_prompt(
            database,
            name=f"Prompt {index:02d}",
            details=f"Needle match {index:02d}",
        )
        for index in range(45)
    ]
    nonmatching_id = _insert_prompt(database, name="Unrelated")
    _insert_prompt(database, name="Outside", details="Needle outside collection")
    service = LocalPromptService(database)
    collection_id = service.create_prompt_collection(
        {
            "name": "Paged",
            "prompt_ids": [*matching_ids, nonmatching_id],
        }
    )["collection_id"]

    pages = [
        database.browse_prompts(
            query="needle",
            collection_id=collection_id,
            sort_by="last_modified",
            sort_order="asc",
            page=page,
            page_size=20,
        )
        for page in (1, 2, 3)
    ]

    assert [len(items) for items, *_metadata in pages] == [20, 20, 5]
    assert [metadata for _items, *metadata in pages] == [
        [3, 1, 45],
        [3, 2, 45],
        [3, 3, 45],
    ]
    assert [item["id"] for items, *_metadata in pages for item in items] == matching_ids


def test_browse_prompts_omitted_page_size_keeps_generic_fifty_row_default(database):
    for index in range(51):
        _insert_prompt(database, name=f"Generic {index:02d}")

    items, total_pages, current_page, total_items = database.browse_prompts()

    assert (len(items), total_pages, current_page, total_items) == (50, 2, 1, 51)


def test_browse_prompts_combines_collection_and_literal_text_search(database):
    inside = _insert_prompt(database, name="Inside", details="A Needle in collection")
    outside = _insert_prompt(
        database, name="Outside Needle", details="outside collection"
    )
    deleted = _insert_prompt(
        database, name="Deleted Needle", details="still has membership"
    )
    percent = _insert_prompt(database, name="100% Literal")
    underscore = _insert_prompt(database, name="under_score Literal")
    backslash = _insert_prompt(database, name=r"path\segment Literal")
    _insert_prompt(
        database,
        name="Other fields do not match",
        author="needle author",
        system_prompt="needle system",
        user_prompt="needle user",
    )

    service = LocalPromptService(database)
    collection_id = service.create_prompt_collection(
        {
            "name": "Selected",
            "prompt_ids": [inside, deleted, percent, underscore, backslash],
        }
    )["collection_id"]
    _mark_seeded_prompt_deleted(database, deleted)

    items, total_pages, current_page, total_items = database.browse_prompts(
        query="  nEeDlE  ", collection_id=collection_id
    )

    assert [item["id"] for item in items] == [inside]
    assert (total_pages, current_page, total_items) == (1, 1, 1)
    assert outside not in {item["id"] for item in items}
    for literal, expected_id in (
        ("%", percent),
        ("_", underscore),
        ("\\", backslash),
    ):
        literal_items, _, _, literal_total = database.browse_prompts(query=literal)
        assert [item["id"] for item in literal_items] == [expected_id]
        assert literal_total == 1

    other_field_items, _, _, other_field_total = database.browse_prompts(
        query="needle author"
    )
    assert other_field_items == []
    assert other_field_total == 0


@pytest.mark.parametrize(
    ("query", "expected_name"),
    [
        ("éclair", "Éclair"),
        ("éClAiR", "Éclair"),
        ("détail", "Details holder"),
        ("déTaIl", "Details holder"),
    ],
)
def test_browse_prompts_uses_python_lower_for_unicode_substrings(
    database, query, expected_name
):
    _insert_prompt(database, name="Éclair")
    _insert_prompt(database, name="Details holder", details="A DÉTAIL précis")
    _insert_prompt(database, name="Straße")

    items, total_pages, current_page, total_items = database.browse_prompts(query=query)

    assert [item["name"] for item in items] == [expected_name]
    assert (total_pages, current_page, total_items) == (1, 1, 1)
    casefold_only, _, _, casefold_total = database.browse_prompts(query="STRASSE")
    assert casefold_only == []
    assert casefold_total == 0


@pytest.mark.parametrize("collection_id", [1, (2**63) - 1])
def test_browse_prompts_missing_collection_schema_is_empty_and_not_created(
    database, collection_id
):
    connection = database.get_connection()
    table_names = ("LocalPromptCollections", "LocalPromptCollectionItems")

    assert (
        connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name IN (?, ?)",
            table_names,
        ).fetchone()[0]
        == 0
    )

    result = database.browse_prompts(collection_id=collection_id, page=9)

    assert result == ([], 0, 1, 0)
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name IN (?, ?)",
            table_names,
        ).fetchone()[0]
        == 0
    )


def test_browse_prompts_rejects_collection_id_above_sqlite_integer(database):
    LocalPromptService(database).list_prompt_collections()

    with pytest.raises(ValueError, match="collection_id"):
        database.browse_prompts(collection_id=2**63)


def test_browse_prompts_treats_missing_and_inactive_collections_as_empty(database):
    prompt_id = _insert_prompt(database, name="Collected")
    service = LocalPromptService(database)
    collection_id = service.create_prompt_collection(
        {"name": "Temporary", "prompt_ids": [prompt_id]}
    )["collection_id"]

    missing = database.browse_prompts(collection_id=collection_id + 999, page=9)
    database.get_connection().execute(
        """
        UPDATE LocalPromptCollections
        SET deleted = 1, version = version + 1, updated_at = CURRENT_TIMESTAMP
        WHERE collection_id = ?
        """,
        (collection_id,),
    )
    database.get_connection().commit()
    inactive = database.browse_prompts(collection_id=collection_id, page=9)

    assert missing == ([], 0, 1, 0)
    assert inactive == ([], 0, 1, 0)


@pytest.mark.parametrize(
    ("sort_by", "sort_order", "expected_names"),
    [
        ("name", "asc", ["alpha", "Alpha", "Zulu"]),
        ("name", "desc", ["Zulu", "Alpha", "alpha"]),
        ("last_modified", "asc", ["alpha", "Alpha", "Zulu"]),
        ("last_modified", "desc", ["Zulu", "Alpha", "alpha"]),
    ],
)
def test_browse_prompts_sorts_with_same_direction_id_tie_breaker(
    database, sort_by, sort_order, expected_names
):
    _insert_prompt(
        database,
        name="alpha",
        last_modified="2026-08-08T12:00:00.000Z",
    )
    _insert_prompt(
        database,
        name="Alpha",
        last_modified="2026-08-08T12:00:00.000Z",
    )
    _insert_prompt(
        database,
        name="Zulu",
        last_modified="2026-08-09T12:00:00.000Z",
    )

    items, _, _, _ = database.browse_prompts(sort_by=sort_by, sort_order=sort_order)

    assert [item["name"] for item in items] == expected_names


@pytest.mark.parametrize(
    ("sort_order", "expected_name_pages"),
    [
        ("asc", [["Kelvin", "Zulu"], ["éclair", "Éclair"]]),
        ("desc", [["Éclair", "éclair"], ["Zulu", "Kelvin"]]),
    ],
)
def test_browse_prompts_name_sort_matches_python_lower_across_pages(
    database, sort_order, expected_name_pages
):
    ids = {
        name: _insert_prompt(database, name=name)
        for name in ("Zulu", "Kelvin", "éclair", "Éclair")
    }

    pages = []
    for page in (1, 2):
        items, total_pages, current_page, total_items = database.browse_prompts(
            sort_by="name", sort_order=sort_order, page=page, page_size=2
        )
        pages.append([(item["name"], item["id"]) for item in items])
        assert (total_pages, current_page, total_items) == (2, page, 4)

    assert pages == [
        [(name, ids[name]) for name in expected_names]
        for expected_names in expected_name_pages
    ]


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"query": None}, TypeError, "query"),
        ({"collection_id": True}, ValueError, "collection_id"),
        ({"collection_id": 0}, ValueError, "collection_id"),
        ({"sort_by": "name; DROP TABLE Prompts"}, ValueError, "sort_by"),
        ({"sort_by": None}, TypeError, "sort_by"),
        ({"sort_order": "desc; DELETE FROM Prompts"}, ValueError, "sort_order"),
        ({"sort_order": None}, TypeError, "sort_order"),
        ({"page": True}, ValueError, "page"),
        ({"page": 0}, ValueError, "page"),
        ({"page_size": True}, ValueError, "page_size"),
        ({"page_size": 0}, ValueError, "page_size"),
    ],
)
def test_browse_prompts_rejects_invalid_trust_boundary_values(
    database, kwargs, error, message
):
    _insert_prompt(database, name="Still Here")

    with pytest.raises(error, match=message):
        database.browse_prompts(**kwargs)

    assert (
        database.get_connection().execute("SELECT COUNT(*) FROM Prompts").fetchone()[0]
        == 1
    )


def test_browse_prompts_caps_page_size_and_clamps_requested_page(database):
    for index in range(105):
        _insert_prompt(database, name=f"Bounded {index:03d}")

    capped, total_pages, current_page, total_items = database.browse_prompts(
        page=1, page_size=999
    )
    last, last_total_pages, last_current_page, last_total_items = (
        database.browse_prompts(page=999, page_size=100)
    )

    assert (len(capped), total_pages, current_page, total_items) == (100, 2, 1, 105)
    assert (len(last), last_total_pages, last_current_page, last_total_items) == (
        5,
        2,
        2,
        105,
    )


def test_browse_prompts_empty_result_uses_page_one(database):
    items, total_pages, current_page, total_items = database.browse_prompts(page=99)

    assert items == []
    assert (total_pages, current_page, total_items) == (0, 1, 0)


def test_browse_prompts_count_and_page_share_snapshot_during_deletion(database):
    connection = database.get_connection()
    assert connection.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    target_id = _insert_prompt(database, name="Deleted between count and page")
    _insert_prompt(database, name="Second")
    _insert_prompt(database, name="Third")

    select_count = 0
    deletion_happened = False

    def delete_before_page(statement):
        nonlocal select_count, deletion_happened
        normalized = " ".join(statement.upper().split())
        if not normalized.startswith("SELECT") or "FROM PROMPTS" not in normalized:
            return
        select_count += 1
        if select_count != 2:
            return
        writer = sqlite3.connect(database.db_path_str)
        try:
            writer.execute(
                """
                UPDATE Prompts
                SET deleted = 1,
                    last_modified = '2026-08-09T13:00:00.000Z',
                    version = version + 1,
                    client_id = 'concurrent-deleter'
                WHERE id = ?
                """,
                (target_id,),
            )
            writer.commit()
            deletion_happened = True
        finally:
            writer.close()

    connection.set_trace_callback(delete_before_page)
    try:
        items, total_pages, current_page, total_items = database.browse_prompts(
            page_size=3
        )
    finally:
        connection.set_trace_callback(None)

    assert deletion_happened is True
    assert target_id in {item["id"] for item in items}
    assert (len(items), total_pages, current_page, total_items) == (3, 1, 1, 3)
    active_now = connection.execute(
        "SELECT COUNT(*) FROM Prompts WHERE deleted = 0"
    ).fetchone()[0]
    assert active_now == 2
