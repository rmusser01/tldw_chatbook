from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
from tldw_chatbook.Prompt_Management.prompt_scope_service import LocalPromptService


SQLITE_SIGNED_INTEGER_MAX = PromptsDatabase._SQLITE_SIGNED_INTEGER_MAX


@pytest.fixture
def catalog(tmp_path):
    database = PromptsDatabase(tmp_path / "prompt-collections.db", client_id="catalog")
    service = LocalPromptService(database)
    service.list_prompt_collections(limit=1)
    try:
        yield database, service
    finally:
        database.close_connection()


def _seed_collections(database, names):
    with database.transaction(immediate=True) as conn:
        conn.executemany(
            "INSERT INTO LocalPromptCollections (name, description) VALUES (?, ?)",
            [(name, f"Description {index}") for index, name in enumerate(names)],
        )


def test_catalog_pages_all_rows_with_exact_total_cap_and_python_casefold_order(catalog):
    database, service = catalog
    names = [f"Collection {index:03d}" for index in range(207)]
    names[3:7] = ["zebra", "Ångström", "Straße", "alpha"]
    _seed_collections(database, names)

    first = service.list_prompt_collections(limit=500, offset=0)
    second = service.list_prompt_collections(limit=100, offset=100)
    third = service.list_prompt_collections(limit=100, offset=200)
    records = first["collections"] + second["collections"] + third["collections"]

    assert first["limit"] == 100
    assert first["offset"] == 0
    assert first["total"] == second["total"] == third["total"] == 207
    assert len(records) == 207
    assert len({record["collection_id"] for record in records}) == 207
    assert [record["name"] for record in records] == sorted(
        names, key=lambda name: (name.casefold(), names.index(name) + 1)
    )


def test_catalog_search_is_trimmed_literal_unicode_casefolded_and_exact(catalog):
    database, service = catalog
    _seed_collections(
        database,
        ["Straße", "100% Match", "100x Match", "_draft", "xdraft", "Other"],
    )

    unicode_result = service.list_prompt_collections(query="  STRASSE  ", limit=10)
    percent_result = service.list_prompt_collections(query="%", limit=10)
    underscore_result = service.list_prompt_collections(query="_", limit=10)
    offset_result = service.list_prompt_collections(query="draft", limit=1, offset=1)
    maximum_offset = service.list_prompt_collections(
        query="draft", limit=1, offset=SQLITE_SIGNED_INTEGER_MAX
    )

    assert unicode_result["total"] == 1
    assert [item["name"] for item in unicode_result["collections"]] == ["Straße"]
    assert percent_result["total"] == 1
    assert [item["name"] for item in percent_result["collections"]] == ["100% Match"]
    assert underscore_result["total"] == 1
    assert [item["name"] for item in underscore_result["collections"]] == ["_draft"]
    assert offset_result["total"] == 2
    assert offset_result["offset"] == 1
    assert len(offset_result["collections"]) == 1
    assert maximum_offset["offset"] == SQLITE_SIGNED_INTEGER_MAX
    assert maximum_offset["total"] == 2
    assert maximum_offset["collections"] == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"query": None}, "query"),
        ({"limit": True}, "limit"),
        ({"limit": 0}, "limit"),
        ({"offset": False}, "offset"),
        ({"offset": -1}, "offset"),
        ({"offset": SQLITE_SIGNED_INTEGER_MAX + 1}, "offset"),
    ],
)
def test_catalog_rejects_invalid_bounds(catalog, kwargs, message):
    _database, service = catalog

    with pytest.raises((TypeError, ValueError), match=message):
        service.list_prompt_collections(**kwargs)


def test_catalog_labels_collisions_from_full_active_catalog_and_mutates_by_id(catalog):
    database, service = catalog
    _seed_collections(database, ["Sales", "sales", "Unrelated"])
    rows = (
        database.get_connection()
        .execute(
            "SELECT collection_id, name FROM LocalPromptCollections ORDER BY collection_id"
        )
        .fetchall()
    )
    sales_id, lower_sales_id = int(rows[0][0]), int(rows[1][0])

    first_only = service.list_prompt_collections(query="sales", limit=1, offset=0)
    second_only = service.list_prompt_collections(query="sales", limit=1, offset=1)
    original = service.update_prompt_collection(sales_id, {"name": "SALES"})
    renamed = service.update_prompt_collection(lower_sales_id, {"name": "Revenue"})

    assert first_only["total"] == second_only["total"] == 2
    assert first_only["collections"][0]["display_name"] == f"Sales · #{sales_id}"
    assert second_only["collections"][0]["display_name"] == f"sales · #{lower_sales_id}"
    assert renamed["collection_id"] == lower_sales_id
    assert renamed["name"] == renamed["display_name"] == "Revenue"
    assert original["collection_id"] == sales_id
    assert original["name"] == "SALES"
    assert original["display_name"] == f"SALES · #{sales_id}"


def test_missing_collection_id_wins_over_requested_name_collision(catalog):
    database, service = catalog
    _seed_collections(database, ["Sales"])

    with pytest.raises(ValueError, match="not found"):
        service.update_prompt_collection(999_999, {"name": "sales"})

    listed = service.list_prompt_collections(limit=10)
    assert [(item["name"], item["description"]) for item in listed["collections"]] == [
        ("Sales", "Description 0")
    ]


@pytest.mark.parametrize("requested_name", ["Tombstone", "TOMBSTONE"])
def test_deleted_collection_name_is_reserved_for_create(catalog, requested_name):
    database, service = catalog
    _seed_collections(database, ["Tombstone"])
    database.get_connection().execute(
        "UPDATE LocalPromptCollections SET deleted = 1 WHERE name = 'Tombstone'"
    )
    database.get_connection().commit()

    with pytest.raises(ValueError, match="reserved"):
        service.create_prompt_collection({"name": requested_name})

    assert service.list_prompt_collections(limit=10)["total"] == 0


@pytest.mark.parametrize("requested_name", ["Tombstone", "TOMBSTONE"])
def test_deleted_collection_name_is_reserved_for_rename(catalog, requested_name):
    database, service = catalog
    _seed_collections(database, ["Tombstone", "Active"])
    rows = (
        database.get_connection()
        .execute(
            "SELECT collection_id, name FROM LocalPromptCollections ORDER BY collection_id"
        )
        .fetchall()
    )
    tombstone_id, active_id = int(rows[0][0]), int(rows[1][0])
    database.get_connection().execute(
        "UPDATE LocalPromptCollections SET deleted = 1 WHERE collection_id = ?",
        (tombstone_id,),
    )
    database.get_connection().commit()

    with pytest.raises(ValueError, match="reserved"):
        service.update_prompt_collection(active_id, {"name": requested_name})

    assert service.get_prompt_collection(active_id)["name"] == "Active"


@pytest.mark.parametrize(
    "invalid_collection_id",
    [True, 1.0, 1.5, "1.0", SQLITE_SIGNED_INTEGER_MAX + 1, str(2**63)],
)
def test_collection_id_rejects_non_integral_or_overflow_values_without_mutation(
    catalog, invalid_collection_id
):
    _database, service = catalog
    collection_id = service.create_prompt_collection(
        {"name": "Original", "description": "Before"}
    )["collection_id"]

    with pytest.raises(ValueError, match="collection id"):
        service.update_prompt_collection(
            invalid_collection_id, {"name": "Renamed", "description": "After"}
        )

    record = service.get_prompt_collection(collection_id)
    assert record["name"] == "Original"
    assert record["description"] == "Before"


def test_collection_id_accepts_positive_numeric_string_and_signed_max(catalog):
    database, service = catalog
    with database.transaction(immediate=True) as conn:
        conn.execute(
            """
            INSERT INTO LocalPromptCollections (collection_id, name, description)
            VALUES (?, 'Maximum', 'Before')
            """,
            (SQLITE_SIGNED_INTEGER_MAX,),
        )

    fetched = service.get_prompt_collection(str(SQLITE_SIGNED_INTEGER_MAX))
    updated = service.update_prompt_collection(
        str(SQLITE_SIGNED_INTEGER_MAX), {"description": "After"}
    )

    assert fetched["collection_id"] == SQLITE_SIGNED_INTEGER_MAX
    assert updated["collection_id"] == SQLITE_SIGNED_INTEGER_MAX
    assert updated["description"] == "After"


@pytest.mark.parametrize(
    "invalid_prompt_id",
    [True, 1.0, 1.5, "1", 0, -1, SQLITE_SIGNED_INTEGER_MAX + 1],
)
def test_prompt_ids_reject_non_integer_or_overflow_values_without_mutation(
    catalog, invalid_prompt_id
):
    database, service = catalog
    prompt_id = database.add_prompt(
        name="Valid Prompt",
        author="Writer",
        details="Details",
        user_prompt="Body",
        overwrite=False,
    )[0]
    collection_id = service.create_prompt_collection(
        {
            "name": "Original",
            "description": "Before",
            "prompt_ids": [prompt_id],
        }
    )["collection_id"]

    with pytest.raises(ValueError, match="prompt_ids"):
        service.update_prompt_collection(
            collection_id,
            {
                "name": "Renamed",
                "description": "After",
                "prompt_ids": [invalid_prompt_id],
            },
        )

    record = service.get_prompt_collection(collection_id)
    assert record["name"] == "Original"
    assert record["description"] == "Before"
    assert record["prompt_ids"] == [prompt_id]


def test_catalog_batches_prompt_memberships(catalog):
    database, service = catalog
    prompt_ids = [
        database.add_prompt(
            name=f"Prompt {index}",
            author="Writer",
            details="Details",
            user_prompt="Body",
            overwrite=False,
        )[0]
        for index in range(2)
    ]
    first_id = service.create_prompt_collection(
        {"name": "First", "prompt_ids": prompt_ids}
    )["collection_id"]
    second_id = service.create_prompt_collection(
        {"name": "Second", "prompt_ids": [prompt_ids[1]]}
    )["collection_id"]
    statements = []
    database.get_connection().set_trace_callback(statements.append)
    try:
        result = service.list_prompt_collections(limit=10)
    finally:
        database.get_connection().set_trace_callback(None)

    by_id = {
        item["collection_id"]: item["prompt_ids"] for item in result["collections"]
    }
    membership_selects = [
        statement
        for statement in statements
        if statement.lstrip().upper().startswith("SELECT")
        and "FROM LOCALPROMPTCOLLECTIONITEMS" in statement.upper()
    ]
    assert by_id == {first_id: prompt_ids, second_id: [prompt_ids[1]]}
    assert len(membership_selects) == 1


def test_collection_update_rolls_back_all_fields_on_invalid_prompt_reference(catalog):
    database, service = catalog
    prompt_id = database.add_prompt(
        name="Valid Prompt",
        author="Writer",
        details="Details",
        user_prompt="Body",
        overwrite=False,
    )[0]
    collection_id = service.create_prompt_collection(
        {
            "name": "Original",
            "description": "Before",
            "prompt_ids": [prompt_id],
        }
    )["collection_id"]

    with pytest.raises(ValueError, match="prompt reference"):
        service.update_prompt_collection(
            collection_id,
            {
                "name": "Renamed",
                "description": "After",
                "prompt_ids": [prompt_id, 999_999],
            },
        )

    record = service.get_prompt_collection(collection_id)
    assert record["name"] == "Original"
    assert record["description"] == "Before"
    assert record["prompt_ids"] == [prompt_id]


def test_concurrent_casefold_create_allows_exactly_one_writer(catalog):
    database, _service = catalog
    barrier = Barrier(2)

    def create(name):
        barrier.wait()
        try:
            return LocalPromptService(database).create_prompt_collection({"name": name})
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(create, ["Sales", "sales"]))

    listed = LocalPromptService(database).list_prompt_collections(limit=10)
    assert sum(outcome is not None for outcome in outcomes) == 1
    assert listed["total"] == 1
    assert listed["collections"][0]["name"].casefold() == "sales"


def test_concurrent_casefold_rename_allows_exactly_one_writer(catalog):
    database, service = catalog
    collection_ids = [
        service.create_prompt_collection({"name": name})["collection_id"]
        for name in ("First", "Second")
    ]
    barrier = Barrier(2)

    def rename(args):
        collection_id, name = args
        barrier.wait()
        try:
            return LocalPromptService(database).update_prompt_collection(
                collection_id, {"name": name}
            )
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(
            executor.map(rename, zip(collection_ids, ["Sales", "sales"], strict=True))
        )

    listed = LocalPromptService(database).list_prompt_collections(limit=10)
    assert sum(outcome is not None for outcome in outcomes) == 1
    assert listed["total"] == 2
    assert (
        sum(item["name"].casefold() == "sales" for item in listed["collections"]) == 1
    )
