from __future__ import annotations

import pytest

from tldw_chatbook.Agents.session_todo_store import (
    MAX_TODO_CONTENT_CHARS,
    MAX_TODO_ITEMS,
    TODO_STATUSES,
    SessionTodoStore,
    TodoStoreError,
)


class _StrSubclass(str):
    pass


class _IntSubclass(int):
    pass


class _DictSubclass(dict[str, object]):
    pass


class _ListSubclass(list[object]):
    pass


def _valid_snapshot() -> dict[str, object]:
    return {
        "next_id": 4,
        "tasks": [
            {
                "id": "1",
                "version": 2,
                "content": "A",
                "status": "completed",
            },
            {
                "id": "3",
                "version": 1,
                "content": "B",
                "status": "pending",
                "activeForm": "Working on B",
            },
        ],
    }


def test_public_constants_match_the_session_task_contract() -> None:
    assert MAX_TODO_ITEMS == 50
    assert MAX_TODO_CONTENT_CHARS == 500
    assert TODO_STATUSES == ("pending", "in_progress", "completed")


def test_create_assigns_stable_ids_and_default_record_fields() -> None:
    store = SessionTodoStore()

    first = store.create(content="First")
    second = store.create(content="Second", active_form="Working on second")

    assert first == {
        "id": "1",
        "version": 1,
        "content": "First",
        "status": "pending",
    }
    assert second == {
        "id": "2",
        "version": 1,
        "content": "Second",
        "status": "pending",
        "activeForm": "Working on second",
    }


def test_create_and_get_return_defensive_record_copies() -> None:
    store = SessionTodoStore()

    created = store.create(content="Original")
    created["content"] = "mutated create result"
    fetched = store.get("1")
    fetched["content"] = "mutated get result"

    assert store.get("1") == {
        "id": "1",
        "version": 1,
        "content": "Original",
        "status": "pending",
    }


@pytest.mark.parametrize(
    "content",
    [None, b"bytes", 1, True, _StrSubclass("subclass")],
)
def test_create_rejects_content_that_is_not_an_exact_builtin_string(
    content: object,
) -> None:
    store = SessionTodoStore()

    with pytest.raises(TodoStoreError):
        store.create(content=content)

    assert store.export_snapshot() == {"next_id": 1, "tasks": []}


@pytest.mark.parametrize("content", ["", " ", "\t\n"])
def test_create_rejects_blank_content(content: str) -> None:
    store = SessionTodoStore()

    with pytest.raises(TodoStoreError):
        store.create(content=content)

    assert store.export_snapshot() == {"next_id": 1, "tasks": []}


def test_create_accepts_content_at_character_limit_and_rejects_one_over() -> None:
    store = SessionTodoStore()

    record = store.create(content="é" * MAX_TODO_CONTENT_CHARS)

    assert record["content"] == "é" * MAX_TODO_CONTENT_CHARS
    with pytest.raises(TodoStoreError):
        store.create(content="x" * (MAX_TODO_CONTENT_CHARS + 1))
    assert store.export_snapshot()["next_id"] == 2


@pytest.mark.parametrize(
    "active_form",
    [None, b"bytes", 1, True, _StrSubclass("subclass")],
)
def test_create_rejects_active_form_that_is_not_an_exact_builtin_string(
    active_form: object,
) -> None:
    store = SessionTodoStore()

    with pytest.raises(TodoStoreError):
        store.create(content="Valid", active_form=active_form)

    assert store.export_snapshot() == {"next_id": 1, "tasks": []}


def test_create_accepts_empty_and_bounded_active_form() -> None:
    store = SessionTodoStore()

    empty = store.create(content="First", active_form="")
    bounded = store.create(content="Second", active_form="é" * MAX_TODO_CONTENT_CHARS)

    assert empty["activeForm"] == ""
    assert bounded["activeForm"] == "é" * MAX_TODO_CONTENT_CHARS


def test_create_rejects_active_form_over_character_limit_atomically() -> None:
    store = SessionTodoStore()

    with pytest.raises(TodoStoreError):
        store.create(content="Valid", active_form="x" * (MAX_TODO_CONTENT_CHARS + 1))

    assert store.export_snapshot() == {"next_id": 1, "tasks": []}


@pytest.mark.parametrize(
    ("content", "active_form"),
    [("bad\ud800", "Valid"), ("Valid", "bad\ud800")],
)
def test_create_rejects_non_utf8_text_before_allocating_an_id(
    content: str, active_form: str
) -> None:
    store = SessionTodoStore()

    with pytest.raises(TodoStoreError):
        store.create(content=content, active_form=active_form)

    assert store.export_snapshot() == {"next_id": 1, "tasks": []}


def test_utf8_validation_does_not_chain_the_raw_codec_error() -> None:
    store = SessionTodoStore()

    with pytest.raises(TodoStoreError) as error:
        store.create(content="bad\ud800")

    assert error.value.__cause__ is None


def test_create_rejects_the_fifty_first_live_task_without_allocating_an_id() -> None:
    store = SessionTodoStore()
    for index in range(MAX_TODO_ITEMS):
        store.create(content=f"Task {index}")

    with pytest.raises(TodoStoreError):
        store.create(content="One too many")

    snapshot = store.export_snapshot()
    assert len(snapshot["tasks"]) == MAX_TODO_ITEMS
    assert snapshot["next_id"] == MAX_TODO_ITEMS + 1


@pytest.mark.parametrize(
    "task_id",
    [
        None,
        1,
        True,
        _StrSubclass("1"),
        "",
        "0",
        "01",
        "+1",
        "-1",
        "1.0",
        "١",
        "1 ",
    ],
)
def test_get_rejects_noncanonical_task_ids(task_id: object) -> None:
    store = SessionTodoStore()
    store.create(content="Task")

    with pytest.raises(TodoStoreError):
        store.get(task_id)


def test_get_uses_a_fixed_not_found_error() -> None:
    store = SessionTodoStore()

    with pytest.raises(TodoStoreError) as first_error:
        store.get("1")
    store.create(content="Task")
    with pytest.raises(TodoStoreError) as second_error:
        store.get("999999999999999999999999")

    assert str(first_error.value) == "task not found"
    assert str(second_error.value) == "task not found"


def test_list_after_filters_by_numeric_id_in_creation_order() -> None:
    store = SessionTodoStore()
    for label in ("First", "Second", "Third"):
        store.create(content=label)

    assert [record["id"] for record in store.list_after(None)] == ["1", "2", "3"]
    assert [record["id"] for record in store.list_after(1)] == ["2", "3"]
    assert store.list_after(3) == []
    assert store.list_after(999) == []


def test_list_after_returns_defensive_list_and_record_copies() -> None:
    store = SessionTodoStore()
    store.create(content="First")
    store.create(content="Second")

    records = store.list_after(None)
    records[0]["content"] = "mutated"
    records.clear()

    assert [record["content"] for record in store.list_after(None)] == [
        "First",
        "Second",
    ]


@pytest.mark.parametrize(
    "cursor",
    [-1, True, 1.0, "1", _IntSubclass(1)],
)
def test_list_after_rejects_invalid_numeric_lower_bounds(cursor: object) -> None:
    store = SessionTodoStore()

    with pytest.raises(TodoStoreError):
        store.list_after(cursor)  # type: ignore[arg-type]


def test_snapshot_round_trip_preserves_records_order_and_next_id() -> None:
    payload = _valid_snapshot()

    restored = SessionTodoStore.from_snapshot(payload)

    assert restored.export_snapshot() == payload
    assert restored.create(content="C")["id"] == "4"


def test_snapshot_restore_and_export_are_defensive() -> None:
    payload = _valid_snapshot()
    restored = SessionTodoStore.from_snapshot(payload)

    payload["next_id"] = 999
    input_tasks = payload["tasks"]
    assert type(input_tasks) is list
    input_tasks[0]["content"] = "mutated input"
    input_tasks.clear()
    exported = restored.export_snapshot()
    exported["next_id"] = 999
    output_tasks = exported["tasks"]
    assert type(output_tasks) is list
    output_tasks[0]["content"] = "mutated output"
    output_tasks.clear()

    assert restored.export_snapshot() == _valid_snapshot()


def test_snapshot_preserves_deleted_id_high_water_mark() -> None:
    restored = SessionTodoStore.from_snapshot(
        {
            "next_id": 10,
            "tasks": [
                {
                    "id": "1",
                    "version": 1,
                    "content": "Only live task",
                    "status": "pending",
                }
            ],
        }
    )

    assert restored.create(content="After navigation")["id"] == "10"


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        _DictSubclass({"next_id": 1, "tasks": []}),
        {},
        {"next_id": 1},
        {"tasks": []},
        {"next_id": 1, "tasks": [], "extra": True},
    ],
)
def test_snapshot_rejects_invalid_top_level_shape(payload: object) -> None:
    with pytest.raises(TodoStoreError):
        SessionTodoStore.from_snapshot(payload)


@pytest.mark.parametrize(
    "next_id",
    [None, True, 0, -1, 1.0, "1", _IntSubclass(1)],
)
def test_snapshot_rejects_invalid_next_id_types_and_bounds(next_id: object) -> None:
    with pytest.raises(TodoStoreError):
        SessionTodoStore.from_snapshot({"next_id": next_id, "tasks": []})


@pytest.mark.parametrize(
    "tasks",
    [None, (), {}, _ListSubclass()],
)
def test_snapshot_rejects_tasks_that_are_not_an_exact_builtin_list(
    tasks: object,
) -> None:
    with pytest.raises(TodoStoreError):
        SessionTodoStore.from_snapshot({"next_id": 1, "tasks": tasks})


def test_snapshot_rejects_more_than_fifty_live_tasks() -> None:
    tasks = [
        {
            "id": str(index),
            "version": 1,
            "content": f"Task {index}",
            "status": "pending",
        }
        for index in range(1, MAX_TODO_ITEMS + 2)
    ]

    with pytest.raises(TodoStoreError):
        SessionTodoStore.from_snapshot({"next_id": MAX_TODO_ITEMS + 2, "tasks": tasks})


@pytest.mark.parametrize(
    "record",
    [
        None,
        [],
        _DictSubclass({"id": "1", "version": 1, "content": "A", "status": "pending"}),
        {"version": 1, "content": "A", "status": "pending"},
        {"id": "1", "content": "A", "status": "pending"},
        {"id": "1", "version": 1, "status": "pending"},
        {"id": "1", "version": 1, "content": "A"},
        {
            "id": "1",
            "version": 1,
            "content": "A",
            "status": "pending",
            "unknown": True,
        },
    ],
)
def test_snapshot_rejects_invalid_record_shapes(record: object) -> None:
    with pytest.raises(TodoStoreError):
        SessionTodoStore.from_snapshot({"next_id": 2, "tasks": [record]})


@pytest.mark.parametrize(
    "task_id",
    [
        None,
        1,
        True,
        _StrSubclass("1"),
        "",
        "0",
        "01",
        "+1",
        "-1",
        "1.0",
        "١",
        "1 ",
    ],
)
def test_snapshot_rejects_noncanonical_task_ids(task_id: object) -> None:
    with pytest.raises(TodoStoreError):
        SessionTodoStore.from_snapshot(
            {
                "next_id": 2,
                "tasks": [
                    {
                        "id": task_id,
                        "version": 1,
                        "content": "A",
                        "status": "pending",
                    }
                ],
            }
        )


def test_snapshot_rejects_duplicate_task_ids() -> None:
    record = {"id": "1", "version": 1, "content": "A", "status": "pending"}

    with pytest.raises(TodoStoreError):
        SessionTodoStore.from_snapshot({"next_id": 2, "tasks": [record, dict(record)]})


@pytest.mark.parametrize(
    "version",
    [None, True, 0, -1, 1.0, "1", _IntSubclass(1)],
)
def test_snapshot_rejects_invalid_versions(version: object) -> None:
    with pytest.raises(TodoStoreError):
        SessionTodoStore.from_snapshot(
            {
                "next_id": 2,
                "tasks": [
                    {
                        "id": "1",
                        "version": version,
                        "content": "A",
                        "status": "pending",
                    }
                ],
            }
        )


@pytest.mark.parametrize(
    "content",
    [
        None,
        b"bytes",
        True,
        _StrSubclass("A"),
        "",
        " ",
        "x" * (MAX_TODO_CONTENT_CHARS + 1),
        "bad\ud800",
    ],
)
def test_snapshot_rejects_invalid_content(content: object) -> None:
    with pytest.raises(TodoStoreError):
        SessionTodoStore.from_snapshot(
            {
                "next_id": 2,
                "tasks": [
                    {
                        "id": "1",
                        "version": 1,
                        "content": content,
                        "status": "pending",
                    }
                ],
            }
        )


@pytest.mark.parametrize(
    "active_form",
    [
        None,
        b"bytes",
        True,
        _StrSubclass("A"),
        "x" * (MAX_TODO_CONTENT_CHARS + 1),
        "bad\ud800",
    ],
)
def test_snapshot_rejects_invalid_active_form(active_form: object) -> None:
    with pytest.raises(TodoStoreError):
        SessionTodoStore.from_snapshot(
            {
                "next_id": 2,
                "tasks": [
                    {
                        "id": "1",
                        "version": 1,
                        "content": "A",
                        "status": "pending",
                        "activeForm": active_form,
                    }
                ],
            }
        )


@pytest.mark.parametrize(
    "status",
    [None, True, "deleted", "Pending", _StrSubclass("pending")],
)
def test_snapshot_rejects_invalid_status(status: object) -> None:
    with pytest.raises(TodoStoreError):
        SessionTodoStore.from_snapshot(
            {
                "next_id": 2,
                "tasks": [
                    {
                        "id": "1",
                        "version": 1,
                        "content": "A",
                        "status": status,
                    }
                ],
            }
        )


def test_snapshot_rejects_more_than_one_in_progress_task() -> None:
    with pytest.raises(TodoStoreError):
        SessionTodoStore.from_snapshot(
            {
                "next_id": 3,
                "tasks": [
                    {
                        "id": "1",
                        "version": 1,
                        "content": "A",
                        "status": "in_progress",
                    },
                    {
                        "id": "2",
                        "version": 1,
                        "content": "B",
                        "status": "in_progress",
                    },
                ],
            }
        )


@pytest.mark.parametrize("next_id", [1, 3])
def test_snapshot_requires_next_id_above_every_live_id(next_id: int) -> None:
    with pytest.raises(TodoStoreError):
        SessionTodoStore.from_snapshot(
            {
                "next_id": next_id,
                "tasks": [
                    {
                        "id": "3",
                        "version": 1,
                        "content": "A",
                        "status": "pending",
                    }
                ],
            }
        )


def test_empty_snapshot_accepts_any_positive_next_id() -> None:
    restored = SessionTodoStore.from_snapshot({"next_id": 42, "tasks": []})

    assert restored.create(content="First after restore")["id"] == "42"


def test_snapshot_errors_do_not_reflect_payload_values() -> None:
    secret = "DO-NOT-ECHO-THIS" + "x" * MAX_TODO_CONTENT_CHARS

    with pytest.raises(TodoStoreError) as error:
        SessionTodoStore.from_snapshot(
            {
                "next_id": 2,
                "tasks": [
                    {
                        "id": "1",
                        "version": 1,
                        "content": secret,
                        "status": "pending",
                    }
                ],
            }
        )

    assert secret not in str(error.value)
