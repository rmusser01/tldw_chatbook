from __future__ import annotations

import logging
import multiprocessing
from queue import Queue
from threading import Event, Thread, current_thread
from typing import Any, Callable

import pytest

from tldw_chatbook.Agents import session_todo_store as todo_store_module
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


class _ObservedLock:
    """Observe one named thread immediately before real lock acquisition."""

    def __init__(
        self,
        lock: Any,
        *,
        thread_name: str,
        before_acquire: Callable[[], None],
    ) -> None:
        self._lock = lock
        self._thread_name = thread_name
        self._before_acquire = before_acquire

    def __enter__(self) -> object:
        if current_thread().name == self._thread_name:
            self._before_acquire()
        return self._lock.__enter__()

    def __exit__(self, *args: object) -> object:
        return self._lock.__exit__(*args)

    def locked(self) -> bool:
        return bool(self._lock.locked())


class _ParkingTaskMap(dict[str, dict[str, object]]):
    """Park one named contender during a selected mapping operation."""

    def __init__(
        self,
        initial: dict[str, dict[str, object]],
        *,
        operation: str,
        thread_name: str,
    ) -> None:
        super().__init__(initial)
        self.operation = operation
        self.thread_name = thread_name
        self.entered = Event()
        self.release = Event()
        self._parked = False

    def _park(self, operation: str) -> None:
        if (
            operation == self.operation
            and current_thread().name == self.thread_name
            and not self._parked
        ):
            self._parked = True
            self.entered.set()
            self.release.wait(5)

    def __len__(self) -> int:
        self._park("len")
        return super().__len__()

    def get(
        self,
        key: str,
        default: dict[str, object] | None = None,
    ) -> dict[str, object] | None:
        self._park("get")
        return super().get(key, default)

    def __setitem__(self, key: str, value: dict[str, object]) -> None:
        self._park("setitem")
        super().__setitem__(key, value)


def _run_forced_mutation_pair(
    store: SessionTodoStore,
    first_action: Callable[[], object],
    second_action: Callable[[], object],
    *,
    operation: str,
) -> tuple[list[object | None], list[BaseException | None]]:
    """Park the first mutation inside state work as the second attempts entry."""
    first_name = "first-contender"
    second_name = "second-contender"
    second_attempted_mutation_lock = Event()
    second_done = Event()
    tasks = _ParkingTaskMap(
        dict(store._tasks),  # type: ignore[attr-defined]
        operation=operation,
        thread_name=first_name,
    )
    store._tasks = tasks  # type: ignore[assignment]
    store._mutation_lock = _ObservedLock(  # type: ignore[attr-defined]
        store._mutation_lock,  # type: ignore[attr-defined]
        thread_name=second_name,
        before_acquire=second_attempted_mutation_lock.set,
    )
    results: list[object | None] = [None, None]
    errors: list[BaseException | None] = [None, None]

    def run(index: int, action: Callable[[], object]) -> None:
        try:
            results[index] = action()
        except BaseException as exc:
            errors[index] = exc
        finally:
            if index == 1:
                second_done.set()

    first = Thread(target=run, args=(0, first_action), name=first_name)
    second = Thread(target=run, args=(1, second_action), name=second_name)
    first.start()
    try:
        assert tasks.entered.wait(2)
        assert store._mutation_lock.locked()  # type: ignore[attr-defined]
        assert store._state_lock.locked()  # type: ignore[attr-defined]
        second.start()
        assert second_attempted_mutation_lock.wait(2)
        assert not second_done.is_set()
    finally:
        tasks.release.set()
        first.join(5)
        if second.ident is not None:
            second.join(5)

    assert not first.is_alive()
    assert not second.is_alive()
    return results, errors


class _SpawnProcessTimeout(AssertionError):
    """Report a child that required forced cleanup."""

    def __init__(self, pid: int | None) -> None:
        super().__init__("spawned task-store regression process timed out")
        self.pid = pid


def _run_spawned_target(
    target: Callable[..., None],
    *args: object,
    timeout: float = 5,
) -> object:
    """Run a spawn-safe target and always reclaim its process and queue."""
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(target=target, args=(result_queue, *args))
    started = False
    try:
        process.start()
        started = True
        process.join(timeout)
        if process.is_alive():
            raise _SpawnProcessTimeout(process.pid)
        assert process.exitcode == 0
        return result_queue.get(timeout=2)
    finally:
        try:
            if started and process.is_alive():
                process.terminate()
                process.join(2)
            if started and process.is_alive():
                process.kill()
                process.join(2)
            close_process = getattr(process, "close", None)
            if callable(close_process):
                close_process()
        finally:
            try:
                result_queue.close()
            finally:
                result_queue.join_thread()


def _callback_read_process(result_queue: Any) -> None:
    """Exercise callback reads in a spawn-safe subprocess."""
    store = SessionTodoStore()

    def read_from_callback(snapshot: list[dict[str, object]]) -> None:
        threaded_records: Queue[list[dict[str, object]]] = Queue()
        reader = Thread(
            target=lambda: threaded_records.put(store.list_after(None)),
            name="callback-reader",
        )
        reader.start()
        direct_record = store.get("1")
        reader.join(2)
        if reader.is_alive():
            result_queue.put(("reader-deadlocked", None))
            return
        result_queue.put(
            (
                direct_record,
                threaded_records.get_nowait(),
                snapshot,
            )
        )

    store.create(content="Readable", on_change=read_from_callback)


def _callback_mutation_process(result_queue: Any, mutation: str) -> None:
    """Attempt a forbidden same-thread mutation from a callback."""
    store = SessionTodoStore()
    inner_error: tuple[str, str] | None = None

    def mutate_from_callback(snapshot: list[dict[str, object]]) -> None:
        nonlocal inner_error
        try:
            if mutation == "create":
                store.create(content="PRIVATE-INNER-CONTENT")
            elif mutation == "update":
                store.update(
                    task_id="1",
                    expected_version=1,
                    content="PRIVATE-INNER-CONTENT",
                )
            else:
                store.update(
                    task_id="1",
                    expected_version=1,
                    status="deleted",
                )
        except BaseException as exc:
            inner_error = (type(exc).__name__, str(exc))

    outer = store.create(content="Outer", on_change=mutate_from_callback)
    result_queue.put((outer, inner_error, store.export_snapshot()))


def _callback_base_exception_process(result_queue: Any) -> None:
    """Raise a non-Exception callback failure after a successful commit."""
    store = SessionTodoStore()

    def fail_callback(snapshot: list[dict[str, object]]) -> None:
        raise KeyboardInterrupt("PRIVATE-BASE-EXCEPTION")

    outer = store.create(content="Outer", on_change=fail_callback)
    result_queue.put((outer, store.export_snapshot()))


def _stuck_process(result_queue: Any) -> None:
    """Stay alive until the spawn-cleanup regression terminates this process."""
    Event().wait()


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
        store.get("999999")

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


def test_snapshot_rejects_reversed_task_id_order() -> None:
    with pytest.raises(TodoStoreError) as error:
        SessionTodoStore.from_snapshot(
            {
                "next_id": 4,
                "tasks": [
                    {
                        "id": "3",
                        "version": 1,
                        "content": "Later",
                        "status": "pending",
                    },
                    {
                        "id": "1",
                        "version": 1,
                        "content": "Earlier",
                        "status": "completed",
                    },
                ],
            }
        )

    assert str(error.value) == "task ids out of order"


def test_snapshot_rejects_non_increasing_task_id_order() -> None:
    with pytest.raises(TodoStoreError) as error:
        SessionTodoStore.from_snapshot(
            {
                "next_id": 5,
                "tasks": [
                    {
                        "id": "1",
                        "version": 1,
                        "content": "First",
                        "status": "completed",
                    },
                    {
                        "id": "4",
                        "version": 1,
                        "content": "Third",
                        "status": "pending",
                    },
                    {
                        "id": "2",
                        "version": 1,
                        "content": "Second",
                        "status": "pending",
                    },
                ],
            }
        )

    assert str(error.value) == "task ids out of order"


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


def test_numeric_bound_is_the_largest_exact_json_integer() -> None:
    assert todo_store_module.MAX_TODO_NUMBER == (1 << 53) - 1


def test_snapshot_accepts_maximum_task_numbers_and_exhausted_next_id() -> None:
    maximum = todo_store_module.MAX_TODO_NUMBER
    payload = {
        "next_id": maximum + 1,
        "tasks": [
            {
                "id": str(maximum),
                "version": maximum,
                "content": "Boundary",
                "status": "pending",
            }
        ],
    }

    restored = SessionTodoStore.from_snapshot(payload)

    assert restored.export_snapshot() == payload
    assert restored.get(str(maximum)) == payload["tasks"][0]


@pytest.mark.parametrize(
    "task_id",
    [
        lambda: str(todo_store_module.MAX_TODO_NUMBER + 1),
        lambda: "9" * 100_000,
    ],
    ids=["one-over", "one-hundred-thousand-digits"],
)
def test_get_rejects_oversized_task_id_with_fixed_validation_error(
    task_id: Callable[[], str],
) -> None:
    store = SessionTodoStore()

    with pytest.raises(TodoStoreError, match="^invalid task id$"):
        store.get(task_id())


@pytest.mark.parametrize(
    "task_id",
    [
        lambda: str(todo_store_module.MAX_TODO_NUMBER + 1),
        lambda: "9" * 100_000,
    ],
    ids=["one-over", "one-hundred-thousand-digits"],
)
def test_snapshot_rejects_oversized_task_id_before_numeric_conversion(
    task_id: Callable[[], str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_conversion(value: str) -> int:
        raise AssertionError("oversized task ID reached numeric conversion")

    monkeypatch.setattr(todo_store_module, "_task_id_number", forbidden_conversion)

    with pytest.raises(TodoStoreError, match="^invalid task id$"):
        SessionTodoStore.from_snapshot(
            {
                "next_id": 1,
                "tasks": [
                    {
                        "id": task_id(),
                        "version": 1,
                        "content": "Oversized ID",
                        "status": "pending",
                    }
                ],
            }
        )


@pytest.mark.parametrize(
    "next_id",
    [lambda: todo_store_module.MAX_TODO_NUMBER + 2, lambda: 10**5000],
)
def test_snapshot_rejects_unusable_next_id_without_decimal_conversion(
    next_id: Callable[[], int],
) -> None:
    with pytest.raises(TodoStoreError, match="^invalid snapshot next id$"):
        SessionTodoStore.from_snapshot({"next_id": next_id(), "tasks": []})


@pytest.mark.parametrize(
    "version",
    [lambda: todo_store_module.MAX_TODO_NUMBER + 1, lambda: 10**5000],
)
def test_snapshot_rejects_unusable_version_without_decimal_conversion(
    version: Callable[[], int],
) -> None:
    with pytest.raises(TodoStoreError, match="^invalid task version$"):
        SessionTodoStore.from_snapshot(
            {
                "next_id": 2,
                "tasks": [
                    {
                        "id": "1",
                        "version": version(),
                        "content": "Boundary",
                        "status": "pending",
                    }
                ],
            }
        )


def test_create_uses_last_id_once_then_fails_without_reuse_or_callback() -> None:
    maximum = todo_store_module.MAX_TODO_NUMBER
    callbacks: list[list[dict[str, object]]] = []
    store = SessionTodoStore.from_snapshot({"next_id": maximum, "tasks": []})

    created = store.create(content="Last ID")
    store.update(
        task_id=str(maximum),
        expected_version=1,
        status="deleted",
    )
    before = store.export_snapshot()

    with pytest.raises(TodoStoreError, match="^task id space exhausted$"):
        store.create(content="Must not reuse", on_change=callbacks.append)

    assert created["id"] == str(maximum)
    assert before == {"next_id": maximum + 1, "tasks": []}
    assert store.export_snapshot() == before
    assert callbacks == []


def test_create_prefers_id_exhaustion_when_live_task_capacity_is_also_reached() -> None:
    maximum = todo_store_module.MAX_TODO_NUMBER
    task_numbers = [*range(1, MAX_TODO_ITEMS), maximum]
    payload = {
        "next_id": maximum + 1,
        "tasks": [
            {
                "id": str(task_number),
                "version": 1,
                "content": f"Task {index}",
                "status": "pending",
            }
            for index, task_number in enumerate(task_numbers)
        ],
    }
    callbacks: list[list[dict[str, object]]] = []
    store = SessionTodoStore.from_snapshot(payload)
    before = store.export_snapshot()

    with pytest.raises(TodoStoreError, match="^task id space exhausted$"):
        store.create(content="Cannot allocate", on_change=callbacks.append)

    assert store.export_snapshot() == before
    assert callbacks == []


@pytest.mark.parametrize("status", ["completed", "deleted"])
def test_update_rejects_version_exhaustion_atomically(status: str) -> None:
    maximum = todo_store_module.MAX_TODO_NUMBER
    callbacks: list[list[dict[str, object]]] = []
    payload = {
        "next_id": 2,
        "tasks": [
            {
                "id": "1",
                "version": maximum,
                "content": "Version boundary",
                "status": "pending",
            }
        ],
    }
    store = SessionTodoStore.from_snapshot(payload)

    with pytest.raises(TodoStoreError, match="^task version exhausted$"):
        store.update(
            task_id="1",
            expected_version=maximum,
            status=status,
            on_change=callbacks.append,
        )

    assert store.export_snapshot() == payload
    assert callbacks == []


def test_update_allows_last_version_and_callbacks_max_snapshot() -> None:
    maximum = todo_store_module.MAX_TODO_NUMBER
    callbacks: list[list[dict[str, object]]] = []
    store = SessionTodoStore.from_snapshot(
        {
            "next_id": 2,
            "tasks": [
                {
                    "id": "1",
                    "version": maximum - 1,
                    "content": "Before boundary update",
                    "status": "pending",
                }
            ],
        }
    )
    expected = {
        "id": "1",
        "version": maximum,
        "content": "At numeric ceiling",
        "status": "completed",
    }

    result = store.update(
        task_id="1",
        expected_version=maximum - 1,
        content="At numeric ceiling",
        status="completed",
        on_change=callbacks.append,
    )

    assert result == expected
    assert store.get("1") == expected
    assert callbacks == [[expected]]


def test_delete_allows_last_version_and_callbacks_empty_snapshot() -> None:
    maximum = todo_store_module.MAX_TODO_NUMBER
    callbacks: list[list[dict[str, object]]] = []
    store = SessionTodoStore.from_snapshot(
        {
            "next_id": 2,
            "tasks": [
                {
                    "id": "1",
                    "version": maximum - 1,
                    "content": "Delete at boundary",
                    "status": "pending",
                }
            ],
        }
    )

    result = store.update(
        task_id="1",
        expected_version=maximum - 1,
        status="deleted",
        on_change=callbacks.append,
    )

    assert result == {"id": "1", "deleted": True, "version": maximum}
    with pytest.raises(TodoStoreError, match="^task not found$"):
        store.get("1")
    assert callbacks == [[]]


def test_stale_conflict_precedes_version_exhaustion_at_max() -> None:
    maximum = todo_store_module.MAX_TODO_NUMBER
    content = "CURRENT-VERSION-CONTENT-SENTINEL"
    payload = {
        "next_id": 2,
        "tasks": [
            {
                "id": "1",
                "version": maximum,
                "content": content,
                "status": "pending",
            }
        ],
    }
    callbacks: list[list[dict[str, object]]] = []
    store = SessionTodoStore.from_snapshot(payload)

    with pytest.raises(
        TodoStoreError,
        match="^task version conflict; use todo_get and retry$",
    ) as exc_info:
        store.update(
            task_id="1",
            expected_version=maximum - 1,
            status="completed",
            on_change=callbacks.append,
        )

    assert exc_info.value.args == ("task version conflict; use todo_get and retry",)
    assert str(maximum) not in str(exc_info.value)
    assert content not in str(exc_info.value)
    assert store.export_snapshot() == payload
    assert callbacks == []


def test_missing_task_precedes_version_exhaustion_at_max() -> None:
    maximum = todo_store_module.MAX_TODO_NUMBER
    callbacks: list[list[dict[str, object]]] = []
    store = SessionTodoStore()
    before = store.export_snapshot()

    with pytest.raises(TodoStoreError, match="^task not found$"):
        store.update(
            task_id=str(maximum),
            expected_version=maximum,
            content="Valid mutation for a missing bounded ID",
            on_change=callbacks.append,
        )

    assert store.export_snapshot() == before
    assert callbacks == []


def test_version_exhaustion_precedes_in_progress_invariant() -> None:
    maximum = todo_store_module.MAX_TODO_NUMBER
    payload = {
        "next_id": 3,
        "tasks": [
            {
                "id": "1",
                "version": 1,
                "content": "Already active",
                "status": "in_progress",
            },
            {
                "id": "2",
                "version": maximum,
                "content": "At version ceiling",
                "status": "pending",
            },
        ],
    }
    callbacks: list[list[dict[str, object]]] = []
    store = SessionTodoStore.from_snapshot(payload)

    with pytest.raises(TodoStoreError, match="^task version exhausted$"):
        store.update(
            task_id="2",
            expected_version=maximum,
            status="in_progress",
            on_change=callbacks.append,
        )

    assert store.export_snapshot() == payload
    assert callbacks == []


@pytest.mark.parametrize(
    ("task_id", "expected_version"),
    [
        ("2", todo_store_module.MAX_TODO_NUMBER),
        ("1", todo_store_module.MAX_TODO_NUMBER - 1),
        ("1", todo_store_module.MAX_TODO_NUMBER),
    ],
    ids=["before-lookup", "before-cas", "before-exhaustion"],
)
def test_delete_only_validation_precedes_version_exhaustion_at_max(
    task_id: str,
    expected_version: int,
) -> None:
    maximum = todo_store_module.MAX_TODO_NUMBER
    payload = {
        "next_id": 2,
        "tasks": [
            {
                "id": "1",
                "version": maximum,
                "content": "Unchanged at maximum",
                "status": "pending",
            }
        ],
    }
    callbacks: list[list[dict[str, object]]] = []
    store = SessionTodoStore.from_snapshot(payload)

    with pytest.raises(
        TodoStoreError, match="^delete must be the only mutation field$"
    ):
        store.update(
            task_id=task_id,
            expected_version=expected_version,
            status="deleted",
            content="Invalid extra field",
            on_change=callbacks.append,
        )

    assert store.export_snapshot() == payload
    assert callbacks == []


def test_invalid_expected_version_precedes_exhaustion_at_max() -> None:
    maximum = todo_store_module.MAX_TODO_NUMBER
    payload = {
        "next_id": 2,
        "tasks": [
            {
                "id": "1",
                "version": maximum,
                "content": "Unchanged at maximum",
                "status": "pending",
            }
        ],
    }
    callbacks: list[list[dict[str, object]]] = []
    store = SessionTodoStore.from_snapshot(payload)

    with pytest.raises(TodoStoreError, match="^invalid expected_version$"):
        store.update(
            task_id="2",
            expected_version=maximum + 1,
            status="completed",
            on_change=callbacks.append,
        )

    assert store.export_snapshot() == payload
    assert callbacks == []


def test_update_rejects_expected_version_above_numeric_bound() -> None:
    callbacks: list[list[dict[str, object]]] = []
    store = SessionTodoStore()
    original = store.create(content="Original")

    with pytest.raises(TodoStoreError, match="^invalid expected_version$"):
        store.update(
            task_id="1",
            expected_version=todo_store_module.MAX_TODO_NUMBER + 1,
            content="Replacement",
            on_change=callbacks.append,
        )

    assert store.get("1") == original
    assert callbacks == []


def test_update_applies_only_supplied_fields_and_increments_version() -> None:
    store = SessionTodoStore()
    store.create(content="Original", active_form="Working")

    updated = store.update(
        task_id="1",
        expected_version=1,
        status="completed",
    )

    assert updated == {
        "id": "1",
        "version": 2,
        "content": "Original",
        "status": "completed",
        "activeForm": "Working",
    }


def test_same_value_update_is_a_successful_versioned_mutation() -> None:
    snapshots: list[list[dict[str, object]]] = []
    store = SessionTodoStore()
    store.create(content="Unchanged")

    updated = store.update(
        task_id="1",
        expected_version=1,
        content="Unchanged",
        on_change=snapshots.append,
    )

    assert updated["version"] == 2
    assert snapshots == [[updated]]


def test_update_none_removes_active_form() -> None:
    store = SessionTodoStore()
    store.create(content="Task", active_form="Working")

    updated = store.update(
        task_id="1",
        expected_version=1,
        active_form=None,
    )

    assert updated == {
        "id": "1",
        "version": 2,
        "content": "Task",
        "status": "pending",
    }


def test_delete_is_versioned_and_returns_the_exact_tombstone() -> None:
    snapshots: list[list[dict[str, object]]] = []
    store = SessionTodoStore()
    store.create(content="Delete me")

    deleted = store.update(
        task_id="1",
        expected_version=1,
        status="deleted",
        on_change=snapshots.append,
    )

    assert deleted == {"id": "1", "deleted": True, "version": 2}
    assert snapshots == [[]]
    with pytest.raises(TodoStoreError, match="^task not found$"):
        store.get("1")
    with pytest.raises(TodoStoreError, match="^task not found$"):
        store.update(task_id="1", expected_version=1, content="Too late")


@pytest.mark.parametrize(
    "extra_fields",
    [
        {"content": "combined"},
        {"active_form": None},
        {"content": "combined", "active_form": "combined"},
    ],
)
def test_delete_command_must_be_the_only_mutation_field(
    extra_fields: dict[str, object],
) -> None:
    callback_snapshots: list[list[dict[str, object]]] = []
    store = SessionTodoStore()
    original = store.create(content="Keep me")

    with pytest.raises(TodoStoreError, match="delete"):
        store.update(
            task_id="1",
            expected_version=999,
            status="deleted",
            on_change=callback_snapshots.append,
            **extra_fields,
        )

    assert store.get("1") == original
    assert callback_snapshots == []


def test_stale_update_uses_a_fixed_nonreflective_retry_error() -> None:
    secret_content = "PRIVATE-CONTENT-13216"
    callbacks: list[list[dict[str, object]]] = []
    store = SessionTodoStore()
    store.create(content="Original")
    winner = store.update(
        task_id="1",
        expected_version=1,
        content=secret_content,
    )

    with pytest.raises(TodoStoreError) as error:
        store.update(
            task_id="1",
            expected_version=1,
            content="STALE-CALLER-CONTENT",
            on_change=callbacks.append,
        )

    assert str(error.value) == "task version conflict; use todo_get and retry"
    assert secret_content not in str(error.value)
    assert "STALE-CALLER-CONTENT" not in str(error.value)
    assert store.get("1") == winner
    assert callbacks == []


def test_only_one_live_task_may_be_in_progress() -> None:
    callback_snapshots: list[list[dict[str, object]]] = []
    store = SessionTodoStore()
    store.create(content="First")
    second = store.create(content="Second")
    store.update(task_id="1", expected_version=1, status="in_progress")

    with pytest.raises(TodoStoreError, match="in_progress"):
        store.update(
            task_id="2",
            expected_version=1,
            status="in_progress",
            on_change=callback_snapshots.append,
        )

    assert store.get("2") == second
    assert callback_snapshots == []


@pytest.mark.parametrize(
    "update_kwargs",
    [
        {},
        {"content": ""},
        {"content": "bad\ud800"},
        {"active_form": b"bytes"},
        {"active_form": "bad\ud800"},
        {"status": "unknown"},
    ],
)
def test_update_validation_failures_are_atomic_and_emit_no_callback(
    update_kwargs: dict[str, object],
) -> None:
    callbacks: list[list[dict[str, object]]] = []
    store = SessionTodoStore()
    store.create(content="Original")
    before = store.export_snapshot()

    with pytest.raises(TodoStoreError):
        store.update(
            task_id="1",
            expected_version=1,
            on_change=callbacks.append,
            **update_kwargs,
        )

    assert store.export_snapshot() == before
    assert callbacks == []


@pytest.mark.parametrize(
    "expected_version",
    [None, True, 0, -1, 1.0, "1", _IntSubclass(1)],
)
def test_update_rejects_invalid_expected_versions_without_mutation(
    expected_version: object,
) -> None:
    callbacks: list[list[dict[str, object]]] = []
    store = SessionTodoStore()
    original = store.create(content="Original")

    with pytest.raises(TodoStoreError, match="expected_version"):
        store.update(
            task_id="1",
            expected_version=expected_version,
            content="Replacement",
            on_change=callbacks.append,
        )

    assert store.get("1") == original
    assert callbacks == []


def test_update_not_found_is_atomic_and_emits_no_callback() -> None:
    callbacks: list[list[dict[str, object]]] = []
    store = SessionTodoStore()
    before = store.export_snapshot()

    with pytest.raises(TodoStoreError, match="^task not found$"):
        store.update(
            task_id="99",
            expected_version=1,
            content="Missing",
            on_change=callbacks.append,
        )

    assert store.export_snapshot() == before
    assert callbacks == []


def test_create_update_and_delete_each_emit_exactly_one_callback() -> None:
    snapshots: list[list[dict[str, object]]] = []
    store = SessionTodoStore()

    store.create(content="Task", on_change=snapshots.append)
    store.update(
        task_id="1",
        expected_version=1,
        status="completed",
        on_change=snapshots.append,
    )
    store.update(
        task_id="1",
        expected_version=2,
        status="deleted",
        on_change=snapshots.append,
    )

    assert len(snapshots) == 3
    assert snapshots[0][0]["version"] == 1
    assert snapshots[1][0]["version"] == 2
    assert snapshots[2] == []


def test_create_validation_and_capacity_failures_emit_no_callback() -> None:
    callbacks: list[list[dict[str, object]]] = []
    store = SessionTodoStore()

    with pytest.raises(TodoStoreError):
        store.create(content="", on_change=callbacks.append)
    for index in range(MAX_TODO_ITEMS):
        store.create(content=f"Task {index}")
    with pytest.raises(TodoStoreError, match="limit"):
        store.create(content="Over capacity", on_change=callbacks.append)

    assert callbacks == []


def test_successive_callback_snapshots_follow_commit_order_and_are_defensive() -> None:
    snapshots: list[list[dict[str, object]]] = []
    store = SessionTodoStore()

    def mutate_first_snapshot(snapshot: list[dict[str, object]]) -> None:
        snapshots.append([dict(record) for record in snapshot])
        snapshot[0]["content"] = "outside mutation"
        snapshot.clear()

    store.create(content="Original", on_change=mutate_first_snapshot)
    store.update(
        task_id="1",
        expected_version=1,
        status="completed",
        on_change=snapshots.append,
    )

    assert snapshots == [
        [
            {
                "id": "1",
                "version": 1,
                "content": "Original",
                "status": "pending",
            }
        ],
        [
            {
                "id": "1",
                "version": 2,
                "content": "Original",
                "status": "completed",
            }
        ],
    ]
    assert store.get("1")["content"] == "Original"


def test_callback_failure_commits_returns_and_logs_one_fixed_safe_warning(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sentinel = (
        "content=PRIVATE-TASK-FRAGMENT "
        "credential=sk-live-CREDENTIAL-FRAGMENT "
        "path=/Users/alice/.ssh/PRIVATE-PATH-FRAGMENT"
    )
    store = SessionTodoStore()

    def fail_callback(snapshot: list[dict[str, object]]) -> None:
        raise RuntimeError(f"{sentinel} snapshot={snapshot!r}")

    with caplog.at_level(
        logging.WARNING,
        logger="tldw_chatbook.Agents.session_todo_store",
    ):
        created = store.create(content="PRIVATE-TASK-FRAGMENT", on_change=fail_callback)

    assert created == store.get("1")
    records = [
        record
        for record in caplog.records
        if record.name == "tldw_chatbook.Agents.session_todo_store"
    ]
    assert len(records) == 1
    record = records[0]
    assert record.msg == "Session todo change callback failed."
    assert record.args == ()
    assert record.exc_info is None
    assert record.exc_text is None
    assert record.stack_info is None
    captured = capsys.readouterr()
    exposed = " ".join(
        (
            record.getMessage(),
            repr(record.args),
            repr(record.exc_info),
            repr(record.exc_text),
            repr(record.stack_info),
            captured.out,
            captured.err,
        )
    )
    for fragment in (
        "PRIVATE-TASK-FRAGMENT",
        "CREDENTIAL-FRAGMENT",
        "PRIVATE-PATH-FRAGMENT",
        "sk-live",
        ".ssh",
    ):
        assert fragment not in exposed


@pytest.mark.parametrize("mutation", ["create", "update", "delete"])
def test_callback_reentrant_mutation_fails_without_deadlock_or_payload_leak(
    mutation: str,
) -> None:
    outer, inner_error, snapshot = _run_spawned_target(
        _callback_mutation_process,
        mutation,
        timeout=2,
    )

    assert outer == {
        "id": "1",
        "version": 1,
        "content": "Outer",
        "status": "pending",
    }
    assert inner_error == (
        "TodoStoreError",
        "task mutation is not allowed from an on_change callback",
    )
    assert "PRIVATE-INNER-CONTENT" not in inner_error[1]
    assert snapshot == {"next_id": 2, "tasks": [outer]}


def test_callback_base_exception_is_contained_after_commit() -> None:
    outer, snapshot = _run_spawned_target(_callback_base_exception_process)

    assert snapshot == {"next_id": 2, "tasks": [outer]}


def test_spawn_regression_cleanup_reaps_a_stuck_child() -> None:
    with pytest.raises(_SpawnProcessTimeout) as error:
        _run_spawned_target(_stuck_process, timeout=0.2)

    assert error.value.pid is not None
    assert error.value.pid not in {
        child.pid for child in multiprocessing.active_children()
    }


def test_spawn_start_failure_closes_queue_and_process_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeQueue:
        def close(self) -> None:
            events.append("queue.close")

        def join_thread(self) -> None:
            events.append("queue.join_thread")

    class FakeProcess:
        pid = None

        def start(self) -> None:
            events.append("process.start")
            raise RuntimeError("start failed")

        def is_alive(self) -> bool:
            raise AssertionError("unstarted process lifecycle was queried")

        def close(self) -> None:
            events.append("process.close")

    queue = FakeQueue()
    process = FakeProcess()

    class FakeContext:
        def Queue(self) -> FakeQueue:  # noqa: N802
            return queue

        def Process(self, **kwargs: object) -> FakeProcess:  # noqa: N802
            assert kwargs["target"] is _callback_read_process
            return process

    monkeypatch.setattr(
        multiprocessing,
        "get_context",
        lambda method: FakeContext(),
    )

    with pytest.raises(RuntimeError, match="^start failed$"):
        _run_spawned_target(_callback_read_process)

    assert events == [
        "process.start",
        "process.close",
        "queue.close",
        "queue.join_thread",
    ]


def test_concurrent_creates_allocate_distinct_present_tasks() -> None:
    store = SessionTodoStore()

    results, errors = _run_forced_mutation_pair(
        store,
        lambda: store.create(content="First"),
        lambda: store.create(content="Second"),
        operation="setitem",
    )

    assert errors == [None, None]
    assert {result["id"] for result in results if type(result) is dict} == {"1", "2"}
    assert {record["content"] for record in store.list_after(None)} == {
        "First",
        "Second",
    }


def test_concurrent_jointly_valid_updates_to_different_tasks_both_commit() -> None:
    store = SessionTodoStore()
    store.create(content="First")
    store.create(content="Second")

    _, errors = _run_forced_mutation_pair(
        store,
        lambda: store.update(task_id="1", expected_version=1, content="First done"),
        lambda: store.update(task_id="2", expected_version=1, content="Second done"),
        operation="get",
    )

    assert errors == [None, None]
    assert store.get("1")["content"] == "First done"
    assert store.get("2")["content"] == "Second done"


def test_concurrent_same_task_cas_has_one_winner_and_one_fixed_conflict() -> None:
    store = SessionTodoStore()
    store.create(content="Original")

    results, errors = _run_forced_mutation_pair(
        store,
        lambda: store.update(task_id="1", expected_version=1, content="First"),
        lambda: store.update(task_id="1", expected_version=1, content="Second"),
        operation="get",
    )

    assert len([result for result in results if result is not None]) == 1
    failures = [error for error in errors if error is not None]
    assert len(failures) == 1
    assert type(failures[0]) is TodoStoreError
    assert str(failures[0]) == "task version conflict; use todo_get and retry"
    winner = store.get("1")
    assert winner["version"] == 2
    assert winner["content"] in {"First", "Second"}


def test_concurrent_in_progress_transitions_allow_only_one_winner() -> None:
    store = SessionTodoStore()
    store.create(content="First")
    store.create(content="Second")

    results, errors = _run_forced_mutation_pair(
        store,
        lambda: store.update(task_id="1", expected_version=1, status="in_progress"),
        lambda: store.update(task_id="2", expected_version=1, status="in_progress"),
        operation="get",
    )

    assert len([result for result in results if result is not None]) == 1
    assert len([error for error in errors if type(error) is TodoStoreError]) == 1
    assert [record["status"] for record in store.list_after(None)].count(
        "in_progress"
    ) == 1


def test_concurrent_creates_at_capacity_have_exactly_one_winner() -> None:
    store = SessionTodoStore()
    for index in range(MAX_TODO_ITEMS - 1):
        store.create(content=f"Existing {index}")

    results, errors = _run_forced_mutation_pair(
        store,
        lambda: store.create(content="First contender"),
        lambda: store.create(content="Second contender"),
        operation="setitem",
    )

    assert len([result for result in results if result is not None]) == 1
    failures = [error for error in errors if error is not None]
    assert len(failures) == 1
    assert type(failures[0]) is TodoStoreError
    assert str(failures[0]) == "task limit reached"
    snapshot = store.export_snapshot()
    assert len(snapshot["tasks"]) == MAX_TODO_ITEMS
    assert snapshot["next_id"] == MAX_TODO_ITEMS + 1


def test_terminal_id_race_has_one_winner_and_fixed_exhaustion_loser() -> None:
    maximum = todo_store_module.MAX_TODO_NUMBER
    store = SessionTodoStore.from_snapshot(
        {
            "next_id": maximum,
            "tasks": [
                {
                    "id": str(task_number),
                    "version": 1,
                    "content": f"Existing {task_number}",
                    "status": "pending",
                }
                for task_number in range(1, MAX_TODO_ITEMS)
            ],
        }
    )
    callbacks: list[list[dict[str, object]]] = []

    results, errors = _run_forced_mutation_pair(
        store,
        lambda: store.create(
            content="First terminal contender",
            on_change=callbacks.append,
        ),
        lambda: store.create(
            content="Second terminal contender",
            on_change=callbacks.append,
        ),
        operation="setitem",
    )

    successes = [result for result in results if type(result) is dict]
    failures = [error for error in errors if error is not None]
    assert len(successes) == 1
    assert successes[0]["id"] == str(maximum)
    assert len(failures) == 1
    assert type(failures[0]) is TodoStoreError
    assert failures[0].args == ("task id space exhausted",)

    snapshot = store.export_snapshot()
    assert snapshot["next_id"] == maximum + 1
    assert len(snapshot["tasks"]) == MAX_TODO_ITEMS
    task_ids = [record["id"] for record in snapshot["tasks"]]
    assert len(set(task_ids)) == MAX_TODO_ITEMS
    assert task_ids.count(str(maximum)) == 1
    assert callbacks == [snapshot["tasks"]]


def test_callback_serializes_mutation_commit_and_return_while_reads_stay_available() -> (
    None
):
    entered = Event()
    release = Event()
    second_waiting_on_mutation_lock = Event()
    second_done = Event()
    store = SessionTodoStore()
    store._mutation_lock = _ObservedLock(  # type: ignore[attr-defined]
        store._mutation_lock,  # type: ignore[attr-defined]
        thread_name="second-mutation",
        before_acquire=second_waiting_on_mutation_lock.set,
    )

    def blocked(snapshot: list[dict[str, object]]) -> None:
        assert snapshot[0]["content"] == "First"
        entered.set()
        release.wait(5)

    first = Thread(
        target=lambda: store.create(content="First", on_change=blocked),
        name="first-mutation",
    )

    def second_mutation() -> None:
        store.create(content="Second")
        second_done.set()

    second = Thread(target=second_mutation, name="second-mutation")
    first.start()
    try:
        assert entered.wait(2)
        second.start()
        assert second_waiting_on_mutation_lock.wait(2)
        assert store.get("1")["content"] == "First"
        assert [record["id"] for record in store.list_after(None)] == ["1"]
        assert not second_done.is_set()
    finally:
        release.set()
        first.join(5)
        if second.ident is not None:
            second.join(5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert second_done.is_set()
    assert [record["id"] for record in store.list_after(None)] == ["1", "2"]


def test_reads_and_mutations_enter_the_real_state_lock() -> None:
    observed = Event()
    store = SessionTodoStore()
    store.create(content="Task")
    store._state_lock = _ObservedLock(  # type: ignore[assignment]
        store._state_lock,
        thread_name=current_thread().name,
        before_acquire=observed.set,
    )

    store.get("1")
    assert observed.is_set()
    observed.clear()

    store.create(content="Second")
    assert observed.is_set()


def test_callback_can_read_directly_and_from_another_thread_without_deadlock() -> None:
    direct_record, threaded_records, callback_snapshot = _run_spawned_target(
        _callback_read_process
    )
    expected = {
        "id": "1",
        "version": 1,
        "content": "Readable",
        "status": "pending",
    }
    assert direct_record == expected
    assert threaded_records == [expected]
    assert callback_snapshot == [expected]
