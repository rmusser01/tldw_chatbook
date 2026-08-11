"""Session-local task records for Console agents."""

from __future__ import annotations

from threading import Lock

MAX_TODO_ITEMS = 50
MAX_TODO_CONTENT_CHARS = 500
TODO_STATUSES = ("pending", "in_progress", "completed")

TodoRecord = dict[str, object]

_MISSING = object()


class TodoStoreError(ValueError):
    """Report a bounded validation or lookup failure."""


def _validate_text(value: object, *, field: str, allow_blank: bool) -> str:
    if type(value) is not str:
        raise TodoStoreError(f"{field} must be a string")
    if not allow_blank and not value.strip():
        raise TodoStoreError(f"{field} must not be blank")
    if len(value) > MAX_TODO_CONTENT_CHARS:
        raise TodoStoreError(f"{field} is too long")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        raise TodoStoreError(f"{field} must be valid UTF-8") from None
    return value


def _validate_task_id(task_id: object) -> str:
    if (
        type(task_id) is not str
        or not task_id
        or not task_id.isascii()
        or not task_id.isdecimal()
        or task_id[0] == "0"
    ):
        raise TodoStoreError("invalid task id")
    return task_id


def _task_id_number(task_id: str) -> int:
    number = 0
    for character in task_id:
        number = (number * 10) + (ord(character) - ord("0"))
    return number


class SessionTodoStore:
    """Own stable-ID task records for one in-process Console session."""

    def __init__(self) -> None:
        self._tasks: dict[str, TodoRecord] = {}
        self._next_id = 1
        self._state_lock = Lock()

    def create(self, *, content: object, active_form: object = _MISSING) -> TodoRecord:
        """Create a pending task and return a defensive record copy."""
        valid_content = _validate_text(content, field="content", allow_blank=False)
        valid_active_form: str | object = _MISSING
        if active_form is not _MISSING:
            valid_active_form = _validate_text(
                active_form, field="activeForm", allow_blank=True
            )

        with self._state_lock:
            if len(self._tasks) >= MAX_TODO_ITEMS:
                raise TodoStoreError("task limit reached")
            task_id = str(self._next_id)
            record: TodoRecord = {
                "id": task_id,
                "version": 1,
                "content": valid_content,
                "status": "pending",
            }
            if valid_active_form is not _MISSING:
                record["activeForm"] = valid_active_form
            self._tasks[task_id] = record
            self._next_id += 1
            return dict(record)

    def get(self, task_id: object) -> TodoRecord:
        """Return a defensive copy of one task record."""
        valid_task_id = _validate_task_id(task_id)
        with self._state_lock:
            record = self._tasks.get(valid_task_id)
            if record is None:
                raise TodoStoreError("task not found")
            return dict(record)

    def list_after(self, cursor: int | None) -> list[TodoRecord]:
        """Return task copies with numeric IDs greater than ``cursor``."""
        if cursor is not None and (type(cursor) is not int or cursor < 0):
            raise TodoStoreError("invalid cursor")
        lower_bound = 0 if cursor is None else cursor
        with self._state_lock:
            return [
                dict(record)
                for task_id, record in self._tasks.items()
                if _task_id_number(task_id) > lower_bound
            ]

    def export_snapshot(self) -> dict[str, object]:
        """Return a defensive pure-data projection of the current state."""
        with self._state_lock:
            return {
                "next_id": self._next_id,
                "tasks": [dict(record) for record in self._tasks.values()],
            }

    @classmethod
    def from_snapshot(cls, payload: object) -> SessionTodoStore:
        """Restore a store from a fully validated pure-data snapshot."""
        if type(payload) is not dict or set(payload) != {"next_id", "tasks"}:
            raise TodoStoreError("invalid task snapshot")

        next_id = payload["next_id"]
        tasks = payload["tasks"]
        if type(next_id) is not int or next_id < 1:
            raise TodoStoreError("invalid snapshot next id")
        if type(tasks) is not list or len(tasks) > MAX_TODO_ITEMS:
            raise TodoStoreError("invalid snapshot tasks")

        restored_tasks: dict[str, TodoRecord] = {}
        max_task_id = 0
        in_progress_count = 0
        required_keys = {"id", "version", "content", "status"}
        allowed_keys = required_keys | {"activeForm"}

        for task in tasks:
            if (
                type(task) is not dict
                or not required_keys.issubset(task)
                or not set(task).issubset(allowed_keys)
            ):
                raise TodoStoreError("invalid snapshot task")

            task_id = _validate_task_id(task["id"])
            if task_id in restored_tasks:
                raise TodoStoreError("duplicate task id")
            version = task["version"]
            if type(version) is not int or version < 1:
                raise TodoStoreError("invalid task version")
            content = _validate_text(
                task["content"], field="content", allow_blank=False
            )
            status = task["status"]
            if type(status) is not str or status not in TODO_STATUSES:
                raise TodoStoreError("invalid task status")

            restored_record: TodoRecord = {
                "id": task_id,
                "version": version,
                "content": content,
                "status": status,
            }
            if "activeForm" in task:
                restored_record["activeForm"] = _validate_text(
                    task["activeForm"], field="activeForm", allow_blank=True
                )
            restored_tasks[task_id] = restored_record
            max_task_id = max(max_task_id, _task_id_number(task_id))
            if status == "in_progress":
                in_progress_count += 1
                if in_progress_count > 1:
                    raise TodoStoreError("multiple active tasks")

        if next_id <= max_task_id:
            raise TodoStoreError("invalid snapshot next id")

        store = cls()
        with store._state_lock:
            store._tasks = restored_tasks
            store._next_id = next_id
        return store
