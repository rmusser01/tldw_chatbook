"""Session-local task records for Console agents."""

from __future__ import annotations

import logging
from collections.abc import Callable
from threading import Lock

MAX_TODO_ITEMS = 50
MAX_TODO_CONTENT_CHARS = 500
TODO_STATUSES = ("pending", "in_progress", "completed")

TodoRecord = dict[str, object]
TodoChangeCallback = Callable[[list[TodoRecord]], None]

_MISSING = object()
_LOG = logging.getLogger(__name__)


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


def _validate_expected_version(value: object) -> int:
    if type(value) is not int or value < 1:
        raise TodoStoreError("expected_version must be an integer at least 1")
    return value


class SessionTodoStore:
    """Own stable-ID task records for one in-process Console session."""

    def __init__(self) -> None:
        self._tasks: dict[str, TodoRecord] = {}
        self._next_id = 1
        self._state_lock = Lock()
        self._mutation_lock = Lock()

    def _snapshot_locked(self) -> list[TodoRecord]:
        return [dict(record) for record in self._tasks.values()]

    def _mutate(
        self,
        commit: Callable[[], TodoRecord],
        on_change: TodoChangeCallback | None,
    ) -> TodoRecord:
        with self._mutation_lock:
            with self._state_lock:
                result = commit()
                snapshot = self._snapshot_locked()
            if on_change is not None:
                try:
                    on_change(snapshot)
                except Exception:
                    _LOG.warning("Session todo change callback failed.")
            return dict(result)

    def create(
        self,
        *,
        content: object,
        active_form: object = _MISSING,
        on_change: TodoChangeCallback | None = None,
    ) -> TodoRecord:
        """Create a pending task.

        Args:
            content: Required bounded task content.
            active_form: Optional bounded active-form label.
            on_change: Optional synchronous callback for the committed snapshot.

        Returns:
            A defensive copy of the created task record.

        Raises:
            TodoStoreError: If text is invalid or the store is at capacity.
        """
        valid_content = _validate_text(content, field="content", allow_blank=False)
        valid_active_form: str | object = _MISSING
        if active_form is not _MISSING:
            valid_active_form = _validate_text(
                active_form, field="activeForm", allow_blank=True
            )

        def commit() -> TodoRecord:
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
            return record

        return self._mutate(commit, on_change)

    def update(
        self,
        *,
        task_id: object,
        expected_version: object,
        content: object = _MISSING,
        status: object = _MISSING,
        active_form: object = _MISSING,
        on_change: TodoChangeCallback | None = None,
    ) -> TodoRecord:
        """Apply a compare-and-swap task patch or deletion.

        Args:
            task_id: Canonical positive decimal task ID.
            expected_version: Exact current positive integer version.
            content: Optional replacement content.
            status: Optional task status or the ``deleted`` command.
            active_form: Optional replacement label; ``None`` removes it.
            on_change: Optional synchronous callback for the committed snapshot.

        Returns:
            A defensive updated record or deletion tombstone.

        Raises:
            TodoStoreError: If validation, lookup, CAS, or invariants fail.
        """
        valid_task_id = _validate_task_id(task_id)
        valid_expected_version = _validate_expected_version(expected_version)
        if content is _MISSING and status is _MISSING and active_form is _MISSING:
            raise TodoStoreError("at least one mutation field is required")

        valid_status: str | object = _MISSING
        if status is not _MISSING:
            if type(status) is not str or status not in (*TODO_STATUSES, "deleted"):
                raise TodoStoreError("invalid task status")
            valid_status = status

        if valid_status == "deleted" and (
            content is not _MISSING or active_form is not _MISSING
        ):
            raise TodoStoreError("delete must be the only mutation field")

        valid_content: str | object = _MISSING
        if content is not _MISSING:
            valid_content = _validate_text(
                content,
                field="content",
                allow_blank=False,
            )

        valid_active_form: str | None | object = _MISSING
        if active_form is not _MISSING:
            if active_form is None:
                valid_active_form = None
            else:
                valid_active_form = _validate_text(
                    active_form,
                    field="activeForm",
                    allow_blank=True,
                )

        def commit() -> TodoRecord:
            record = self._tasks.get(valid_task_id)
            if record is None:
                raise TodoStoreError("task not found")
            if record["version"] != valid_expected_version:
                raise TodoStoreError("task version conflict; use todo_get and retry")

            new_version = valid_expected_version + 1
            if valid_status == "deleted":
                del self._tasks[valid_task_id]
                return {
                    "id": valid_task_id,
                    "deleted": True,
                    "version": new_version,
                }

            updated = dict(record)
            if valid_content is not _MISSING:
                updated["content"] = valid_content
            if valid_status is not _MISSING:
                updated["status"] = valid_status
            if valid_active_form is None:
                updated.pop("activeForm", None)
            elif valid_active_form is not _MISSING:
                updated["activeForm"] = valid_active_form

            if updated["status"] == "in_progress" and any(
                other_id != valid_task_id and other_record["status"] == "in_progress"
                for other_id, other_record in self._tasks.items()
            ):
                raise TodoStoreError("another task is already in_progress")

            updated["version"] = new_version
            self._tasks[valid_task_id] = updated
            return updated

        return self._mutate(commit, on_change)

    def get(self, task_id: object) -> TodoRecord:
        """Return one task record.

        Args:
            task_id: Canonical positive decimal task ID.

        Returns:
            A defensive copy of the matching task record.

        Raises:
            TodoStoreError: If the ID is invalid or the task is not found.
        """
        valid_task_id = _validate_task_id(task_id)
        with self._state_lock:
            record = self._tasks.get(valid_task_id)
            if record is None:
                raise TodoStoreError("task not found")
            return dict(record)

    def list_after(self, cursor: int | None) -> list[TodoRecord]:
        """Return task records after a numeric lower bound.

        Args:
            cursor: Exclusive numeric lower bound, or ``None`` for all tasks.

        Returns:
            Defensive task copies in creation order.

        Raises:
            TodoStoreError: If the cursor is not an exact nonnegative integer.
        """
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
        """Export the current navigation state.

        Returns:
            A defensive pure-data snapshot of the ID counter and task records.
        """
        with self._state_lock:
            return {
                "next_id": self._next_id,
                "tasks": [dict(record) for record in self._tasks.values()],
            }

    @classmethod
    def from_snapshot(cls, payload: object) -> SessionTodoStore:
        """Restore a store from navigation state.

        Args:
            payload: Pure-data snapshot to validate and import.

        Returns:
            A store populated from the validated snapshot.

        Raises:
            TodoStoreError: If the snapshot shape or task state is invalid.
        """
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
            task_id_number = _task_id_number(task_id)
            if task_id_number <= max_task_id:
                raise TodoStoreError("task ids out of order")
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
            max_task_id = task_id_number
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
