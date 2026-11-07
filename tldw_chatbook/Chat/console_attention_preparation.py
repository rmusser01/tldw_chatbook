"""Finite stock attention-mark reads; no permission or receipt cache."""

from __future__ import annotations

import inspect
from dataclasses import dataclass

from ..DB import ChaChaNotes_DB as notes
from ..DB import base_db
from . import conversation_local_marks_service as marks

_MARKS_TYPE = marks.ConversationLocalMarksService
_NOTES_TYPE = notes.CharactersRAGDB
_TRANSACTION_TYPE = notes.TransactionContextManager
_OWNED_CONNECTION = base_db.operation_owned_connection
_MARK_NAMES = (
    "list_console_unseen_marks",
    "parse_console_unseen_mark_type",
    "validate_terminal_receipt_id",
)
_DB_NAMES = (
    "transaction",
    "get_connection",
    "_get_thread_connection",
    "close_connection",
    "_close_connection_handle",
)
_DESCRIPTORS = tuple(
    (owner, name, inspect.getattr_static(owner, name))
    for owner, names in (
        (_MARKS_TYPE, _MARK_NAMES),
        (_NOTES_TYPE, _DB_NAMES),
        (
            _TRANSACTION_TYPE,
            (
                "__init__",
                "__enter__",
                "__exit__",
                "_enter_transaction",
                "_exit_transaction",
            ),
        ),
    )
    for name in names
)


def _functions(descriptor):
    function = (
        descriptor.__func__
        if type(descriptor) in (classmethod, staticmethod)
        else descriptor
    )
    while function is not None:
        yield function
        function = vars(function).get("__wrapped__")


_FUNCTIONS = tuple(
    (
        function,
        function.__code__,
        function.__defaults__,
        function.__kwdefaults__,
        vars(function).get("__wrapped__"),
    )
    for _owner, _name, descriptor in _DESCRIPTORS
    for function in _functions(descriptor)
)
_PREFIX = _MARKS_TYPE.CONSOLE_UNSEEN_PREFIX
_UUID = marks.UUID


def _stock_sources() -> bool:
    return (
        marks.ConversationLocalMarksService is _MARKS_TYPE
        and notes.CharactersRAGDB is _NOTES_TYPE
        and notes.TransactionContextManager is _TRANSACTION_TYPE
        and base_db.operation_owned_connection is _OWNED_CONNECTION
        and marks.UUID is _UUID
        and _MARKS_TYPE.CONSOLE_UNSEEN_PREFIX is _PREFIX
        and all(
            inspect.getattr_static(owner, name, None) is descriptor
            for owner, name, descriptor in _DESCRIPTORS
        )
        and all(
            function.__code__ is code
            and function.__defaults__ is defaults
            and function.__kwdefaults__ is kwdefaults
            and vars(function).get("__wrapped__") is wrapped
            for function, code, defaults, kwdefaults, wrapped in _FUNCTIONS
        )
    )


def _plain_receiver(value, owner, methods, fields) -> bool:
    return (
        type(value) is owner
        and inspect.getattr_static(owner, "__getattribute__") is object.__getattribute__
        and inspect.getattr_static(owner, "__getattr__", None) is None
        and all(inspect.getattr_static(owner, name, None) is None for name in fields)
        and not any(name in vars(value) for name in methods)
    )


@dataclass(frozen=True)
class AttentionMarksRead:
    service: object
    database: object
    database_fields: tuple

    def current(self) -> bool:
        if not _stock_sources() or not _plain_receiver(
            self.service, _MARKS_TYPE, (*_MARK_NAMES, "CONSOLE_UNSEEN_PREFIX"), ("db",)
        ):
            return False
        if not _plain_receiver(
            self.database,
            _NOTES_TYPE,
            _DB_NAMES,
            (
                "db_path",
                "db_path_str",
                "is_memory_db",
                "_local",
                "_connection_quiescence",
            ),
        ):
            return False
        fields = vars(self.database)
        return vars(self.service).get("db") is self.database and all(
            fields.get(name) is value for name, value in self.database_fields
        )

    def __call__(self) -> tuple[tuple[str, str], ...]:
        if not self.current():
            raise RuntimeError("console_attention_source_changed")
        with _OWNED_CONNECTION(self.database):
            return self.service.list_console_unseen_marks()


def capture_attention_marks_read(service: object) -> AttentionMarksRead | None:
    """Capture plain stock receivers; custom readers keep their caller affinity."""
    if not _stock_sources() or not _plain_receiver(
        service, _MARKS_TYPE, _MARK_NAMES, ("db",)
    ):
        return None
    database = vars(service).get("db")
    if (
        type(database) is not _NOTES_TYPE
        or vars(database).get("is_memory_db") is not False
    ):
        return None
    read = AttentionMarksRead(
        service,
        database,
        tuple(
            (name, vars(database).get(name))
            for name in (
                "db_path",
                "db_path_str",
                "is_memory_db",
                "_local",
                "_connection_quiescence",
            )
        ),
    )
    return read if read.current() else None
