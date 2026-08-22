"""Caller-owned transaction extension point for Console sidecar persistence."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from typing import Protocol, cast


class ConsoleTransactionWriter(Protocol):
    """Insert-only capability scoped to one caller-owned transaction callback."""

    def execute(self, statement: str, parameters: tuple[object, ...], /) -> None:
        """Execute one parameterized INSERT through the caller transaction."""

    def executemany(
        self,
        statement: str,
        parameter_rows: Iterable[tuple[object, ...]],
        /,
    ) -> None:
        """Execute parameterized INSERT rows through the caller transaction."""


class ConsoleTransactionContribution(Protocol):
    """Write one sidecar through an existing atomic Console transaction."""

    def write(
        self,
        *,
        writer: ConsoleTransactionWriter,
        conversation_id: str,
        message_ids: Mapping[str, str],
    ) -> None:
        """Write through the caller-owned capability without committing."""


class _CursorConsoleTransactionWriter:
    __slots__ = ("__cursor", "__active")

    def __init__(self, cursor: sqlite3.Cursor) -> None:
        self.__cursor = cursor
        self.__active = True

    def execute(self, statement: str, parameters: tuple[object, ...], /) -> None:
        """Execute one validated parameterized INSERT."""
        self.__require_active()
        self.__validate_statement(statement)
        if type(parameters) is not tuple:
            raise TypeError("Console transaction parameters must be a tuple.")
        self.__cursor.execute(statement, parameters)

    def executemany(
        self,
        statement: str,
        parameter_rows: Iterable[tuple[object, ...]],
        /,
    ) -> None:
        """Execute validated parameterized INSERT rows."""
        self.__require_active()
        self.__validate_statement(statement)
        rows = tuple(parameter_rows)
        if any(type(row) is not tuple for row in rows):
            raise TypeError("Console transaction parameter rows must be tuples.")
        self.__cursor.executemany(statement, rows)

    def _revoke(self) -> None:
        self.__active = False

    def __require_active(self) -> None:
        if not self.__active:
            raise RuntimeError("Console writer requires an active contribution.")

    @staticmethod
    def __validate_statement(statement: str) -> None:
        if type(statement) is not str:
            raise TypeError("Console transaction statement must be a string.")
        normalized = statement.strip()
        first_token = normalized.split(maxsplit=1)[0].upper() if normalized else ""
        if (
            not normalized
            or first_token != "INSERT"
            or ";" in normalized
            or "?" not in normalized
        ):
            raise ValueError(
                "Console transaction writer accepts one parameterized INSERT."
            )


@contextmanager
def _scoped_console_transaction_writer(
    cursor: sqlite3.Cursor,
) -> Iterator[ConsoleTransactionWriter]:
    if not cursor.connection.in_transaction:
        raise sqlite3.DatabaseError(
            "A contribution requires the caller-owned transaction."
        )
    writer = _CursorConsoleTransactionWriter(cursor)
    try:
        yield cast(ConsoleTransactionWriter, writer)
    finally:
        writer._revoke()
