"""Caller-owned transaction extension point for Console sidecar persistence."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from typing import Protocol, cast


_SIMPLE_IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*"
_INSERT_VALUES_PATTERN = re.compile(
    rf"\A\s*INSERT\s+INTO\s+{_SIMPLE_IDENTIFIER}\s*"
    rf"\(\s*(?P<columns>{_SIMPLE_IDENTIFIER}(?:\s*,\s*{_SIMPLE_IDENTIFIER})*)\s*\)"
    rf"\s*VALUES\s*\(\s*(?P<placeholders>\?(?:\s*,\s*\?)*)\s*\)\s*\Z",
    re.ASCII | re.IGNORECASE,
)


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
        arity = self.__statement_arity(statement)
        if type(parameters) is not tuple:
            raise TypeError("Console transaction parameters must be a tuple.")
        if len(parameters) != arity:
            raise ValueError(
                "Console INSERT columns, placeholders, and parameters require the "
                "same non-zero arity."
            )
        self.__cursor.execute(statement, parameters)

    def executemany(
        self,
        statement: str,
        parameter_rows: Iterable[tuple[object, ...]],
        /,
    ) -> None:
        """Execute validated parameterized INSERT rows."""
        self.__require_active()
        arity = self.__statement_arity(statement)
        rows = tuple(parameter_rows)
        if any(type(row) is not tuple for row in rows):
            raise TypeError("Console transaction parameter rows must be tuples.")
        if not rows or any(len(row) != arity for row in rows):
            raise ValueError(
                "Console INSERT executemany requires non-empty rows of matching arity."
            )
        self.__cursor.executemany(statement, rows)

    def _revoke(self) -> None:
        self.__active = False

    def __require_active(self) -> None:
        if not self.__active:
            raise RuntimeError("Console writer requires an active contribution.")

    @staticmethod
    def __statement_arity(statement: str) -> int:
        if type(statement) is not str:
            raise TypeError("Console transaction statement must be a string.")
        match = _INSERT_VALUES_PATTERN.fullmatch(statement)
        if match is None:
            raise ValueError(
                "Console transaction writer accepts one parameterized INSERT INTO "
                "simple columns VALUES placeholders statement."
            )
        column_count = match.group("columns").count(",") + 1
        placeholder_count = match.group("placeholders").count(",") + 1
        if column_count != placeholder_count:
            raise ValueError(
                "Console INSERT columns, placeholders, and parameters require the "
                "same non-zero arity."
            )
        return column_count


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
