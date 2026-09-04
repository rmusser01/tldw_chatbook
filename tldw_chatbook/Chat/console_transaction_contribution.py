"""Caller-owned transaction extension point for Console sidecar persistence."""

from __future__ import annotations

import re
import sqlite3
from abc import ABC, abstractmethod
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
_SQLITE_MAX_INTEGER = (1 << 63) - 1


class ConsoleTransactionWriter(Protocol):
    """Narrow capability scoped to one caller-owned contribution transaction."""

    def next_trajectory_sequence(self) -> int:
        """Allocate one seq for the accepted conversation in this transaction."""

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


class ConsoleExactNativeIdTransactionContribution(ABC):
    """Write a sidecar using only exact native-to-durable message identities."""

    __slots__ = ()

    @abstractmethod
    def write_exact(
        self,
        *,
        writer: ConsoleTransactionWriter,
        conversation_id: str,
        native_message_ids: Mapping[str, str],
    ) -> None:
        """Write without the legacy user/assistant role aliases."""


ConsolePromotionTransactionContribution = (
    ConsoleTransactionContribution | ConsoleExactNativeIdTransactionContribution
)


class ConsoleDurableFingerprintContribution(Protocol):
    """Optional canonical plan input for a non-frozen durable contribution."""

    def durable_acceptance_fingerprint(self) -> Mapping[str, object]:
        """Return bounded immutable data which fully determines the write."""


class _CursorConsoleTransactionWriter:
    __slots__ = (
        "__cursor",
        "__active",
        "__conversation_id",
        "__next_trajectory_sequence",
    )

    def __init__(self, cursor: sqlite3.Cursor, conversation_id: str) -> None:
        self.__cursor = cursor
        self.__active = True
        self.__conversation_id = conversation_id
        self.__next_trajectory_sequence: int | None = None

    def next_trajectory_sequence(self) -> int:
        """Allocate one trajectory sequence through the private caller cursor."""
        self.__require_active()
        if self.__next_trajectory_sequence is None:
            row = self.__cursor.execute(
                "SELECT MAX(seq) FROM message_trajectory_metadata "
                "WHERE conversation_id = ?",
                (self.__conversation_id,),
            ).fetchone()
            if row is None:
                raise sqlite3.DatabaseError(
                    "Unable to read the trajectory sequence maximum."
                )
            maximum = row[0]
            if maximum is None:
                self.__next_trajectory_sequence = 1
            elif (
                type(maximum) is not int
                or maximum < 0
                or maximum >= _SQLITE_MAX_INTEGER
            ):
                raise sqlite3.DatabaseError("Invalid trajectory sequence maximum.")
            else:
                self.__next_trajectory_sequence = maximum + 1
        if self.__next_trajectory_sequence > _SQLITE_MAX_INTEGER:
            raise sqlite3.DatabaseError("Trajectory sequence exceeds SQLite limits.")
        allocated = self.__next_trajectory_sequence
        self.__next_trajectory_sequence += 1
        return allocated

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
    conversation_id: str,
) -> Iterator[ConsoleTransactionWriter]:
    if not cursor.connection.in_transaction:
        raise sqlite3.DatabaseError(
            "A contribution requires the caller-owned transaction."
        )
    writer = _CursorConsoleTransactionWriter(cursor, conversation_id)
    try:
        yield cast(ConsoleTransactionWriter, writer)
    finally:
        writer._revoke()
