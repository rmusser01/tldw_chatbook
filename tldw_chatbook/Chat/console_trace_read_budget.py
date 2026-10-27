"""Private, nonrenewable materialization allowance for retained trace reads."""

import sqlite3
from collections.abc import Mapping
from dataclasses import fields, is_dataclass


class _TraceReadLimit(RuntimeError):
    """Body-free resource refusal, distinct from native content uncertainty."""


class _TraceReadBudget:
    """One allowance shared by both captured partitions and nested consumers.

    Reservations cover representation expansion, not a Python heap measurement.
    Strings use the four-byte Unicode upper bound. JSON container/token overhead
    is reserved by scanning its bounded raw representation before decoding; no
    decoded tree or second serializer is used to decide admission.
    """

    __slots__ = ("_remaining",)

    def __init__(self):
        self._remaining = 16 * 1024 * 1024

    def reserve(self, amount: int) -> None:
        if amount < 0 or amount > self._remaining:
            raise _TraceReadLimit("mobile_trace_input_limit")
        self._remaining -= amount

    def container(self, count: int = 0) -> None:
        self.reserve(64 + 32 * count)

    def value(self, value: object, *, text_copies: int = 1) -> None:
        """Reserve a clone/serialization before constructing it, including repeats."""
        if isinstance(value, str):
            self.reserve(64 + 4 * len(value) * text_copies)
        elif isinstance(value, bytes):
            self.reserve(64 + len(value) * text_copies)
        elif isinstance(value, Mapping):
            self.container(len(value))
            for key, item in value.items():
                self.value(key, text_copies=text_copies)
                self.value(item, text_copies=text_copies)
        elif isinstance(value, (list, tuple, sqlite3.Row)):
            self.container(len(value))
            for item in value:
                self.value(item, text_copies=text_copies)
        elif is_dataclass(value) and not isinstance(value, type):
            self.container(len(fields(value)))
            for field in fields(value):
                self.value(getattr(value, field.name), text_copies=text_copies)
        else:
            self.reserve(64)

    def json(self, raw: str | bytes) -> None:
        """Reserve decode expansion before the native JSON codec runs."""
        self.reserve(64 + 4 * len(raw))
        quoted = escaped = False
        for item in raw:
            code = ord(item) if isinstance(item, str) else item
            if quoted:
                if escaped:
                    escaped = False
                elif code == 92:
                    escaped = True
                elif code == 34:
                    quoted = False
            elif code == 34:
                quoted = True
                self.reserve(64)
            elif code in (91, 93, 123, 125, 44, 58):
                self.reserve(256)

    def base64(self, raw: object) -> None:
        size = len(raw)
        # bytes conversion, encoded bytes, decoded ASCII and the enclosing value.
        self.reserve(192 + size + 8 * ((size + 2) // 3))
