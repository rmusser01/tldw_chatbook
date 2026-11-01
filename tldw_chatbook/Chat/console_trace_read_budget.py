"""Private, nonrenewable materialization allowance for retained trace reads."""

import sqlite3
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from weakref import WeakKeyDictionary

_mobile_pairs = WeakKeyDictionary()


from tldw_chatbook.Backup_Recovery.limits import _TraceReadLimit


class _TraceReadBudget:
    """One allowance shared by both captured partitions and nested consumers.

    Reservations cover representation expansion, not a Python heap measurement.
    Strings use the four-byte Unicode upper bound. JSON container/token overhead
    is reserved by scanning its bounded raw representation before decoding; no
    decoded tree or second serializer is used to decide admission.
    """

    __slots__ = ("__weakref__", "_parent", "_remaining")

    def __init__(self):
        self._remaining = 16 * 1024 * 1024
        self._parent = None

    def require_capacity(self, amount: int) -> None:
        """Check every applicable bound without consuming either allowance."""
        parent = self._parent
        if parent is not None and not _mobile_budget_pair(parent, self):
            raise ValueError("mobile_budget_pair_changed")
        if (
            amount < 0
            or amount > self._remaining
            or (parent is not None and amount > parent._remaining)
        ):
            raise _TraceReadLimit("mobile_trace_input_limit")

    def reserve(self, amount: int) -> None:
        self.require_capacity(amount)
        if self._parent is not None:
            self._parent._remaining -= amount
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


def _mobile_cutover_budget():
    """Issue one fixed allowance for a complete ordinary cutover observation."""
    budget = _TraceReadBudget()
    budget._remaining = 64 * 1024**2
    budget.reserve(512)
    return budget


def _mobile_operation_budgets():
    """Issue the fixed public envelope and original native proof sub-bound.

    No caller chooses a capacity or parent. These accounting objects confer no
    session, path, owner, or captured-input authority.
    """
    aggregate = _TraceReadBudget()
    aggregate._remaining = 4 * 1024**3
    aggregate.reserve(512)
    native = _TraceReadBudget()
    native._parent = aggregate
    _mobile_pairs[native] = aggregate
    return aggregate, native


def _mobile_budget_pair(aggregate, native):
    return (
        type(aggregate) is _TraceReadBudget
        and type(native) is _TraceReadBudget
        and aggregate._parent is None
        and native._parent is aggregate
        and _mobile_pairs.get(native) is aggregate
    )


def _require_mobile_budget_pair(pair):
    if type(pair) is not tuple or len(pair) != 2 or not _mobile_budget_pair(*pair):
        raise ValueError("mobile_budget_pair_changed")
    return pair[0]
