"""One guard for handing a value to a Textual ``Select``.

TASK-32533. ``Select._validate_value`` raises ``InvalidSelectValueError`` for any
value outside the widget's options -- at mount for a composed ``value=``, and on
every later assignment. Before this task an unhandled raise from a widget
handler exited the whole app (critique #3's P0), and the app-level keep-alive
that replaced the exit only downgrades it: the guard is what actually prevents
the failure.

Every Chatbook site that hands a Select a value it did not itself pick out of
that Select's options routes through here, so the next one is a one-line change
rather than a fourth spelling.
"""

from __future__ import annotations

from typing import Any, Iterable

from textual.widgets import Select

__all__ = ["select_value_or_blank", "assign_select_value"]


def _offers(options: Iterable[tuple[Any, Any]], value: Any) -> bool:
    # Textual's own check is `value not in self._legal_values`; `Select.NULL`
    # has no `__eq__`, so this is the same identity/equality semantics.
    return any(option == value for _label, option in options)


def select_value_or_blank(options: Iterable[tuple[Any, Any]], value: Any) -> Any:
    """Return ``value`` if ``options`` offers it, else ``Select.NULL``.

    For the compose-time ``Select(options, value=...)`` form, where no widget
    exists yet. Only legal on a select that allows a blank selection (Textual's
    default).
    """
    options = list(options)
    return value if _offers(options, value) else Select.NULL


def assign_select_value(select: Select, value: Any) -> bool:
    """Assign ``value`` to ``select``, or its blank row, or leave it alone.

    Returns ``False`` when neither ``value`` nor ``Select.NULL`` is on offer --
    a select built with ``allow_blank=False`` whose options no longer contain
    the value. The caller decides what a refused assignment means; a stale
    selection is never worse than the exception it replaces.
    """
    for candidate in (value, Select.NULL):
        if _offers(select._options, candidate):
            select.value = candidate
            return True
    return False
