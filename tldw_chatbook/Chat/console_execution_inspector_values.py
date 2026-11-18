"""Finite, display-only values from the original Inspector scope summaries."""

from dataclasses import dataclass

from .console_display_state import ConsoleDisplayRow, ConsoleInspectorAction
from .console_execution_projection_values import _boolean, _optional, _record, _text, _tuple


@dataclass(frozen=True, slots=True)
class ConsoleInspectorScopes:
    conversation_id: str | None
    dictionary_rows: tuple[ConsoleDisplayRow, ...]
    dictionary_actions: tuple[ConsoleInspectorAction, ...]
    world_book_rows: tuple[ConsoleDisplayRow, ...]
    world_book_actions: tuple[ConsoleInspectorAction, ...]


def _row(value, decode):
    return _record(value, ConsoleDisplayRow, {
        "label": _text, "value": _text, "status": _text, "recovery": _text, "code": _text,
    }, decode)


def _action(value, decode):
    return _record(value, ConsoleInspectorAction, {
        "widget_id": _text, "label": _text, "enabled": _boolean,
        "disabled_reason": _text, "classes": _text,
    }, decode)


def inspector_scopes(value, decode):
    return _record(value, ConsoleInspectorScopes, {
        "conversation_id": _optional(_text),
        "dictionary_rows": _tuple(_row), "dictionary_actions": _tuple(_action),
        "world_book_rows": _tuple(_row), "world_book_actions": _tuple(_action),
    }, decode)


def build_inspector_scopes(conversation_id, dictionaries, world_books) -> ConsoleInspectorScopes:
    """Apply the shared original renderers only to actual producer results."""
    from ..UI.Console_Modules import inspector_scope_display as display

    return ConsoleInspectorScopes(
        conversation_id,
        display.dictionary_rows(conversation_id, None, dictionaries),
        display.dictionary_actions(conversation_id, dictionaries),
        display.world_book_rows(conversation_id, world_books),
        display.world_book_actions(conversation_id, world_books),
    )
