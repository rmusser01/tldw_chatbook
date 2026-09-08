"""Compatibility-preserving lazy public exports for recovery isolation."""

from importlib import import_module

_EXPORTS = {
    "KanbanBackend": (".kanban_scope_service", "KanbanBackend"),
    "KanbanScopeService": (".kanban_scope_service", "KanbanScopeService"),
    "LocalKanbanService": (".local_kanban_service", "LocalKanbanService"),
    "ServerKanbanService": (".server_kanban_service", "ServerKanbanService"),
}

__all__ = [
    "KanbanBackend",
    "KanbanScopeService",
    "LocalKanbanService",
    "ServerKanbanService",
]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module, __name__), symbol)
    globals()[name] = value
    return value
