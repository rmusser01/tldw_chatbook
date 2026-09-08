"""Public service exports, resolved lazily for dependency-light recovery."""

from importlib import import_module

_EXPORTS = {
    "LocalWritingService": "local_writing_service",
    "ServerWritingService": "server_writing_service",
    "WritingBackend": "writing_scope_service",
    "WritingScopeService": "writing_scope_service",
}
__all__ = [
    "LocalWritingService",
    "ServerWritingService",
    "WritingBackend",
    "WritingScopeService",
]


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("." + module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
