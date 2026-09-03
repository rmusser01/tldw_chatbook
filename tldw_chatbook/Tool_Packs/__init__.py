"""Portable Tool policy pack authority primitives."""

from importlib import import_module

from tldw_chatbook.Tool_Packs.binding import (
    ProfileMutationError,
    ProfileMutationResult,
    ToolProfileLifecycleCoordinator,
    profile_policy_digest,
)

__all__ = [
    "ProfileMutationError",
    "ProfileMutationResult",
    "ToolPackReceiptReconciliationResult",
    "ToolPackService",
    "ToolProfileLifecycleCoordinator",
    "ToolProfileListing",
    "ToolProfilePresentation",
    "profile_policy_digest",
]

_LAZY_SERVICE_EXPORTS = frozenset(
    {
        "ToolPackReceiptReconciliationResult",
        "ToolPackService",
        "ToolProfileListing",
        "ToolProfilePresentation",
    }
)


def __getattr__(name: str):
    """Load presentation orchestration only when a caller asks for it."""
    if name not in _LAZY_SERVICE_EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module("tldw_chatbook.Tool_Packs.service"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_SERVICE_EXPORTS)
