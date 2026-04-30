from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "CONTRACT_ID": (".server_parity_contracts", "CONTRACT_ID"),
    "CONTRACT_VERSION": (".server_parity_contracts", "CONTRACT_VERSION"),
    "ActiveServerStatusContract": (".server_parity_contracts", "ActiveServerStatusContract"),
    "FutureSyncStatusContract": (".server_parity_contracts", "FutureSyncStatusContract"),
    "NotificationFeedItemContract": (".server_parity_contracts", "NotificationFeedItemContract"),
    "SourceSelectorStateContract": (".server_parity_contracts", "SourceSelectorStateContract"),
    "UnsupportedActionPresentationContract": (
        ".server_parity_contracts",
        "UnsupportedActionPresentationContract",
    ),
    "WorkspaceIsolationContract": (".server_parity_contracts", "WorkspaceIsolationContract"),
    "build_server_parity_fixture_payloads": (
        ".server_parity_contracts",
        "build_server_parity_fixture_payloads",
    ),
    "build_server_parity_handoff_packet": (
        ".server_parity_contracts",
        "build_server_parity_handoff_packet",
    ),
    "notification_feed_item_from_payload": (
        ".server_parity_contracts",
        "notification_feed_item_from_payload",
    ),
    "sync_status_contract": (".server_parity_contracts", "sync_status_contract"),
    "workspace_isolation_contract": (".server_parity_contracts", "workspace_isolation_contract"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
