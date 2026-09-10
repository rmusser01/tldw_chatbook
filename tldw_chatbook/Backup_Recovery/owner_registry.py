"""Installed recovery owner registration, independent of service composition.

Registration is process-local code authority, never read from TOML or archives.
Factories install declarations only; durable payload adapters qualify separately.
"""

from __future__ import annotations

from threading import RLock

from .models import OwnerAdapter

_adapters: dict[str, OwnerAdapter] = {}
_lock = RLock()


def register(adapter: OwnerAdapter) -> None:
    """Register one installed logical owner; reject ambiguous duplicate authority.

    Shared physical stores use different logical IDs and an explicit shared_group
    in discovered items. They never require replacing a registered adapter.
    """
    if not isinstance(adapter.owner_id, str) or not adapter.owner_id:
        raise ValueError("invalid_owner")
    if not isinstance(adapter.activation_required, bool) or not all(
        callable(getattr(adapter, method, None))
        for method in ("discover", "capture", "validate", "relocate", "schema_policy")
    ):
        raise ValueError("invalid_owner")
    with _lock:
        if adapter.owner_id in _adapters:
            raise ValueError("duplicate_owner")
        _adapters[adapter.owner_id] = adapter


def registered() -> tuple[OwnerAdapter, ...]:
    """Return an immutable, deterministically ordered installed-adapter snapshot."""
    with _lock:
        return tuple(_adapters[key] for key in sorted(_adapters))


def install_adapters() -> tuple[OwnerAdapter, ...]:
    """Install the shipped inert declarations before preview or capture."""
    from importlib import import_module

    factories = (
        ("DB.recovery_core", "core_adapters"),
        ("DB.recovery_operations", "recovery_adapters"),
        ("Backup_Recovery.config_adapter", "recovery_adapters"),
        ("Backup_Recovery.recovered_media", "recovery_adapters"),
        ("Backup_Recovery.rag_inventory", "recovery_adapters"),
        *(
            (name + ".recovery", "recovery_adapters")
            for name in (
                "Agents",
                "Evals",
                "Kanban_Interop",
                "MCP",
                "Model_Artifacts",
                "Notes",
                "Notifications",
                "Persona_Visual",
                "Research_Interop",
                "Scheduling",
                "Skills_Interop",
                "Study_Interop",
                "Subscriptions",
                "Sync_Interop",
                "TTS",
                "Widgets.Tamagotchi",
                "Workspaces",
                "Writing_Interop",
                "runtime_policy",
            )
        ),
    )
    declarations = tuple(
        adapter
        for module, factory in factories
        for adapter in getattr(import_module("tldw_chatbook." + module), factory)()
    )
    with _lock:
        for adapter in declarations:
            existing = _adapters.get(adapter.owner_id)
            if existing is not None and type(existing) is not type(adapter):
                raise ValueError("conflicting_installed_owner")
        for adapter in declarations:
            if adapter.owner_id not in _adapters:
                register(adapter)
        return registered()
