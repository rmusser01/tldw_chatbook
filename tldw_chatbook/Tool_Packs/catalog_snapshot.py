"""Immutable, complete inventories for portable Tool Pack export."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
from types import MappingProxyType
from typing import Literal, Protocol

from tldw_chatbook.MCP.hub_tool_catalog import HubTool
from tldw_chatbook.Tool_Packs.contracts import (
    PortableFallback,
    ToolPackError,
    canonical_json_bytes,
    portable_contract_sha256,
)


PermissionAuthority = Literal["mcp", "builtin"]
NONPORTABLE_CATEGORIES = (
    "display_only_server_source",
    "library_capability",
    "managed_skill_approval",
    "runtime_orchestration",
    "skills",
)


def _freeze_json(value: object) -> object:
    """Copy and recursively freeze JSON-shaped tool schemas for a review."""
    if type(value) is dict:
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if type(value) is list:
        return tuple(_freeze_json(item) for item in value)
    return value


def thaw_hub_tool(tool: HubTool) -> HubTool:
    """Return a resolver-compatible detached copy of an inventory tool."""
    def thaw(value: object) -> object:
        if isinstance(value, Mapping):
            return {key: thaw(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return [thaw(item) for item in value]
        return value

    schema = thaw(tool.input_schema)
    return HubTool(
        server_key=tool.server_key,
        server_label=tool.server_label,
        source=tool.source,
        name=tool.name,
        description=tool.description,
        input_schema=schema,
        tags=tuple(tool.tags),
        stale=tool.stale,
        executable=tool.executable,
    )


def _frozen_hub_tool(tool: HubTool) -> HubTool:
    return HubTool(
        server_key=tool.server_key,
        server_label=tool.server_label,
        source=tool.source,
        name=tool.name,
        description=tool.description,
        input_schema=_freeze_json(tool.input_schema),  # type: ignore[arg-type]
        tags=tuple(tool.tags),
        stale=tool.stale,
        executable=tool.executable,
    )


class PermissionInventoryAdapter(Protocol):
    """One code-owned, complete definition source for a governed namespace."""

    namespace: str

    def snapshot(self) -> tuple[HubTool, ...]:
        """Return every currently defined tool for ``namespace``."""


@dataclass(frozen=True, slots=True)
class PermissionInventoryTool:
    """A portable-contract fingerprinted permission authority."""

    authority: PermissionAuthority
    tool: HubTool
    contract_sha256: str

    @property
    def identity(self) -> tuple[str, str, str]:
        return (self.authority, self.tool.server_key, self.tool.name)


@dataclass(frozen=True, slots=True)
class PermissionInventorySnapshot:
    """The deterministic complete inventory used by exactly one export capture."""

    tools: tuple[PermissionInventoryTool, ...]
    excluded_counts: tuple[tuple[str, int], ...]
    digest: str


class PermissionInventoryRegistry:
    """Classifies every permission-governed namespace before exporting it."""

    def __init__(
        self,
        *,
        current_permission_namespaces: Callable[[], set[str]],
        excluded_counts: Callable[[], dict[str, int]] | None = None,
    ) -> None:
        self._current_permission_namespaces = current_permission_namespaces
        self._excluded_counts = excluded_counts or (lambda: {})
        self._adapters: dict[str, PermissionInventoryAdapter] = {}

    def register(self, adapter: PermissionInventoryAdapter) -> None:
        namespace = getattr(adapter, "namespace", None)
        if type(namespace) is not str or not namespace or namespace in self._adapters:
            raise ToolPackError("export", "inventory_incomplete")
        self._adapters[namespace] = adapter

    def capture(self) -> PermissionInventorySnapshot:
        """Capture all classified authorities or fail closed before any export."""
        try:
            namespaces = self._current_permission_namespaces()
            raw_excluded = self._excluded_counts()
        except Exception:
            raise ToolPackError("export", "inventory_incomplete") from None
        if type(namespaces) is not set or any(type(item) is not str for item in namespaces):
            raise ToolPackError("export", "inventory_incomplete")
        if len({item.casefold() for item in namespaces}) != len(namespaces):
            raise ToolPackError("export", "inventory_incomplete")
        if namespaces != set(self._adapters):
            raise ToolPackError("export", "inventory_incomplete")
        if type(raw_excluded) is not dict or any(
            type(key) is not str
            or type(value) is not int
            or value < 0
            or key not in NONPORTABLE_CATEGORIES
            for key, value in raw_excluded.items()
        ):
            raise ToolPackError("export", "inventory_incomplete")
        excluded = dict.fromkeys(NONPORTABLE_CATEGORIES, 0)
        excluded.update(raw_excluded)

        inventory: list[PermissionInventoryTool] = []
        for namespace in sorted(namespaces):
            adapter = self._adapters[namespace]
            if getattr(adapter, "complete", True) is not True:
                raise ToolPackError("export", "inventory_incomplete")
            try:
                tools = adapter.snapshot()
            except Exception:
                raise ToolPackError("export", "inventory_incomplete") from None
            if type(tools) is not tuple:
                raise ToolPackError("export", "inventory_incomplete")
            authority: PermissionAuthority = (
                "builtin" if namespace == "agent:builtin" else "mcp"
            )
            try:
                if namespace == "*":
                    raise ToolPackError("export", "inventory_incomplete")
                PortableFallback(authority, namespace, "ask")
            except ToolPackError:
                raise ToolPackError("export", "inventory_incomplete") from None
            for tool in tools:
                if type(tool) is not HubTool or tool.server_key != namespace:
                    raise ToolPackError("export", "inventory_incomplete")
                try:
                    frozen_tool = _frozen_hub_tool(tool)
                    inventory.append(
                        PermissionInventoryTool(
                            authority,
                            frozen_tool,
                            portable_contract_sha256(tool, operation="export"),
                        )
                    )
                except Exception:
                    raise ToolPackError("export", "inventory_incomplete") from None

        inventory.sort(key=lambda item: item.identity)
        identities = [item.identity for item in inventory]
        folded = [tuple(part.casefold() for part in identity) for identity in identities]
        if len(set(identities)) != len(identities) or len(set(folded)) != len(folded):
            raise ToolPackError("export", "inventory_incomplete")
        excluded_items = tuple(sorted(excluded.items()))
        try:
            digest = hashlib.sha256(
                canonical_json_bytes(
                    {
                        "tools": [
                            {
                                "authority": item.authority,
                                "server_key": item.tool.server_key,
                                "tool_name": item.tool.name,
                                "contract_sha256": item.contract_sha256,
                            }
                            for item in inventory
                        ],
                        "excluded_counts": dict(excluded_items),
                    },
                    operation="export",
                )
            ).hexdigest()
        except ToolPackError:
            raise ToolPackError("export", "inventory_incomplete") from None
        return PermissionInventorySnapshot(tuple(inventory), excluded_items, digest)


__all__ = [
    "PermissionInventoryAdapter",
    "PermissionInventoryRegistry",
    "PermissionInventorySnapshot",
    "PermissionInventoryTool",
    "NONPORTABLE_CATEGORIES",
    "thaw_hub_tool",
]
