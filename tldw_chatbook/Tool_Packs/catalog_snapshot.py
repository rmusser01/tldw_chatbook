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
    complete: bool

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
    namespaces: tuple[tuple[PermissionAuthority, str], ...]
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
            if getattr(adapter, "complete", None) is not True:
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
                        "namespaces": [list(item) for item in sorted(("builtin" if name == "agent:builtin" else "mcp", name) for name in namespaces)],
                        "excluded_counts": dict(excluded_items),
                    },
                    operation="export",
                )
            ).hexdigest()
        except ToolPackError:
            raise ToolPackError("export", "inventory_incomplete") from None
        snapshot_namespaces = tuple(
            sorted(
                ("builtin" if name == "agent:builtin" else "mcp", name)
                for name in namespaces
            )
        )
        return PermissionInventorySnapshot(
            tuple(inventory), snapshot_namespaces, excluded_items, digest
        )

    def capture_for_export(self) -> PermissionInventorySnapshot:
        """Reject generic assembly at the production export boundary."""
        raise ToolPackError("export", "inventory_incomplete")

    @classmethod
    def v1(
        cls,
        *,
        fallback_root: object,
        builtin_mcp_inventory: Callable[[], object],
        external_catalog: Callable[[], object],
        excluded_counts: Callable[[], dict[str, int]] | None = None,
    ) -> "_V1PermissionInventoryRegistry":
        return _V1PermissionInventoryRegistry(
            fallback_root=fallback_root,
            builtin_mcp_inventory=builtin_mcp_inventory,
            external_catalog=external_catalog,
            excluded_counts=excluded_counts,
        )


@dataclass(frozen=True, slots=True)
class _SnapshotAdapter:
    namespace: str
    tools: tuple[HubTool, ...]
    complete: bool = True

    def snapshot(self) -> tuple[HubTool, ...]:
        return self.tools


def _strict_raw_tools(raw: object, *, namespace: str, label: str) -> tuple[HubTool, ...]:
    if type(raw) is not list:
        raise ToolPackError("export", "inventory_incomplete")
    names: set[str] = set()
    tools: list[HubTool] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ToolPackError("export", "inventory_incomplete")
        name = item.get("name")
        description = item.get("description")
        schema = item.get("inputSchema")
        if (
            type(name) is not str
            or not name
            or name != name.strip()
            or type(description) is not str
            or not description
            or "inputSchema" not in item
            or type(schema) is not dict
            or name.casefold() in names
        ):
            raise ToolPackError("export", "inventory_incomplete")
        names.add(name.casefold())
        tools.append(
            HubTool(namespace, label, "builtin" if namespace.startswith("builtin:") else "local", name, description, schema, (), False, True)
        )
    return tuple(tools)


class _V1PermissionInventoryRegistry(PermissionInventoryRegistry):
    """The only production assembly for all classified V1 authorities."""

    def __init__(
        self,
        *,
        fallback_root: object,
        builtin_mcp_inventory: Callable[[], object],
        external_catalog: Callable[[], object],
        excluded_counts: Callable[[], dict[str, int]] | None,
    ) -> None:
        self._fallback_root = fallback_root
        self._builtin_mcp_inventory = builtin_mcp_inventory
        self._external_catalog = external_catalog
        self._v1_excluded_counts = excluded_counts

    def capture_for_export(self) -> PermissionInventorySnapshot:
        try:
            from tldw_chatbook.Agents.builtin_tool_gate import tool_ref
            from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider
            from tldw_chatbook.Agents.raw_shell_tool_provider import RawShellToolProvider
            from tldw_chatbook.Agents.session_todo_store import SessionTodoStore
            from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider, build_gateable_tool, gateable_builtin_tools
            from tldw_chatbook.Agents.virtual_cli_provider import VirtualCliProvider

            builtin_tools: dict[str, HubTool] = {}
            provider = BuiltinToolProvider()
            for entry in provider.list_catalog():
                tool = provider.tool_for(entry.name)
                if tool is not None:
                    ref = tool_ref(tool)
                    builtin_tools[tool.name] = HubTool(ref.server_key, "Built-in tools", "builtin", tool.name, ref.description, ref.input_schema, ref.tags, False, True)
            for entry in gateable_builtin_tools():
                tool = build_gateable_tool(entry)
                ref = tool_ref(tool)
                builtin_tools.setdefault(tool.name, HubTool(ref.server_key, "Built-in tools", "builtin", tool.name, ref.description, ref.input_schema, ref.tags, False, True))

            builtin_raw = self._builtin_mcp_inventory()
            if not isinstance(builtin_raw, Mapping):
                raise ToolPackError("export", "inventory_incomplete")
            builtin_mcp = _strict_raw_tools(builtin_raw.get("tools"), namespace="builtin:tldw_chatbook", label="tldw_chatbook")
            local = tuple(LocalToolProvider(workspace_root=self._fallback_root, admitted_roots=None, todo_store=SessionTodoStore()).hub_tools()) + (RawShellToolProvider.hub_tool(),)
            virtual = tuple(VirtualCliProvider(workspace_root=self._fallback_root, admitted_roots=None).hub_tools())
            catalog = self._external_catalog()
            if type(catalog) is not list:
                raise ToolPackError("export", "inventory_incomplete")
            external: list[_SnapshotAdapter] = []
            profile_ids: set[str] = set()
            for record in catalog:
                if not isinstance(record, Mapping):
                    raise ToolPackError("export", "inventory_incomplete")
                profile_id = record.get("profile_id")
                snapshot = record.get("discovery_snapshot")
                if (
                    type(profile_id) is not str
                    or not profile_id
                    or profile_id != profile_id.strip()
                    or any(char.isspace() or char == ":" for char in profile_id)
                    or profile_id.casefold() in profile_ids
                    or not isinstance(snapshot, Mapping)
                ):
                    raise ToolPackError("export", "inventory_incomplete")
                profile_ids.add(profile_id.casefold())
                external.append(_SnapshotAdapter(f"local:{profile_id}", _strict_raw_tools(snapshot.get("tools"), namespace=f"local:{profile_id}", label=profile_id)))
            adapters = (
                _SnapshotAdapter("agent:builtin", tuple(builtin_tools.values())),
                _SnapshotAdapter("builtin:tldw_chatbook", builtin_mcp),
                _SnapshotAdapter("local:__local__", local),
                _SnapshotAdapter("local:__virtual_cli__", virtual),
                *external,
            )
            registry = PermissionInventoryRegistry(
                current_permission_namespaces=lambda: {adapter.namespace for adapter in adapters},
                excluded_counts=self._v1_excluded_counts,
            )
            for adapter in adapters:
                registry.register(adapter)
            return registry.capture()
        except ToolPackError:
            raise
        except Exception:
            raise ToolPackError("export", "inventory_incomplete") from None


__all__ = [
    "PermissionInventoryAdapter",
    "PermissionInventoryRegistry",
    "PermissionInventorySnapshot",
    "PermissionInventoryTool",
    "NONPORTABLE_CATEGORIES",
    "thaw_hub_tool",
]
