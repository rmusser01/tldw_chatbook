from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from tldw_chatbook.Agents.builtin_tool_gate import tool_ref
from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider
from tldw_chatbook.Agents.raw_shell_tool_provider import RawShellToolProvider
from tldw_chatbook.Agents.session_todo_store import SessionTodoStore
from tldw_chatbook.Agents.tool_catalog import (
    BuiltinToolProvider,
    build_gateable_tool,
    gateable_builtin_tools,
)
from tldw_chatbook.Agents.virtual_cli_provider import VirtualCliProvider
from tldw_chatbook.MCP.hub_tool_catalog import (
    HubTool,
    builtin_tools_from_inventory,
    local_tools_from_record,
)
from tldw_chatbook.Tool_Packs.catalog_snapshot import (
    PermissionInventoryAdapter,
    PermissionInventoryRegistry,
    thaw_hub_tool,
)
from tldw_chatbook.Tool_Packs.contracts import (
    ToolPackError,
    canonical_json_bytes,
    portable_contract_sha256,
)


class _BuiltinAdapter(PermissionInventoryAdapter):
    namespace = "agent:builtin"

    def snapshot(self) -> tuple[HubTool, ...]:
        return ()


class _ToolsAdapter(PermissionInventoryAdapter):
    def __init__(
        self,
        namespace: str,
        tools: tuple[HubTool, ...],
        *,
        complete: bool = True,
        snapshot: Callable[[], tuple[HubTool, ...]] | None = None,
    ) -> None:
        self.namespace = namespace
        self.complete = complete
        self._tools = tools
        self._snapshot = snapshot

    def snapshot(self) -> tuple[HubTool, ...]:
        return self._snapshot() if self._snapshot is not None else self._tools


def test_unclassified_permission_namespace_blocks_export() -> None:
    """Adding a governed namespace cannot silently make exports incomplete."""
    registry = PermissionInventoryRegistry(
        current_permission_namespaces=lambda: {"agent:builtin", "local:new"}
    )
    registry.register(_BuiltinAdapter())

    with pytest.raises(ToolPackError, match=r"^tool_pack\.export\.inventory_incomplete$"):
        registry.capture()


def test_captured_inventory_does_not_retain_a_mutable_provider_schema() -> None:
    """A provider changing its schema after capture cannot rewrite a review."""
    source = HubTool(
        server_key="local:docs",
        server_label="Docs",
        source="local",
        name="search",
        description="Search documents.",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        tags=(),
        stale=False,
        executable=True,
    )
    registry = PermissionInventoryRegistry(
        current_permission_namespaces=lambda: {"local:docs"}
    )
    registry.register(_ToolsAdapter("local:docs", (source,)))

    captured = registry.capture()
    source.input_schema["properties"]["q"]["type"] = "number"  # type: ignore[index]

    assert captured.tools[0].tool.input_schema == {
        "type": "object",
        "properties": {"q": {"type": "string"}},
    }
    with pytest.raises(TypeError):
        captured.tools[0].tool.input_schema["type"] = "array"  # type: ignore[index]
    assert portable_contract_sha256(thaw_hub_tool(captured.tools[0].tool)) == (
        captured.tools[0].contract_sha256
    )


def test_review_always_reports_every_named_nonportable_category() -> None:
    """A zero count remains visible when a nonportable category is absent."""
    registry = PermissionInventoryRegistry(
        current_permission_namespaces=lambda: {"agent:builtin"}
    )
    registry.register(_BuiltinAdapter())

    captured = registry.capture()

    assert dict(captured.excluded_counts) == {
        "display_only_server_source": 0,
        "library_capability": 0,
        "managed_skill_approval": 0,
        "runtime_orchestration": 0,
        "skills": 0,
    }


@pytest.mark.parametrize("same_case", [True, False])
def test_inventory_rejects_exact_or_casefolded_duplicate_identities(
    same_case: bool,
) -> None:
    first = _hub("search")
    second = _hub("search" if same_case else "SEARCH")
    registry = PermissionInventoryRegistry(
        current_permission_namespaces=lambda: {"local:docs"}
    )
    registry.register(_ToolsAdapter("local:docs", (first, second)))

    with pytest.raises(ToolPackError, match=r"inventory_incomplete$"):
        registry.capture()


def test_incomplete_or_raising_inventory_adapter_fails_closed() -> None:
    incomplete = PermissionInventoryRegistry(
        current_permission_namespaces=lambda: {"local:docs"}
    )
    incomplete.register(_ToolsAdapter("local:docs", (), complete=False))
    raising = PermissionInventoryRegistry(
        current_permission_namespaces=lambda: {"local:docs"}
    )
    raising.register(
        _ToolsAdapter(
            "local:docs",
            (),
            snapshot=lambda: (_ for _ in ()).throw(RuntimeError("private path")),
        )
    )

    for registry in (incomplete, raising):
        with pytest.raises(ToolPackError, match=r"inventory_incomplete$"):
            registry.capture()


def test_invalid_definition_and_casefolded_empty_namespaces_fail_closed() -> None:
    invalid_definition = PermissionInventoryRegistry(
        current_permission_namespaces=lambda: {"local:docs"}
    )
    invalid_definition.register(
        _ToolsAdapter("local:docs", (replace(_hub("search"), name=""),))
    )
    folded_namespaces = PermissionInventoryRegistry(
        current_permission_namespaces=lambda: {"local:docs", "local:DOCS"}
    )
    folded_namespaces.register(_ToolsAdapter("local:docs", ()))
    folded_namespaces.register(_ToolsAdapter("local:DOCS", ()))

    for registry in (invalid_definition, folded_namespaces):
        with pytest.raises(
            ToolPackError,
            match=r"^tool_pack\.export\.inventory_incomplete$",
        ):
            registry.capture()


def _hub(name: str, *, server_key: str = "local:docs") -> HubTool:
    return HubTool(
        server_key=server_key,
        server_label=server_key,
        source="local",
        name=name,
        description=f"{name} description",
        input_schema={"type": "object", "properties": {}},
        tags=(),
        stale=False,
        executable=True,
    )


def _all_builtin_hubs() -> tuple[HubTool, ...]:
    provider = BuiltinToolProvider()
    tools = {
        entry.name: provider.tool_for(entry.name)
        for entry in provider.list_catalog()
    }
    for entry in gateable_builtin_tools():
        tools.setdefault(entry.tool_name, build_gateable_tool(entry))
    hubs: list[HubTool] = []
    for name, tool in tools.items():
        assert tool is not None
        ref = tool_ref(tool)
        hubs.append(
            HubTool(
                server_key=ref.server_key,
                server_label="Built-in tools",
                source="builtin",
                name=name,
                description=ref.description,
                input_schema=ref.input_schema,
                tags=ref.tags,
                stale=False,
                executable=True,
            )
        )
    return tuple(hubs)


def test_concrete_v1_provider_inventory_is_complete_unbound_and_path_private(
    tmp_path: Path,
) -> None:
    sentinel_root = tmp_path / "private-workspace-sentinel"
    sentinel_root.mkdir()
    builtin_mcp = tuple(
        builtin_tools_from_inventory(
            {
                "tools": [
                    {
                        "name": "search_media",
                        "description": "Search local media.",
                        "inputSchema": {"type": "object", "properties": {}},
                    }
                ]
            }
        )
    )
    external = tuple(
        local_tools_from_record(
            {
                "profile_id": "docs",
                "is_connected": False,
                "discovery_snapshot": {
                    "tools": [
                        {
                            "name": "search",
                            "description": "Search cached docs.",
                            "inputSchema": {"type": "object", "properties": {}},
                        }
                    ]
                },
            }
        )
    )
    local_provider = LocalToolProvider(
        workspace_root=sentinel_root,
        admitted_roots=None,
        todo_store=SessionTodoStore(),
    )
    local = tuple(local_provider.hub_tools()) + (RawShellToolProvider.hub_tool(),)
    virtual = tuple(
        VirtualCliProvider(
            workspace_root=sentinel_root,
            admitted_roots=None,
        ).hub_tools()
    )
    tools_by_namespace = {
        "agent:builtin": _all_builtin_hubs(),
        "builtin:tldw_chatbook": builtin_mcp,
        "local:__local__": local,
        "local:__virtual_cli__": virtual,
        "local:docs": external,
    }
    registry = PermissionInventoryRegistry(
        current_permission_namespaces=lambda: set(tools_by_namespace),
        excluded_counts=lambda: {
            "display_only_server_source": 3,
            "library_capability": 2,
            "managed_skill_approval": 1,
            "runtime_orchestration": 4,
            "skills": 5,
        },
    )
    for namespace, tools in tools_by_namespace.items():
        registry.register(_ToolsAdapter(namespace, tools))

    snapshot = registry.capture()
    identities = {item.identity for item in snapshot.tools}
    assert ("builtin", "agent:builtin", "calculator") in identities
    assert ("mcp", "builtin:tldw_chatbook", "search_media") in identities
    assert ("mcp", "local:__local__", "shell_exec") in identities
    assert ("mcp", "local:__local__", "todo_create") in identities
    assert ("mcp", "local:__virtual_cli__", "ls") in identities
    assert ("mcp", "local:docs", "search") in identities
    assert dict(snapshot.excluded_counts)["runtime_orchestration"] == 4

    schema_bytes = canonical_json_bytes(
        [
            thaw_hub_tool(item.tool).input_schema
            for item in snapshot.tools
        ],
        operation="export",
    )
    assert str(sentinel_root).encode() not in schema_bytes
    assert b"root_alias" not in schema_bytes
