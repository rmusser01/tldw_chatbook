from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from tldw_chatbook.MCP.hub_tool_catalog import HubTool
from tldw_chatbook.Tool_Packs.catalog_snapshot import (
    PermissionInventoryAdapter,
    PermissionInventoryRegistry,
    capture_v1_inventory,
    thaw_hub_tool,
)
from tldw_chatbook.Tool_Packs.contracts import (
    ToolPackError,
    canonical_json_bytes,
    portable_contract_sha256,
)


class _BuiltinAdapter(PermissionInventoryAdapter):
    namespace = "agent:builtin"
    complete = True

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


class _LocalControlService:
    def __init__(self, inventory: object, external_servers: object) -> None:
        self._inventory = inventory
        self._external_servers = external_servers

    def get_inventory(self) -> object:
        return self._inventory

    def get_external_servers(self) -> object:
        return self._external_servers


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


def test_empty_adapter_without_explicit_completeness_fails_closed() -> None:
    class ImplicitlyEmpty:
        namespace = "local:docs"

        def snapshot(self) -> tuple[HubTool, ...]:
            return ()

    registry = PermissionInventoryRegistry(
        current_permission_namespaces=lambda: {"local:docs"}
    )
    registry.register(ImplicitlyEmpty())  # type: ignore[arg-type]

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


def test_concrete_v1_provider_inventory_is_complete_unbound_and_path_private(
    tmp_path: Path,
) -> None:
    sentinel_root = tmp_path / "private-workspace-sentinel"
    sentinel_root.mkdir()
    builtin_inventory = {
        "tools": [
            {
                "name": "search_media",
                "description": "Search local media.",
                "inputSchema": {"type": "object", "properties": {}},
            }
        ]
    }
    external_catalog = [
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
        },
        {
            "profile_id": "empty",
            "is_connected": True,
            "discovery_snapshot": {"tools": []},
        },
    ]
    local_service = _LocalControlService(builtin_inventory, external_catalog)
    registry = PermissionInventoryRegistry.v1(
        local_service,
        fallback_root=sentinel_root,
        excluded_counts=lambda: {
            "display_only_server_source": 3,
            "library_capability": 2,
            "managed_skill_approval": 1,
            "runtime_orchestration": 4,
            "skills": 5,
        },
    )
    snapshot = capture_v1_inventory(registry)
    identities = {item.identity for item in snapshot.tools}
    assert ("builtin", "agent:builtin", "calculator") in identities
    assert ("mcp", "builtin:tldw_chatbook", "search_media") in identities
    assert ("mcp", "local:__local__", "shell_exec") in identities
    assert ("mcp", "local:__local__", "todo_create") in identities
    assert ("mcp", "local:__virtual_cli__", "ls") in identities
    assert ("mcp", "local:docs", "search") in identities
    assert ("mcp", "local:empty") in snapshot.namespaces
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


@pytest.mark.parametrize(
    ("builtin_inventory", "external_catalog"),
    [
        ({}, []),
        (
            {
                "tools": [
                    {
                        "name": "search_media",
                        "description": "Search local media.",
                    }
                ]
            },
            [],
        ),
        ({"tools": []}, [{"profile_id": "docs", "is_connected": False}]),
        (
            {"tools": []},
            [
                {
                    "profile_id": "docs",
                    "is_connected": False,
                    "discovery_snapshot": {
                        "tools": [
                            {"name": "search"},
                            {"name": "SEARCH"},
                        ]
                    },
                }
            ],
        ),
        (
            {"tools": []},
            [
                {
                    "profile_id": "docs",
                    "is_connected": False,
                    "discovery_snapshot": {"tools": []},
                },
                {
                    "profile_id": "DOCS",
                    "is_connected": False,
                    "discovery_snapshot": {"tools": []},
                },
            ],
        ),
    ],
)
def test_v1_registry_rejects_missing_or_partial_raw_provider_inventory(
    tmp_path: Path,
    builtin_inventory: dict[str, object],
    external_catalog: list[dict[str, object]],
) -> None:
    local_service = _LocalControlService(builtin_inventory, external_catalog)
    registry = PermissionInventoryRegistry.v1(
        local_service,
        fallback_root=tmp_path,
    )

    with pytest.raises(ToolPackError, match=r"inventory_incomplete$"):
        capture_v1_inventory(registry)
