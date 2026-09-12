"""Architecture ownership tripwires for portable Tool Pack V1."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tldw_chatbook.MCP.hub_tool_catalog import HubTool
from tldw_chatbook.Tool_Packs.catalog_snapshot import (
    NONPORTABLE_CATEGORIES,
    PermissionInventoryAdapter,
    PermissionInventoryRegistry,
)
from tldw_chatbook.Tool_Packs.contracts import ToolPackError


_ROOT = Path(__file__).parents[2]
_TOOL_PACKS = _ROOT / "tldw_chatbook" / "Tool_Packs"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_workspace_registry_depends_on_guard_protocol_not_tool_pack_internals() -> None:
    path = _ROOT / "tldw_chatbook" / "Workspaces" / "registry_service.py"
    source = path.read_text(encoding="utf-8")

    assert "class WorkspaceToolProfileGuard(Protocol):" in source
    assert not any("Tool_Packs" in name for name in _imports(path))


def test_settings_owns_lifecycle_orchestration_but_not_policy_store_writes() -> None:
    path = _ROOT / "tldw_chatbook" / "UI" / "Screens" / "settings_screen.py"
    source = path.read_text(encoding="utf-8")
    imports = _imports(path)

    assert any("Tool_Packs.service" in name for name in imports)
    assert "tool_pack_service" in source
    assert not any("MCP.permission_store" in name for name in imports)
    for forbidden in (
        "install_profile_if_absent(",
        "update_imported_profile(",
        "replace_profile_with_tombstone(",
        ".set_global_default(",
        ".set_server_default(",
        ".set_tool_state(",
    ):
        assert forbidden not in source


def test_mcp_permissions_owns_rule_edits_but_not_pack_lifecycle() -> None:
    workbench = _ROOT / "tldw_chatbook" / "UI" / "MCP_Modules" / "mcp_workbench.py"
    canvas = _ROOT / "tldw_chatbook" / "UI" / "MCP_Modules" / "mcp_permissions_mode.py"
    source = workbench.read_text(encoding="utf-8")

    assert "service.set_global_default" in source
    assert "service.set_server_default" in source
    assert "service.set_tool_state" in source
    assert not any("Tool_Packs" in name for name in _imports(workbench) | _imports(canvas))
    for forbidden in (
        "ToolPackService",
        ".inspect_import(",
        ".import_unbound(",
        ".review_first_bind(",
        ".bind_profile(",
        ".remove_profile(",
    ):
        assert forbidden not in source


def test_tool_packs_do_not_import_actor_character_or_persona_pack_internals() -> None:
    forbidden = ("actor", "character_pack", "persona")
    imported = {
        name.casefold()
        for path in _TOOL_PACKS.glob("*.py")
        for name in _imports(path)
    }

    assert not {
        name for name in imported if any(marker in name for marker in forbidden)
    }


class _Adapter(PermissionInventoryAdapter):
    complete = True

    def __init__(self, namespace: str) -> None:
        self.namespace = namespace

    def snapshot(self) -> tuple[HubTool, ...]:
        return ()


def test_every_permission_namespace_is_registered_or_explicitly_excluded() -> None:
    """A new governed namespace blocks export until its owner classifies it."""
    assert set(NONPORTABLE_CATEGORIES) == {
        "display_only_server_source",
        "library_capability",
        "managed_skill_approval",
        "runtime_orchestration",
        "skills",
    }
    namespaces = {"agent:builtin", "local:docs", "local:new-provider"}
    registry = PermissionInventoryRegistry(
        current_permission_namespaces=lambda: namespaces,
        excluded_counts=lambda: {name: 0 for name in NONPORTABLE_CATEGORIES},
    )
    registry.register(_Adapter("agent:builtin"))
    registry.register(_Adapter("local:docs"))

    with pytest.raises(ToolPackError, match=r"^tool_pack\.export\.inventory_incomplete$"):
        registry.capture()

    registry.register(_Adapter("local:new-provider"))
    assert {identity for identity in registry.capture().namespaces} == {
        ("builtin", "agent:builtin"),
        ("mcp", "local:docs"),
        ("mcp", "local:new-provider"),
    }
