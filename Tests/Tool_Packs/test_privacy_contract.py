"""Cross-surface privacy ratchets for portable Tool Pack V1."""

from __future__ import annotations

from dataclasses import asdict
from io import BytesIO
from types import MappingProxyType

import pytest

from tldw_chatbook.MCP.hub_tool_catalog import HubTool
from tldw_chatbook.MCP.permission_store import PermissionStoreSnapshot, definition_hash
from tldw_chatbook.Tool_Packs import export as export_module
from tldw_chatbook.Tool_Packs.catalog_snapshot import (
    PermissionInventoryAdapter,
    PermissionInventoryRegistry,
)
from tldw_chatbook.Tool_Packs.contracts import (
    PortableToolRule,
    ToolPackError,
    ToolPackManifest,
    ToolProfilePayload,
)
from tldw_chatbook.Tool_Packs.export import ToolPackExportService, write_tool_pack_archive


_SENTINELS = (
    "PRIVATE-CREDENTIAL-7de3",
    "PRIVATE-COMMAND-91ad",
    "PRIVATE-ARGUMENTS-633b",
    "PRIVATE-ENVIRONMENT-a1c4",
    "https://private-endpoint.invalid/9f0e",
    "/Users/private/workspace-320d",
    r"C:\\private\\workspace-84b2",
    "PRIVATE-PERSONA-f66a",
    "PRIVATE-RECEIPT-a01c",
    "PRIVATE-APPROVAL-4b7d",
    "PRIVATE-SCHEMA-PROSE-6e9f",
)
_FORBIDDEN_KEYS = {
    "command",
    "args",
    "env",
    "endpoint",
    "url",
    "credential",
    "secret",
    "api_key",
    "workspace_id",
    "persona_id",
    "session_grant",
    "approval_history",
    "description",
    "input_schema",
    "inputSchema",
    "receipt_id",
    "receipt_digest",
    "executable",
    "plugin",
    "skill",
    "runtime_install",
}


def _recursive_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            nested
            for item in value.values()
            for nested in _recursive_keys(item)
        }
    if isinstance(value, (list, tuple)):
        return {nested for item in value for nested in _recursive_keys(item)}
    return set()


def _tool() -> HubTool:
    return HubTool(
        server_key="local:docs",
        server_label=_SENTINELS[0],
        source="local",
        name="search",
        description=f"Search prose {_SENTINELS[-1]}",
        input_schema={
            "type": "object",
            "credential": _SENTINELS[0],
            "command": _SENTINELS[1],
        },
        tags=("network",),
        stale=False,
        executable=True,
    )


class _Adapter(PermissionInventoryAdapter):
    namespace = "local:docs"
    complete = True

    def __init__(self, tool: HubTool) -> None:
        self.tool = tool

    def snapshot(self) -> tuple[HubTool, ...]:
        return (self.tool,)


class _Store:
    def __init__(self, tool: HubTool) -> None:
        profile = {
            "global_default": "ask",
            "servers": {
                "local:docs": {
                    "tools": {
                        "search": {
                            "state": "allow",
                            "definition_hash": definition_hash(
                                tool.description, tool.input_schema
                            ),
                        }
                    }
                }
            },
            "workspace_id": _SENTINELS[5],
            "persona_id": _SENTINELS[7],
            "session_grant": _SENTINELS[9],
            "approval_history": [_SENTINELS[9]],
            "receipt_id": _SENTINELS[8],
        }
        payload = {
            "schema_version": 1,
            "kill_switch": True,
            "profiles": {"default": profile},
            "server_configuration": {
                "credential": _SENTINELS[0],
                "command": _SENTINELS[1],
                "args": _SENTINELS[2],
                "env": _SENTINELS[3],
                "endpoint": _SENTINELS[4],
            },
        }
        self.snapshot = PermissionStoreSnapshot(
            MappingProxyType(payload), "sha256:" + "a" * 64, True
        )

    def read_snapshot_strict(self) -> PermissionStoreSnapshot:
        return self.snapshot


def test_export_review_manifest_profile_and_archive_are_policy_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local gates and tool prose may influence hashes but never travel."""
    tool = _tool()
    registry = PermissionInventoryRegistry(
        current_permission_namespaces=lambda: {"local:docs"}
    )
    registry.register(_Adapter(tool))
    monkeypatch.setattr(export_module, "capture_v1_inventory", lambda value: value.capture())

    review = ToolPackExportService(_Store(tool), registry).capture(
        profile_id="default",
        display_name="Safe policy",
        suggested_id="safe-policy",
    )
    sink = BytesIO()
    write_tool_pack_archive(review.snapshot, sink)

    portable = {
        "manifest": review.snapshot.manifest.to_dict(),
        "profile": review.snapshot.payload.to_dict(),
        "review": asdict(review),
    }
    assert not (_recursive_keys(portable) & _FORBIDDEN_KEYS)
    rendered = repr(portable).encode() + sink.getvalue()
    for sentinel in _SENTINELS:
        assert sentinel.encode() not in rendered
    assert b"kill_switch" not in sink.getvalue()


@pytest.mark.parametrize(
    ("container", "field"),
    [
        ("manifest", "plugin"),
        ("manifest", "runtime_install"),
        ("profile", "skills"),
        ("profile", "executable"),
        ("tool", "command"),
        ("tool", "input_schema"),
        ("fallback", "endpoint"),
    ],
)
def test_v1_rejects_executable_composition_and_privacy_fields(
    container: str, field: str
) -> None:
    profile = {
        "schema": "tldw.tool-profile/v1",
        "fallbacks": [
            {"authority": "builtin", "server_key": "agent:builtin", "state": "ask"},
            {"authority": "mcp", "server_key": "*", "state": "ask"},
            {"authority": "mcp", "server_key": "local:docs", "state": "ask"},
        ],
        "tools": [
            {
                "authority": "mcp",
                "server_key": "local:docs",
                "tool_name": "search",
                "state": "deny",
            }
        ],
    }
    if container == "profile":
        profile[field] = _SENTINELS[0]
    elif container == "tool":
        profile["tools"][0][field] = _SENTINELS[0]
    elif container == "fallback":
        profile["fallbacks"][0][field] = _SENTINELS[0]
    else:
        raw = {
            "schema": "tldw.tool-pack/v1",
            "producer": {"name": "tldw_chatbook", "version": "1"},
            "required_features": [],
            "profile": {
                "suggested_id": "safe-policy",
                "display_name": "Safe policy",
                "payload": "profile/profile.json",
            },
            "files": [
                {
                    "path": "profile/profile.json",
                    "size": 1,
                    "sha256": "a" * 64,
                }
            ],
            "content_digest": "b" * 64,
            field: _SENTINELS[0],
        }

    with pytest.raises(ToolPackError) as caught:
        if container == "manifest":
            ToolPackManifest.from_dict(raw)
        else:
            ToolProfilePayload.from_dict(profile)
    assert caught.value.category in {"manifest_invalid", "payload_invalid"}


def test_private_dependency_failures_emit_only_stable_codes(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Failure messages are suitable for notifications and silent in logs."""
    class SensitiveStore:
        def read_snapshot_strict(self) -> PermissionStoreSnapshot:
            raise RuntimeError(" ".join(_SENTINELS))

    registry = PermissionInventoryRegistry(current_permission_namespaces=lambda: set())
    monkeypatch.setattr(export_module, "capture_v1_inventory", lambda value: value.capture())

    with pytest.raises(ToolPackError) as caught:
        ToolPackExportService(SensitiveStore(), registry).capture(
            profile_id="default",
            display_name="Safe policy",
            suggested_id="safe-policy",
        )

    public = str(caught.value)
    captured = capsys.readouterr()
    emitted = f"{public}\n{caplog.text}\n{captured.out}\n{captured.err}"
    assert public == "tool_pack.export.store_invalid"
    for sentinel in _SENTINELS:
        assert sentinel not in emitted


def test_portable_rule_schema_cannot_smuggle_runtime_install_state() -> None:
    rule = PortableToolRule("mcp", "local:docs", "search", "deny", None)

    assert set(rule.to_dict()) == {"authority", "server_key", "tool_name", "state"}
