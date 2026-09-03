"""Pinned bounds and structural performance evidence for Tool Pack V1."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import socket
import time
import tracemalloc
from types import MappingProxyType

import pytest

from tldw_chatbook.MCP.hub_tool_catalog import HubTool
from tldw_chatbook.MCP.permission_store import (
    MCPPermissionStore,
    PermissionStoreSnapshot,
    ProfileMutationError,
    profile_policy_digest,
)
from tldw_chatbook.MCP import permission_store as permission_store_module
from tldw_chatbook.Tool_Packs import export as export_module
from tldw_chatbook.Tool_Packs import importer as importer_module
from tldw_chatbook.Tool_Packs.catalog_snapshot import (
    PermissionInventoryAdapter,
    PermissionInventoryRegistry,
)
from tldw_chatbook.Tool_Packs.contracts import (
    MAX_FALLBACKS,
    MAX_JSON_NODES,
    MAX_SERVERS,
    MAX_TOOLS,
    ToolPackError,
    ToolProfilePayload,
    strict_json_object,
)
from tldw_chatbook.Tool_Packs.export import ToolPackExportService, write_tool_pack_archive
from tldw_chatbook.Tool_Packs.importer import ToolPackImportService
from tldw_chatbook.Tool_Packs.receipt_store import (
    MAX_RECEIPT_BYTES,
    MAX_RECEIPT_STORE_BYTES,
    ToolPackReceiptStore,
)


class _Adapter(PermissionInventoryAdapter):
    complete = True

    def __init__(self, namespace: str, tools: tuple[HubTool, ...]) -> None:
        self.namespace = namespace
        self.tools = tools
        self.calls = 0

    def snapshot(self) -> tuple[HubTool, ...]:
        self.calls += 1
        return self.tools


class _Store:
    def __init__(self) -> None:
        self.calls = 0
        self.snapshot = PermissionStoreSnapshot(
            MappingProxyType(
                {
                    "schema_version": 1,
                    "kill_switch": False,
                    "profiles": MappingProxyType(
                        {
                            "default": MappingProxyType(
                                {"global_default": "deny", "servers": MappingProxyType({})}
                            )
                        }
                    ),
                }
            ),
            "sha256:" + "a" * 64,
            True,
        )

    def read_snapshot_strict(self) -> PermissionStoreSnapshot:
        self.calls += 1
        return self.snapshot


def _registry_at_maximum() -> tuple[PermissionInventoryRegistry, tuple[_Adapter, ...]]:
    namespaces = ["agent:builtin"] + [f"local:s{index:03d}" for index in range(255)]
    tools = tuple(
        HubTool(
            server_key="local:s000",
            server_label="Server",
            source="local",
            name=f"tool-{index:04d}",
            description="Bounded tool",
            input_schema={"type": "object"},
            tags=(),
            stale=False,
            executable=True,
        )
        for index in range(MAX_TOOLS)
    )
    adapters = tuple(
        _Adapter(namespace, tools if namespace == "local:s000" else ())
        for namespace in namespaces
    )
    registry = PermissionInventoryRegistry(
        current_permission_namespaces=lambda: set(namespaces)
    )
    for adapter in adapters:
        registry.register(adapter)
    return registry, adapters


def test_maximum_inventory_round_trips_with_one_capture_per_source_and_no_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_property,
) -> None:
    registry, adapters = _registry_at_maximum()
    export_store = _Store()
    monkeypatch.setattr(export_module, "capture_v1_inventory", lambda value: value.capture())
    network_attempts = 0

    def reject_network(*_args, **_kwargs):
        nonlocal network_attempts
        network_attempts += 1
        raise AssertionError("Tool Pack review must not construct a network connection")

    monkeypatch.setattr(socket, "socket", reject_network)
    tracemalloc.start()
    started = time.perf_counter()
    review = ToolPackExportService(export_store, registry).capture(
        profile_id="default", display_name="Maximum policy", suggested_id="maximum"
    )
    sink = BytesIO()
    write_tool_pack_archive(review.snapshot, sink)
    export_elapsed = time.perf_counter() - started
    export_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    assert len(review.payload.tools) == MAX_TOOLS
    assert len(review.payload.fallbacks) == MAX_FALLBACKS
    assert len({(item.authority, item.server_key) for item in review.payload.fallbacks} - {("mcp", "*")}) == MAX_SERVERS
    assert export_store.calls == 1
    assert all(adapter.calls == 1 for adapter in adapters)

    archive_path = tmp_path / "maximum.tldw-tool-pack"
    archive_path.write_bytes(sink.getvalue())
    import_store = _Store()
    inventory = registry.capture()
    inventory_reads = 0
    archive_reads = 0
    reference_reads = 0
    real_archive_read = importer_module._read_regular_archive

    def capture_inventory(_value):
        nonlocal inventory_reads
        inventory_reads += 1
        return inventory

    def read_archive(path: Path) -> bytes:
        nonlocal archive_reads
        archive_reads += 1
        return real_archive_read(path)

    def references(_profile_id: str) -> bool:
        nonlocal reference_reads
        reference_reads += 1
        return False

    monkeypatch.setattr(importer_module, "capture_v1_inventory", capture_inventory)
    monkeypatch.setattr(importer_module, "_read_regular_archive", read_archive)
    tracemalloc.start()
    started = time.perf_counter()
    imported = ToolPackImportService(
        import_store,
        registry,
        references,
        now=lambda: datetime(2026, 9, 2, tzinfo=timezone.utc),
    ).inspect_archive(archive_path, destination_id="maximum")
    import_elapsed = time.perf_counter() - started
    import_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    assert len(imported.matched) == MAX_TOOLS
    assert import_store.calls == inventory_reads == archive_reads == reference_reads == 1
    assert network_attempts == 0
    record_property("tool_pack_max_export_seconds", export_elapsed)
    record_property("tool_pack_max_export_peak_bytes", export_peak)
    record_property("tool_pack_max_import_seconds", import_elapsed)
    record_property("tool_pack_max_import_peak_bytes", import_peak)


def test_count_and_json_boundaries_accept_exact_and_reject_one_over() -> None:
    fallbacks = [
        {"authority": "builtin", "server_key": "agent:builtin", "state": "ask"},
        {"authority": "mcp", "server_key": "*", "state": "ask"},
        *[
            {"authority": "mcp", "server_key": f"local:s{index:03d}", "state": "ask"}
            for index in range(255)
        ],
    ]
    tools = [
        {
            "authority": "mcp",
            "server_key": "local:s000",
            "tool_name": f"tool-{index:04d}",
            "state": "deny",
        }
        for index in range(MAX_TOOLS)
    ]
    exact = {"schema": "tldw.tool-profile/v1", "fallbacks": fallbacks, "tools": tools}

    assert len(ToolProfilePayload.from_dict(exact).tools) == MAX_TOOLS
    with pytest.raises(ToolPackError, match=r"^tool_pack\.import\.too_large$"):
        ToolProfilePayload.from_dict({**exact, "tools": [*tools, tools[-1]]})
    with pytest.raises(ToolPackError, match=r"^tool_pack\.import\.too_large$"):
        ToolProfilePayload.from_dict(
            {
                **exact,
                "fallbacks": [
                    *fallbacks,
                    {"authority": "mcp", "server_key": "local:overflow", "state": "ask"},
                ],
            }
        )

    exact_nodes = {str(index): None for index in range(MAX_JSON_NODES - 1)}
    encoded = json.dumps(exact_nodes, separators=(",", ":")).encode()
    assert len(strict_json_object(encoded, category="payload_invalid", max_bytes=len(encoded))) == MAX_JSON_NODES - 1
    one_over = json.dumps([None] * MAX_JSON_NODES, separators=(",", ":")).encode()
    with pytest.raises(ToolPackError, match=r"^tool_pack\.import\.payload_invalid$"):
        strict_json_object(one_over, category="payload_invalid", max_bytes=len(one_over))


def test_archive_and_receipt_capacity_accept_exact_and_reject_one_over(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "capacity.tldw-tool-pack"
    archive.write_bytes(b"x" * importer_module._MAX_ARCHIVE_BYTES)
    assert len(importer_module._read_regular_archive(archive)) == importer_module._MAX_ARCHIVE_BYTES
    archive.write_bytes(b"x" * (importer_module._MAX_ARCHIVE_BYTES + 1))
    with pytest.raises(ToolPackError, match=r"^tool_pack\.import\.too_large$"):
        importer_module._read_regular_archive(archive)

    receipts = ToolPackReceiptStore(tmp_path / "receipts")
    reservations = [receipts.reserve(MAX_RECEIPT_BYTES) for _ in range(8)]
    assert len(reservations) * MAX_RECEIPT_BYTES == MAX_RECEIPT_STORE_BYTES
    with pytest.raises(ToolPackError, match=r"^tool_pack\.import\.capacity_exceeded$"):
        receipts.reserve(1)
    for reservation in reservations:
        reservation.release()
    with pytest.raises(ToolPackError, match=r"^tool_pack\.import\.capacity_exceeded$"):
        receipts.reserve(MAX_RECEIPT_BYTES + 1)


def _imported_profile() -> dict[str, object]:
    profile: dict[str, object] = {
        "global_default": "ask",
        "servers": {"agent:builtin": {"default": "ask"}},
        "profile_kind": "tool_pack_imported",
        "tool_pack_lifecycle": {
            "schema": "tldw.tool-pack-lifecycle/v1",
            "origin": "imported",
            "pack_digest": "b" * 64,
            "imported_at": "2026-09-02T00:00:00Z",
            "first_bind_confirmation_required": True,
            "receipt_id": "tp-" + "c" * 32,
            "receipt_digest": "d" * 64,
            "counts": {"matched": 0, "omitted": 0, "pending_deny": 0},
            "policy_digest": "0" * 64,
            "revision": 1,
        },
    }
    profile["tool_pack_lifecycle"]["policy_digest"] = profile_policy_digest(profile)  # type: ignore[index]
    return profile


def test_permission_store_accepts_exact_projected_cap_and_rejects_one_less(
    tmp_path: Path,
) -> None:
    candidate = _imported_profile()
    probe = MCPPermissionStore(tmp_path / "probe.json")
    projected = probe.load()
    projected["profiles"]["research"] = candidate
    exact_bytes = permission_store_module._projected_store_size(projected)

    exact = MCPPermissionStore(tmp_path / "exact.json")
    exact.install_profile_if_absent(
        "research",
        candidate,
        expected_generation=exact.read_snapshot_strict().generation,
        max_profiles=2,
        max_store_bytes=exact_bytes,
    )
    assert "research" in exact.read_snapshot_strict().payload["profiles"]

    rejected = MCPPermissionStore(tmp_path / "rejected.json")
    with pytest.raises(ProfileMutationError, match="store_bytes_limit"):
        rejected.install_profile_if_absent(
            "research",
            candidate,
            expected_generation=rejected.read_snapshot_strict().generation,
            max_profiles=2,
            max_store_bytes=exact_bytes - 1,
        )
