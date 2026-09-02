"""Exact matching and immutable review tests for Tool Pack inspection."""

from dataclasses import FrozenInstanceError, fields
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from types import MappingProxyType

import pytest

from tldw_chatbook.MCP.hub_tool_catalog import HubTool
from tldw_chatbook.MCP.permission_store import PermissionStoreSnapshot
from tldw_chatbook.Tool_Packs import importer as importer_module
from tldw_chatbook.Tool_Packs.catalog_snapshot import (
    PermissionInventorySnapshot,
    PermissionInventoryTool,
)
from tldw_chatbook.Tool_Packs.contracts import (
    TOOL_PROFILE_SCHEMA,
    PortableFallback,
    PortableToolRule,
    ToolPackError,
    ToolProfilePayload,
    portable_contract_sha256,
)
from tldw_chatbook.Tool_Packs.export import (
    ToolPackExportSnapshot,
    _manifest_for,
    write_tool_pack_archive,
)

from tldw_chatbook.Tool_Packs.importer import (
    MappedToolRule,
    ServerMapping,
    ToolPackImportReview,
    ToolPackImportService,
)


def test_importer_public_api_is_available() -> None:
    assert ServerMapping
    assert MappedToolRule
    assert ToolPackImportReview
    assert ToolPackImportService


def _tool(
    *,
    server_key: str = "local:docs",
    name: str = "search",
    tags: tuple[str, ...] = (),
    executable: bool = True,
) -> HubTool:
    return HubTool(
        server_key=server_key,
        server_label="ignored display label",
        source="local",
        name=name,
        description=f"{name} description",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        tags=tags,
        stale=not executable,
        executable=executable,
    )


def _inventory(*tools: HubTool) -> PermissionInventorySnapshot:
    entries = tuple(
        PermissionInventoryTool(
            "builtin" if tool.server_key == "agent:builtin" else "mcp",
            tool,
            portable_contract_sha256(tool),
        )
        for tool in tools
    )
    namespaces = tuple(
        sorted({(entry.authority, entry.tool.server_key) for entry in entries})
    )
    return PermissionInventorySnapshot(entries, namespaces, (), "inventory-digest")


class _Store:
    def __init__(self, profiles: dict[str, object] | None = None) -> None:
        self.calls = 0
        self.snapshot = PermissionStoreSnapshot(
            payload=MappingProxyType(
                {
                    "schema_version": 1,
                    "kill_switch": False,
                    "profiles": MappingProxyType(
                        profiles
                        or {
                            "default": MappingProxyType(
                                {
                                    "global_default": "ask",
                                    "servers": MappingProxyType({}),
                                }
                            )
                        }
                    ),
                }
            ),
            generation="sha256:" + "a" * 64,
            file_exists=True,
        )

    def read_snapshot_strict(self) -> PermissionStoreSnapshot:
        self.calls += 1
        return self.snapshot


def _archive(
    path: Path,
    *,
    rules: tuple[PortableToolRule, ...],
) -> Path:
    fallbacks = {
        ("mcp", "*"),
        ("builtin", "agent:builtin"),
        *((rule.authority, rule.server_key) for rule in rules),
    }
    payload = ToolProfilePayload(
        TOOL_PROFILE_SCHEMA,
        tuple(
            PortableFallback(authority, server_key, "ask")
            for authority, server_key in sorted(fallbacks)
        ),
        tuple(
            sorted(
                rules,
                key=lambda rule: (rule.authority, rule.server_key, rule.tool_name),
            )
        ),
    )
    manifest = _manifest_for(
        payload,
        suggested_id="research",
        display_name="Research",
        producer_name="test-producer",
        producer_version="1",
    )
    sink = BytesIO()
    write_tool_pack_archive(ToolPackExportSnapshot(manifest, payload), sink)
    path.write_bytes(sink.getvalue())
    return path


@pytest.fixture
def fixed_now() -> datetime:
    return datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)


def test_inspects_canonical_export_with_one_strict_snapshot_and_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixed_now: datetime,
) -> None:
    tool = _tool()
    inventory = _inventory(tool)
    captures = 0

    def capture(value: object) -> PermissionInventorySnapshot:
        nonlocal captures
        captures += 1
        assert value is inventory
        return inventory

    monkeypatch.setattr(importer_module, "capture_v1_inventory", capture)
    store = _Store()
    references: list[str] = []
    archive = _archive(
        tmp_path / "valid.tldw-tool-pack",
        rules=(
            PortableToolRule(
                "mcp", "local:docs", "search", "allow", portable_contract_sha256(tool)
            ),
        ),
    )
    service = ToolPackImportService(
        store,
        inventory,
        lambda profile_id: references.append(profile_id) or False,
        now=lambda: fixed_now,
    )

    review = service.inspect_archive(archive, destination_id="Research")

    assert review.destination_id == "research"
    assert review.store_generation == store.snapshot.generation
    assert review.inventory_digest == inventory.digest
    assert review.display_name == "Research"
    assert review.producer == ("test-producer", "1")
    assert len(review.content_digest) == 64
    assert ("mcp", "local:docs") in {
        (fallback.authority, fallback.server_key) for fallback in review.fallbacks
    }
    assert review.matched[0].destination_identity == ("mcp", "local:docs", "search")
    assert store.calls == captures == 1
    assert references == ["research"]


def test_explicit_external_mapping_matches_exact_contract_and_maps_pending_deny(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixed_now: datetime,
) -> None:
    destination_tool = _tool(server_key="local:destination")
    inventory = _inventory(destination_tool)
    monkeypatch.setattr(
        importer_module, "capture_v1_inventory", lambda _registry: inventory
    )
    archive = _archive(
        tmp_path / "mapped.tldw-tool-pack",
        rules=(
            PortableToolRule(
                "mcp",
                "source:mcp",
                "blocked",
                "deny",
                None,
            ),
            PortableToolRule(
                "mcp",
                "source:mcp",
                "search",
                "allow",
                portable_contract_sha256(destination_tool),
            ),
        ),
    )
    review = ToolPackImportService(
        _Store(), inventory, lambda _profile_id: False, now=lambda: fixed_now
    ).inspect_archive(
        archive,
        destination_id="research",
        mappings=(ServerMapping("source:mcp", "local:destination"),),
    )

    assert review.matched[0].destination_identity == (
        "mcp",
        "local:destination",
        "search",
    )
    assert review.pending_denies == (
        PortableToolRule("mcp", "local:destination", "blocked", "deny", None),
    )


def test_risk_tag_only_change_is_changed_and_allow_is_omitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixed_now: datetime,
) -> None:
    source_tool = _tool()
    destination_tool = _tool(tags=("high-risk",))
    inventory = _inventory(destination_tool)
    monkeypatch.setattr(
        importer_module, "capture_v1_inventory", lambda _registry: inventory
    )
    rule = PortableToolRule(
        "mcp",
        "local:docs",
        "search",
        "allow",
        portable_contract_sha256(source_tool),
    )
    review = ToolPackImportService(
        _Store(), inventory, lambda _profile_id: False, now=lambda: fixed_now
    ).inspect_archive(
        _archive(tmp_path / "changed.tldw-tool-pack", rules=(rule,)),
        destination_id="research",
    )

    assert review.matched == ()
    assert review.changed == (rule,)
    assert review.omitted_allow_ask == (rule,)


def test_labels_and_connection_state_do_not_replace_raw_contract_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixed_now: datetime,
) -> None:
    destination_tool = _tool(executable=False)
    inventory = _inventory(destination_tool)
    monkeypatch.setattr(
        importer_module, "capture_v1_inventory", lambda _registry: inventory
    )
    rule = PortableToolRule(
        "mcp",
        "local:docs",
        "search",
        "ask",
        portable_contract_sha256(destination_tool),
    )
    review = ToolPackImportService(
        _Store(), inventory, lambda _profile_id: False, now=lambda: fixed_now
    ).inspect_archive(
        _archive(tmp_path / "cached.tldw-tool-pack", rules=(rule,)),
        destination_id="research",
    )

    assert review.matched[0].source_rule.tool_name == "search"
    assert review.matched[0].destination_connected is False


def test_missing_allow_and_ask_are_omitted_while_missing_deny_is_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixed_now: datetime,
) -> None:
    inventory = _inventory()
    monkeypatch.setattr(
        importer_module, "capture_v1_inventory", lambda _registry: inventory
    )
    fingerprint = "1" * 64
    rules = (
        PortableToolRule("mcp", "source:mcp", "allow", "allow", fingerprint),
        PortableToolRule("mcp", "source:mcp", "ask", "ask", fingerprint),
        PortableToolRule("mcp", "source:mcp", "deny", "deny", None),
    )
    review = ToolPackImportService(
        _Store(), inventory, lambda _profile_id: False, now=lambda: fixed_now
    ).inspect_archive(
        _archive(tmp_path / "missing.tldw-tool-pack", rules=rules),
        destination_id="research",
    )

    assert review.missing == tuple(sorted(rules, key=lambda item: item.tool_name))
    assert {rule.state for rule in review.omitted_allow_ask} == {"allow", "ask"}
    assert tuple(rule.state for rule in review.pending_denies) == ("deny",)


@pytest.mark.parametrize("destination_id", ["default", "WS-123", "...", "🔥"])
def test_rejects_reserved_or_invalid_destination_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixed_now: datetime,
    destination_id: str,
) -> None:
    tool = _tool()
    inventory = _inventory(tool)
    monkeypatch.setattr(
        importer_module, "capture_v1_inventory", lambda _registry: inventory
    )
    archive = _archive(
        tmp_path / "id.tldw-tool-pack",
        rules=(
            PortableToolRule(
                "mcp",
                "local:docs",
                "search",
                "allow",
                portable_contract_sha256(tool),
            ),
        ),
    )

    with pytest.raises(ToolPackError) as raised:
        ToolPackImportService(
            _Store(), inventory, lambda _profile_id: False, now=lambda: fixed_now
        ).inspect_archive(archive, destination_id=destination_id)

    assert raised.value.category == "mapping_invalid"


@pytest.mark.parametrize("existing", ["research", "RESEARCH"])
def test_rejects_exact_and_casefold_profile_collisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixed_now: datetime,
    existing: str,
) -> None:
    tool = _tool()
    inventory = _inventory(tool)
    monkeypatch.setattr(
        importer_module, "capture_v1_inventory", lambda _registry: inventory
    )
    archive = _archive(
        tmp_path / "collision.tldw-tool-pack",
        rules=(
            PortableToolRule(
                "mcp",
                "local:docs",
                "search",
                "allow",
                portable_contract_sha256(tool),
            ),
        ),
    )
    existing_profile = MappingProxyType(
        {"global_default": "deny", "servers": MappingProxyType({})}
    )

    with pytest.raises(ToolPackError) as raised:
        ToolPackImportService(
            _Store({existing: existing_profile}),
            inventory,
            lambda _profile_id: False,
            now=lambda: fixed_now,
        ).inspect_archive(archive, destination_id="research")

    assert raised.value.category == "destination_referenced"


def test_rejects_active_archived_or_dangling_reference_reported_by_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixed_now: datetime,
) -> None:
    tool = _tool()
    inventory = _inventory(tool)
    monkeypatch.setattr(
        importer_module, "capture_v1_inventory", lambda _registry: inventory
    )
    archive = _archive(
        tmp_path / "referenced.tldw-tool-pack",
        rules=(
            PortableToolRule(
                "mcp", "local:docs", "search", "deny", portable_contract_sha256(tool)
            ),
        ),
    )
    checked: list[str] = []

    with pytest.raises(ToolPackError) as raised:
        ToolPackImportService(
            _Store(),
            inventory,
            lambda profile_id: checked.append(profile_id) or True,
            now=lambda: fixed_now,
        ).inspect_archive(archive, destination_id="research")

    assert raised.value.category == "destination_referenced"
    assert checked == ["research"]


def test_manual_mappings_are_capped_and_external_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixed_now: datetime,
) -> None:
    destination = _tool(server_key="local:destination")
    inventory = _inventory(destination)
    monkeypatch.setattr(
        importer_module, "capture_v1_inventory", lambda _registry: inventory
    )
    archive = _archive(
        tmp_path / "mapping-invalid.tldw-tool-pack",
        rules=(
            PortableToolRule(
                "mcp",
                "source:mcp",
                "search",
                "allow",
                portable_contract_sha256(destination),
            ),
        ),
    )
    service = ToolPackImportService(
        _Store(), inventory, lambda _profile_id: False, now=lambda: fixed_now
    )
    over_cap = tuple(
        ServerMapping(f"source:{index}", "local:destination") for index in range(257)
    )

    for mappings, category in (
        ((ServerMapping("source:mcp", "local:__local__"),), "mapping_invalid"),
        ((ServerMapping("source:mcp", "local:missing"),), "mapping_invalid"),
        (over_cap, "too_large"),
    ):
        with pytest.raises(ToolPackError) as raised:
            service.inspect_archive(
                archive, destination_id="research", mappings=mappings
            )
        assert raised.value.category == category


def test_mapping_rejects_duplicate_resulting_tool_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixed_now: datetime,
) -> None:
    destination = _tool(server_key="local:destination")
    inventory = _inventory(destination)
    monkeypatch.setattr(
        importer_module, "capture_v1_inventory", lambda _registry: inventory
    )
    fingerprint = portable_contract_sha256(destination)
    archive = _archive(
        tmp_path / "mapping-collision.tldw-tool-pack",
        rules=(
            PortableToolRule("mcp", "local:destination", "search", "ask", fingerprint),
            PortableToolRule("mcp", "source:mcp", "search", "allow", fingerprint),
        ),
    )

    with pytest.raises(ToolPackError) as raised:
        ToolPackImportService(
            _Store(), inventory, lambda _profile_id: False, now=lambda: fixed_now
        ).inspect_archive(
            archive,
            destination_id="research",
            mappings=(ServerMapping("source:mcp", "local:destination"),),
        )

    assert raised.value.category == "identity_duplicate"


def test_review_is_deeply_immutable_and_expires_at_exactly_fifteen_minutes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixed_now: datetime,
) -> None:
    tool = _tool()
    inventory = _inventory(tool)
    monkeypatch.setattr(
        importer_module, "capture_v1_inventory", lambda _registry: inventory
    )
    review = ToolPackImportService(
        _Store(), inventory, lambda _profile_id: False, now=lambda: fixed_now
    ).inspect_archive(
        _archive(
            tmp_path / "immutable.tldw-tool-pack",
            rules=(
                PortableToolRule(
                    "mcp",
                    "local:docs",
                    "search",
                    "allow",
                    portable_contract_sha256(tool),
                ),
            ),
        ),
        destination_id="Research Tools",
    )

    assert review.destination_id == "research-tools"
    assert review.expires_at == fixed_now + timedelta(minutes=15)
    assert "callback" not in {field.name for field in fields(review)}
    assert "secret" not in repr(review).casefold()
    with pytest.raises(FrozenInstanceError):
        review.destination_id = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        review.matched[0].destination_identity[0] = "changed"  # type: ignore[index]


def test_reference_service_is_queried_once_with_archived_rows_included(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixed_now: datetime,
) -> None:
    tool = _tool()
    inventory = _inventory(tool)
    monkeypatch.setattr(
        importer_module, "capture_v1_inventory", lambda _registry: inventory
    )

    class References:
        def __init__(self) -> None:
            self.calls: list[tuple[str, bool]] = []

        def references_profile(
            self, profile_id: str, *, include_archived: bool
        ) -> bool:
            self.calls.append((profile_id, include_archived))
            return False

    references = References()
    ToolPackImportService(
        _Store(), inventory, references, now=lambda: fixed_now
    ).inspect_archive(
        _archive(
            tmp_path / "references.tldw-tool-pack",
            rules=(
                PortableToolRule(
                    "mcp",
                    "local:docs",
                    "search",
                    "allow",
                    portable_contract_sha256(tool),
                ),
            ),
        ),
        destination_id="research",
    )

    assert references.calls == [("research", True)]


def test_two_source_servers_cannot_map_to_one_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixed_now: datetime,
) -> None:
    first = _tool(server_key="local:destination", name="first")
    second = _tool(server_key="local:destination", name="second")
    inventory = _inventory(first, second)
    monkeypatch.setattr(
        importer_module, "capture_v1_inventory", lambda _registry: inventory
    )
    archive = _archive(
        tmp_path / "many-to-one.tldw-tool-pack",
        rules=(
            PortableToolRule(
                "mcp", "source:first", "first", "ask", portable_contract_sha256(first)
            ),
            PortableToolRule(
                "mcp",
                "source:second",
                "second",
                "ask",
                portable_contract_sha256(second),
            ),
        ),
    )

    with pytest.raises(ToolPackError) as raised:
        ToolPackImportService(
            _Store(), inventory, lambda _profile_id: False, now=lambda: fixed_now
        ).inspect_archive(
            archive,
            destination_id="research",
            mappings=(
                ServerMapping("source:first", "local:destination"),
                ServerMapping("source:second", "local:destination"),
            ),
        )

    assert raised.value.category == "mapping_invalid"
